"""
Skills 层 — AI 可调用的原子操作。
每个 skill 是纯函数：(df, **params) -> SkillResult
不依赖任何外部状态，独立可测。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import pandas as pd

# ── 结果数据类 ────────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    df: pd.DataFrame
    summary: str
    chart_option: Optional[dict] = None
    error: Optional[str] = None


# ── 条件解析 ──────────────────────────────────────────────────────────────────

_CN_NUM = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_ZH_OPS = [
    ("大于等于", ">="), ("小于等于", "<="), ("不等于", "!="), ("不是", "!="),
    ("大于", ">"), ("小于", "<"), ("等于", "=="), ("为", "=="),
]


def _cn2int(s: str) -> int:
    return _CN_NUM.get(s, -1) if not s.isdigit() else int(s)


def _match_col(hint: str, df: pd.DataFrame) -> Optional[str]:
    cols = sorted(df.columns, key=len, reverse=True)
    hint_l = hint.lower()
    for col in cols:
        if col.lower() == hint_l:
            return col
    return None


def _apply_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    cond = condition.strip()
    if not cond:
        return df

    # 1. 包含
    m = re.match(r"^(.+?)\s*包含\s*(.+)$", cond)
    if m:
        col_hint, val = m.group(1).strip(), m.group(2).strip()
        col = _match_col(col_hint, df)
        if col:
            return df[df[col].astype(str).str.contains(re.escape(val), na=False)]

    # 2. Top-N 极值
    m = re.match(
        r"^(.+?)(?:(最大)|(最小))的?(\d+|[一两二三四五六七八九十]+)(?:行|条|个)?$", cond
    )
    if m:
        col_hint = m.group(1).strip()
        is_max = bool(m.group(2))
        n = _cn2int(m.group(4))
        col = _match_col(col_hint, df)
        if col and n > 0:
            return df.nlargest(n, col) if is_max else df.nsmallest(n, col)

    # 3. 单极值
    m = re.match(r"^(.+?)(最大|最小)(?:的行|的记录)?$", cond)
    if m:
        col_hint, extreme = m.group(1).strip(), m.group(2)
        col = _match_col(col_hint, df)
        if col:
            val = df[col].max() if extreme == "最大" else df[col].min()
            return df[df[col] == val]

    # 4. 中文运算符
    for zh_op, sym in _ZH_OPS:
        if zh_op in cond:
            parts = cond.split(zh_op, 1)
            col_hint, raw_val = parts[0].strip(), parts[1].strip()
            col = _match_col(col_hint, df)
            if col:
                try:
                    num = float(raw_val)
                    return df.query(f"`{col}` {sym} {num}")
                except ValueError:
                    if sym == "==":
                        return df[df[col].astype(str) == raw_val]

    # 5. 符号运算符
    sym_ops = [">=", "<=", "!=", "==", ">", "<"]
    for op in sym_ops:
        if op in cond:
            parts = cond.split(op, 1)
            col_hint, raw_val = parts[0].strip(), parts[1].strip()
            col = _match_col(col_hint, df)
            if col:
                try:
                    num = float(raw_val)
                    return df.query(f"`{col}` {op} {num}")
                except ValueError:
                    if op == "==":
                        return df[df[col].astype(str) == raw_val]

    return df


# ── Skills ────────────────────────────────────────────────────────────────────

def skill_delete_columns(df: pd.DataFrame, columns: list) -> SkillResult:
    if not columns:
        return SkillResult(df, "", error="columns 不能为空")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return SkillResult(df, "", error=f"列不存在：{missing}，可用列：{list(df.columns)}")
    result = df.drop(columns=columns)
    return SkillResult(result, f"已删除列：{columns}（剩余 {len(result.columns)} 列）")


def skill_delete_rows(df: pd.DataFrame, condition: str) -> SkillResult:
    try:
        to_delete = _apply_condition(df, condition)
        if to_delete is df:
            return SkillResult(df, "", error=f"无法解析条件：{condition!r}")
        keep_idx = df.index.difference(to_delete.index)
        result = df.loc[keep_idx].reset_index(drop=True)
        return SkillResult(result, f"已删除 {len(to_delete)} 行（条件：{condition}），剩余 {len(result)} 行")
    except Exception as e:
        return SkillResult(df, "", error=str(e))


def skill_filter_rows(df: pd.DataFrame, condition: str) -> SkillResult:
    if not condition.strip():
        return SkillResult(df, f"无筛选条件，返回全部 {len(df)} 行")
    try:
        result = _apply_condition(df, condition)
        return SkillResult(result.reset_index(drop=True),
                           f"筛选条件「{condition}」，找到 {len(result)} 行（原始 {len(df)} 行）")
    except Exception as e:
        return SkillResult(df, "", error=str(e))


def skill_fill_missing(
    df: pd.DataFrame,
    method: str = "mean",
    columns: Optional[list] = None,
) -> SkillResult:
    result = df.copy()
    target_cols = columns if columns else list(df.columns)
    filled = 0
    for col in target_cols:
        if col not in result.columns:
            continue
        null_cnt = int(result[col].isnull().sum())
        if null_cnt == 0:
            continue
        if method == "mean" and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(result[col].mean())
        elif method == "median" and pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].fillna(result[col].median())
        elif method == "zero":
            result[col] = result[col].fillna(0)
        elif method == "ffill":
            result[col] = result[col].ffill()
        elif method == "bfill":
            result[col] = result[col].bfill()
        else:
            result[col] = result[col].fillna(method)
        filled += null_cnt
    return SkillResult(result, f"已填充缺失值 {filled} 处（方法：{method}）")


def skill_deduplicate(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> SkillResult:
    before = len(df)
    result = df.drop_duplicates(subset=columns if columns else None).reset_index(drop=True)
    removed = before - len(result)
    subset_desc = f"列 {columns}" if columns else "所有列"
    return SkillResult(result, f"按{subset_desc}去重，删除 {removed} 行，剩余 {len(result)} 行")


def skill_rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> SkillResult:
    if old_name not in df.columns:
        return SkillResult(df, "", error=f"列 {old_name!r} 不存在，可用：{list(df.columns)}")
    result = df.rename(columns={old_name: new_name})
    return SkillResult(result, f"已将列 {old_name!r} 重命名为 {new_name!r}")


def skill_sort(df: pd.DataFrame, column: str, ascending: bool = True) -> SkillResult:
    if column not in df.columns:
        return SkillResult(df, "", error=f"列 {column!r} 不存在")
    result = df.sort_values(column, ascending=ascending).reset_index(drop=True)
    direction = "升序" if ascending else "降序"
    return SkillResult(result, f"已按列 {column!r} {direction}排列")


def skill_aggregate(
    df: pd.DataFrame,
    group_by: list,
    agg: dict,
) -> SkillResult:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        return SkillResult(df, "", error=f"分组列不存在：{missing}")
    missing_val = [c for c in agg if c not in df.columns]
    if missing_val:
        return SkillResult(df, "", error=f"聚合列不存在：{missing_val}")
    try:
        result = df.groupby(group_by).agg(**{
            f"{col}_{func}": (col, func) for col, func in agg.items()
        }).reset_index()
        return SkillResult(result, f"已按 {group_by} 分组，聚合 {agg}")
    except Exception as e:
        return SkillResult(df, "", error=str(e))


def skill_add_column(df: pd.DataFrame, name: str, expression: str) -> SkillResult:
    try:
        result = df.copy()
        result[name] = result.eval(expression)
        return SkillResult(result, f"已新增列 {name!r}（表达式：{expression}）")
    except Exception as e:
        return SkillResult(df, "", error=f"表达式执行失败：{e}")


def skill_build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: list,
    title: str = "",
) -> SkillResult:
    from core.chart_builder import build_chart as _build
    try:
        option = _build(chart_type, df, x, y, title=title)
        return SkillResult(df, f"已生成 {chart_type} 图表（x={x}, y={y}）", chart_option=option)
    except ValueError as e:
        return SkillResult(df, "", error=str(e))
    except Exception as e:
        return SkillResult(df, "", error=f"图表生成失败：{e}")


def skill_describe(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> SkillResult:
    target = df[columns] if columns else df
    desc = target.describe(include="all").T.reset_index().rename(columns={"index": "列名"})
    return SkillResult(desc, f"已生成统计摘要（{len(desc)} 列）")


# ── Skill Registry ─────────────────────────────────────────────────────────────

SKILL_REGISTRY: dict = {
    "delete_columns": {
        "description": "删除指定列。适用于：'删除年龄列'、'去掉ID和备注列'",
        "params_schema": {"columns": "list[str] — 要删除的列名列表"},
        "fn": skill_delete_columns,
    },
    "delete_rows": {
        "description": "删除满足条件的行。适用于：'删除销售额小于500的行'",
        "params_schema": {"condition": "str — 筛选条件（被删除的行满足此条件）"},
        "fn": skill_delete_rows,
    },
    "filter_rows": {
        "description": "保留满足条件的行（查找/筛选）。适用于：'查找销售额最大的三行'、'筛选部门为销售的数据'",
        "params_schema": {"condition": "str — 筛选条件"},
        "fn": skill_filter_rows,
    },
    "fill_missing": {
        "description": "填充缺失值。method 可选：mean/median/zero/ffill/bfill 或固定值",
        "params_schema": {
            "method": "str — 填充方法",
            "columns": "list[str]（可选）— 仅填充指定列，默认全列",
        },
        "fn": skill_fill_missing,
    },
    "deduplicate": {
        "description": "去除重复行。适用于：'去重'、'删除重复记录'",
        "params_schema": {"columns": "list[str]（可选）— 基于哪些列判断重复，默认全列"},
        "fn": skill_deduplicate,
    },
    "rename_column": {
        "description": "重命名列。适用于：'把销售额改名为revenue'",
        "params_schema": {
            "old_name": "str — 原列名",
            "new_name": "str — 新列名",
        },
        "fn": skill_rename_column,
    },
    "sort": {
        "description": "按列排序。适用于：'按销售额降序排列'",
        "params_schema": {
            "column": "str — 排序列名",
            "ascending": "bool — true 升序，false 降序（默认 true）",
        },
        "fn": skill_sort,
    },
    "aggregate": {
        "description": "分组聚合。适用于：'按部门统计销售额总和'、'各月平均销售额'",
        "params_schema": {
            "group_by": "list[str] — 分组列",
            "agg": "dict — {列名: 聚合函数}，函数可选 sum/mean/count/max/min",
        },
        "fn": skill_aggregate,
    },
    "add_column": {
        "description": "新增计算列。适用于：'新增一列提成=销售额*0.1'",
        "params_schema": {
            "name": "str — 新列名",
            "expression": "str — Pandas eval 表达式，如 '销售额 * 0.1'",
        },
        "fn": skill_add_column,
    },
    "build_chart": {
        "description": "生成图表。适用于：'画柱状图'、'用饼图展示各部门占比'",
        "params_schema": {
            "chart_type": "str — bar/line/pie/area/scatter",
            "x": "str — X 轴列名",
            "y": "list[str] — Y 轴列名列表",
            "title": "str（可选）— 图表标题",
        },
        "fn": skill_build_chart,
    },
    "describe": {
        "description": "输出统计摘要。适用于：'数据概览'、'统计摘要'",
        "params_schema": {"columns": "list[str]（可选）— 仅统计指定列，默认全列"},
        "fn": skill_describe,
    },
}
