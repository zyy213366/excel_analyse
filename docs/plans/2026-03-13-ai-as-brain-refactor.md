# AI-as-Brain 架构重构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将系统重构为"AI 是大脑、Skills 是工具"的架构——LLM 理解自然语言并编排原子 skill 调用，彻底替换脆弱的关键词匹配。

**Architecture:** 用户指令 → AI Planner（Claude API）接收 df schema + 可用 skill 列表 → 返回结构化 JSON plan（技能调用序列）→ Executor 逐步执行 skill pipeline → 返回修改后的 df + 图表。每个 skill 是纯函数，独立可测，互相正交。

**Tech Stack:** Python 3.10, FastAPI, Pandas, Anthropic SDK (claude-haiku-4-5), Alpine.js, ECharts 5.x

---

## 问题诊断（不要跳过，理解这个再动手）

### 当前架构的根本缺陷

```
用户输入 → api_analyze（300行 if/elif）
         → _parse_process_instruction（关键词匹配）
         → ExcelProcessor（硬编码操作集合）
```

**根本问题：AI 只是个分类器，不是大脑。**

| 问题 | 表现 |
|------|------|
| 关键词匹配太脆弱 | "删除年龄列" 无法识别（没有列名关键词） |
| 操作不可组合 | 无法"清洗后删除age列再画柱状图" |
| 添加新操作成本极高 | 要改 routes.py + excel_processor.py + 测试 |
| LLM 被浪费 | 只用来做 mode 分类，实际能力完全没用上 |

### 目标架构

```
用户输入
  ↓
AI Planner（Claude haiku）
  接收：df schema（列名+类型+前3行样本）+ 可用 skill 列表 + 用户指令
  返回：JSON plan = [{skill, params, reason}, ...]
  ↓
Skill Executor
  顺序执行 plan，df 沿 pipeline 传递
  ↓
返回：最终 df + chart_option（如有）+ 每步 summary
```

---

## Task 1: 创建 Skills 层（`core/skills.py`）

**Files:**
- Create: `core/skills.py`
- Create: `tests/test_skills.py`

技能是**纯函数**：`(df: pd.DataFrame, **params) -> SkillResult`

每个技能返回统一的 `SkillResult` 数据类：

```python
@dataclass
class SkillResult:
    df: pd.DataFrame          # 执行后的 dataframe（图表技能时为原 df）
    summary: str              # 人类可读的执行摘要
    chart_option: dict | None = None   # ECharts option（仅图表技能）
    error: str | None = None  # 如有错误
```

**Step 1: 写失败测试**

创建 `tests/test_skills.py`，内容如下：

```python
"""Skills 单元测试"""
import pytest
import pandas as pd
from core.skills import (
    SkillResult,
    skill_delete_columns,
    skill_delete_rows,
    skill_filter_rows,
    skill_fill_missing,
    skill_deduplicate,
    skill_rename_column,
    skill_sort,
    skill_aggregate,
    skill_add_column,
    skill_build_chart,
    skill_describe,
    SKILL_REGISTRY,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "姓名": ["张三", "李四", "王五", "张三"],
        "年龄": [25, None, 30, 25],
        "部门": ["销售", "研发", "销售", "销售"],
        "销售额": [1200, 800, 1500, 1200],
    })


# ── delete_columns ──────────────────────────────────────────
class TestDeleteColumns:
    def test_delete_one(self, df):
        r = skill_delete_columns(df, columns=["年龄"])
        assert "年龄" not in r.df.columns
        assert "姓名" in r.df.columns
        assert r.error is None

    def test_delete_multiple(self, df):
        r = skill_delete_columns(df, columns=["年龄", "部门"])
        assert "年龄" not in r.df.columns
        assert "部门" not in r.df.columns

    def test_nonexistent_column_returns_error(self, df):
        r = skill_delete_columns(df, columns=["不存在列"])
        assert r.error is not None

    def test_empty_list_returns_error(self, df):
        r = skill_delete_columns(df, columns=[])
        assert r.error is not None


# ── delete_rows ─────────────────────────────────────────────
class TestDeleteRows:
    def test_delete_by_condition(self, df):
        r = skill_delete_rows(df, condition="销售额 > 1000")
        assert len(r.df) == 1   # 只剩 800 那行
        assert all(r.df["销售额"] <= 1000)

    def test_invalid_condition_returns_error(self, df):
        r = skill_delete_rows(df, condition="不存在列 > 0")
        assert r.error is not None


# ── filter_rows ─────────────────────────────────────────────
class TestFilterRows:
    def test_gt(self, df):
        r = skill_filter_rows(df, condition="销售额 > 1000")
        assert len(r.df) == 3
        assert all(r.df["销售额"] > 1000)

    def test_eq_string(self, df):
        r = skill_filter_rows(df, condition="部门 == 研发")
        assert len(r.df) == 1

    def test_contains(self, df):
        r = skill_filter_rows(df, condition="姓名 包含 张")
        assert len(r.df) == 2

    def test_chinese_op_gt(self, df):
        r = skill_filter_rows(df, condition="销售额大于1000")
        assert len(r.df) == 3

    def test_top_n_max(self, df):
        r = skill_filter_rows(df, condition="销售额最大的两行")
        assert len(r.df) == 2
        assert r.df["销售额"].max() == 1500

    def test_empty_condition_returns_all(self, df):
        r = skill_filter_rows(df, condition="")
        assert len(r.df) == len(df)


# ── fill_missing ────────────────────────────────────────────
class TestFillMissing:
    def test_fill_mean(self, df):
        r = skill_fill_missing(df, method="mean")
        assert r.df["年龄"].isnull().sum() == 0
        assert r.df["年龄"].iloc[1] == pytest.approx(26.67, rel=0.01)

    def test_fill_zero(self, df):
        r = skill_fill_missing(df, method="zero")
        assert r.df["年龄"].iloc[1] == 0

    def test_fill_specific_columns(self, df):
        r = skill_fill_missing(df, method="mean", columns=["年龄"])
        assert r.df["年龄"].isnull().sum() == 0


# ── deduplicate ─────────────────────────────────────────────
class TestDeduplicate:
    def test_dedup_all_cols(self, df):
        r = skill_deduplicate(df)
        assert len(r.df) == 3  # 张三那行重复

    def test_dedup_subset(self, df):
        r = skill_deduplicate(df, columns=["部门"])
        assert len(r.df) == 2  # 销售/研发各保留一行


# ── rename_column ───────────────────────────────────────────
class TestRenameColumn:
    def test_rename(self, df):
        r = skill_rename_column(df, old_name="销售额", new_name="revenue")
        assert "revenue" in r.df.columns
        assert "销售额" not in r.df.columns

    def test_rename_nonexistent_returns_error(self, df):
        r = skill_rename_column(df, old_name="不存在", new_name="x")
        assert r.error is not None


# ── sort ────────────────────────────────────────────────────
class TestSort:
    def test_sort_desc(self, df):
        r = skill_sort(df, column="销售额", ascending=False)
        vals = r.df["销售额"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_sort_asc(self, df):
        r = skill_sort(df, column="年龄", ascending=True)
        non_null = r.df["年龄"].dropna().tolist()
        assert non_null == sorted(non_null)


# ── aggregate ───────────────────────────────────────────────
class TestAggregate:
    def test_group_sum(self, df):
        r = skill_aggregate(df, group_by=["部门"], agg={"销售额": "sum"})
        assert "销售额_sum" in r.df.columns
        sales_sum = r.df[r.df["部门"] == "销售"]["销售额_sum"].values[0]
        assert sales_sum == 3900  # 1200+1500+1200

    def test_group_mean(self, df):
        r = skill_aggregate(df, group_by=["部门"], agg={"销售额": "mean"})
        assert "销售额_mean" in r.df.columns

    def test_invalid_col_returns_error(self, df):
        r = skill_aggregate(df, group_by=["不存在列"], agg={"销售额": "sum"})
        assert r.error is not None


# ── add_column ──────────────────────────────────────────────
class TestAddColumn:
    def test_arithmetic_expression(self, df):
        # 新列 = 销售额 * 0.1
        r = skill_add_column(df, name="提成", expression="销售额 * 0.1")
        assert "提成" in r.df.columns
        assert r.df["提成"].iloc[0] == pytest.approx(120.0)

    def test_invalid_expression_returns_error(self, df):
        r = skill_add_column(df, name="x", expression="不存在列 + 1")
        assert r.error is not None


# ── build_chart ─────────────────────────────────────────────
class TestBuildChart:
    @pytest.fixture
    def agg_df(self):
        return pd.DataFrame({"部门": ["销售", "研发"], "销售额_sum": [3900, 800]})

    def test_bar(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="bar", x="部门", y=["销售额_sum"])
        assert r.chart_option is not None
        assert r.chart_option["series"][0]["type"] == "bar"

    def test_pie(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="pie", x="部门", y=["销售额_sum"])
        assert r.chart_option["series"][0]["type"] == "pie"

    def test_line(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="line", x="部门", y=["销售额_sum"])
        assert r.chart_option["series"][0]["type"] == "line"

    def test_unknown_type_returns_error(self, agg_df):
        r = skill_build_chart(agg_df, chart_type="unknown", x="部门", y=["销售额_sum"])
        assert r.error is not None


# ── describe ────────────────────────────────────────────────
class TestDescribe:
    def test_describe_all(self, df):
        r = skill_describe(df)
        assert r.df is not None
        assert len(r.df) > 0

    def test_describe_subset(self, df):
        r = skill_describe(df, columns=["销售额"])
        assert "销售额" in r.df.columns or "销售额" in r.df.index.tolist()


# ── SKILL_REGISTRY ──────────────────────────────────────────
class TestSkillRegistry:
    def test_all_skills_registered(self):
        expected = {
            "delete_columns", "delete_rows", "filter_rows",
            "fill_missing", "deduplicate", "rename_column",
            "sort", "aggregate", "add_column", "build_chart", "describe",
        }
        assert expected.issubset(set(SKILL_REGISTRY.keys()))

    def test_each_entry_has_description(self):
        for name, meta in SKILL_REGISTRY.items():
            assert "description" in meta, f"{name} 缺少 description"
            assert "params_schema" in meta, f"{name} 缺少 params_schema"
            assert "fn" in meta, f"{name} 缺少 fn"
```

**Step 2: 运行测试，确认全部失败**

```bash
python -m pytest tests/test_skills.py -v --tb=short
```
期望：`ImportError: cannot import name 'skill_delete_columns' from 'core.skills'`（文件不存在）

**Step 3: 实现 `core/skills.py`**

创建 `core/skills.py`，完整内容：

```python
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


# ── 条件解析（从 excel_processor.py 迁移，合并到此处）─────────────────────────

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
    """大小写不敏感的列名匹配，优先长列名"""
    cols = sorted(df.columns, key=len, reverse=True)
    hint_l = hint.lower()
    for col in cols:
        if col.lower() == hint_l:
            return col
    return None


def _apply_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    将自然语言/符号条件字符串应用到 df，返回筛选后的子集。
    支持：
      1. 包含：  "列名 包含 值"
      2. Top-N：  "列名最大的3行" / "列名最小的两"
      3. 单极值： "列名最大" / "列名最小"
      4. 中文运算符：大于/小于/等于/为/大于等于/小于等于
      5. 符号运算符：>=/<=/!=/==/>/< + 字符串等值
    """
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

    return df  # 无法解析，返回原始（上层处理错误）


# ── Skills ────────────────────────────────────────────────────────────────────

def skill_delete_columns(df: pd.DataFrame, columns: list[str]) -> SkillResult:
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
    columns: Optional[list[str]] = None,
) -> SkillResult:
    result = df.copy()
    target_cols = columns if columns else list(df.columns)
    filled = 0
    for col in target_cols:
        if col not in result.columns:
            continue
        null_cnt = result[col].isnull().sum()
        if null_cnt == 0:
            continue
        if method == "mean" and pd.api.types.is_numeric_dtype(result[col]):
            result[col].fillna(result[col].mean(), inplace=True)
        elif method == "median" and pd.api.types.is_numeric_dtype(result[col]):
            result[col].fillna(result[col].median(), inplace=True)
        elif method == "zero":
            result[col].fillna(0, inplace=True)
        elif method == "ffill":
            result[col].fillna(method="ffill", inplace=True)
        elif method == "bfill":
            result[col].fillna(method="bfill", inplace=True)
        else:
            result[col].fillna(method, inplace=True)  # 固定值
        filled += null_cnt
    return SkillResult(result, f"已填充缺失值 {filled} 处（方法：{method}）")


def skill_deduplicate(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> SkillResult:
    before = len(df)
    result = df.drop_duplicates(subset=columns).reset_index(drop=True)
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
    group_by: list[str],
    agg: dict[str, str],
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
    y: list[str],
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
    columns: Optional[list[str]] = None,
) -> SkillResult:
    target = df[columns] if columns else df
    desc = target.describe(include="all").T.reset_index().rename(columns={"index": "列名"})
    return SkillResult(desc, f"已生成统计摘要（{len(desc)} 列）")


# ── Skill Registry（AI Planner 用这个列表生成 prompt）──────────────────────────

SKILL_REGISTRY: dict[str, dict[str, Any]] = {
    "delete_columns": {
        "description": "删除指定列。适用于：'删除年龄列'、'去掉ID和备注列'",
        "params_schema": {"columns": "list[str] — 要删除的列名列表"},
        "fn": skill_delete_columns,
    },
    "delete_rows": {
        "description": "删除满足条件的行。适用于：'删除销售额小于500的行'、'去掉部门为市场的记录'",
        "params_schema": {"condition": "str — 筛选条件（被删除的行满足此条件）"},
        "fn": skill_delete_rows,
    },
    "filter_rows": {
        "description": "保留满足条件的行（查找/筛选）。适用于：'查找销售额最大的三行'、'筛选部门为销售的数据'",
        "params_schema": {"condition": "str — 筛选条件"},
        "fn": skill_filter_rows,
    },
    "fill_missing": {
        "description": "填充缺失值。method 可选：mean（均值）、median（中位数）、zero（填0）、ffill（向前填充）、bfill（向后填充）或固定值字符串",
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
            "x": "str — X 轴列名（饼图为分类列）",
            "y": "list[str] — Y 轴列名（饼图为数值列，取第一个）",
            "title": "str（可选）— 图表标题",
        },
        "fn": skill_build_chart,
    },
    "describe": {
        "description": "输出统计摘要（均值/最大/最小/标准差等）。适用于：'数据概览'、'统计摘要'",
        "params_schema": {"columns": "list[str]（可选）— 仅统计指定列，默认全列"},
        "fn": skill_describe,
    },
}
```

**Step 4: 运行测试**

```bash
python -m pytest tests/test_skills.py -v --tb=short
```
期望：全部 PASS

**Step 5: Commit**

```bash
git add core/skills.py tests/test_skills.py
git commit -m "feat: add skills layer with 11 atomic operations and full test coverage"
```

---

## Task 2: 创建 AI Planner（`core/ai_planner.py`）

**Files:**
- Create: `core/ai_planner.py`
- Create: `tests/test_ai_planner.py`

AI Planner 调用 Claude API，接收 df schema + 指令，返回 JSON plan。

**Step 1: 写失败测试**

创建 `tests/test_ai_planner.py`：

```python
"""AI Planner 测试（用 mock 替换真实 LLM 调用）"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.ai_planner import AIPLanner, PlanStep, plan_to_steps


@pytest.fixture
def schema_df():
    return pd.DataFrame({
        "姓名": ["张三", "李四"],
        "年龄": [25, 30],
        "部门": ["销售", "研发"],
        "销售额": [1200, 800],
    })


class TestPlanStep:
    def test_valid_step(self):
        step = PlanStep(skill="filter_rows", params={"condition": "销售额 > 1000"}, reason="筛选高销售")
        assert step.skill == "filter_rows"
        assert step.params["condition"] == "销售额 > 1000"

    def test_from_dict(self):
        d = {"skill": "delete_columns", "params": {"columns": ["年龄"]}, "reason": "删列"}
        step = PlanStep(**d)
        assert step.skill == "delete_columns"


class TestPlanToSteps:
    def test_parses_valid_json(self):
        raw = '{"plan": [{"skill": "filter_rows", "params": {"condition": "销售额 > 1000"}, "reason": "筛选"}]}'
        steps = plan_to_steps(raw)
        assert len(steps) == 1
        assert steps[0].skill == "filter_rows"

    def test_handles_markdown_code_block(self):
        raw = '```json\n{"plan": [{"skill": "sort", "params": {"column": "销售额", "ascending": false}, "reason": "排序"}]}\n```'
        steps = plan_to_steps(raw)
        assert len(steps) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="无法解析"):
            plan_to_steps("这不是JSON")

    def test_unknown_skill_raises(self):
        raw = '{"plan": [{"skill": "fly_to_moon", "params": {}, "reason": ""}]}'
        with pytest.raises(ValueError, match="未知 skill"):
            plan_to_steps(raw)


class TestAIPlanner:
    def test_build_prompt_contains_schema(self, schema_df):
        planner = AIPLanner()
        prompt = planner._build_prompt(schema_df, "删除年龄列")
        assert "年龄" in prompt
        assert "delete_columns" in prompt
        assert "删除年龄列" in prompt

    def test_build_prompt_contains_skill_list(self, schema_df):
        planner = AIPLanner()
        prompt = planner._build_prompt(schema_df, "随便")
        assert "filter_rows" in prompt
        assert "aggregate" in prompt

    @patch("core.ai_planner.anthropic.Anthropic")
    def test_plan_calls_claude_api(self, mock_anthropic_cls, schema_df):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"plan": [{"skill": "filter_rows", "params": {"condition": "销售额 > 1000"}, "reason": "筛选"}]}')]
        mock_client.messages.create.return_value = mock_msg

        planner = AIPLanner()
        steps = planner.plan(schema_df, "筛选销售额大于1000的行")
        assert len(steps) == 1
        assert steps[0].skill == "filter_rows"
        mock_client.messages.create.assert_called_once()

    @patch("core.ai_planner.anthropic.Anthropic")
    def test_plan_multi_step(self, mock_anthropic_cls, schema_df):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='''{"plan": [
            {"skill": "delete_columns", "params": {"columns": ["年龄"]}, "reason": "删除年龄"},
            {"skill": "aggregate", "params": {"group_by": ["部门"], "agg": {"销售额": "sum"}}, "reason": "分组求和"},
            {"skill": "build_chart", "params": {"chart_type": "bar", "x": "部门", "y": ["销售额_sum"]}, "reason": "画图"}
        ]}''')]
        mock_client.messages.create.return_value = mock_msg

        planner = AIPLanner()
        steps = planner.plan(schema_df, "删除年龄列，按部门统计销售额，画柱状图")
        assert len(steps) == 3
        assert steps[2].skill == "build_chart"
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_ai_planner.py -v --tb=short
```
期望：`ImportError: cannot import name 'AIPLanner'`

**Step 3: 实现 `core/ai_planner.py`**

```python
"""
AI Planner — 调用 Claude 将自然语言指令转为 skill 调用序列。
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Any
import anthropic
import pandas as pd
from core.skills import SKILL_REGISTRY


@dataclass
class PlanStep:
    skill: str
    params: dict[str, Any]
    reason: str = ""


def plan_to_steps(raw_text: str) -> list[PlanStep]:
    """
    将 LLM 返回的原始文本解析为 PlanStep 列表。
    兼容带 ```json ... ``` 的 markdown 代码块格式。
    """
    text = raw_text.strip()
    # 剥去 markdown 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法解析 LLM 响应为 JSON：{e}\n原始内容：{raw_text[:200]}")

    steps_raw = data.get("plan", [])
    steps = []
    for item in steps_raw:
        skill_name = item.get("skill", "")
        if skill_name not in SKILL_REGISTRY:
            raise ValueError(f"未知 skill：{skill_name!r}，可用：{list(SKILL_REGISTRY)}")
        steps.append(PlanStep(
            skill=skill_name,
            params=item.get("params", {}),
            reason=item.get("reason", ""),
        ))
    return steps


class AIPLanner:
    """
    将自然语言指令翻译为 skill 调用序列。
    使用 claude-haiku-4-5 以减少延迟和成本。
    """

    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not key:
            raise RuntimeError("未设置 ANTHROPIC_API_KEY 环境变量")
        self.client = anthropic.Anthropic(api_key=key)

    def _build_schema_desc(self, df: pd.DataFrame) -> str:
        lines = [f"- {col}（{dtype}）" for col, dtype in df.dtypes.items()]
        sample = df.head(3).to_dict("records")
        return "\n".join(lines) + f"\n\n前3行样本：\n{json.dumps(sample, ensure_ascii=False, default=str)}"

    def _build_skill_list(self) -> str:
        lines = []
        for name, meta in SKILL_REGISTRY.items():
            params_str = "，".join(f"{k}: {v}" for k, v in meta["params_schema"].items())
            lines.append(f'- **{name}**: {meta["description"]}\n  参数：{params_str}')
        return "\n".join(lines)

    def _build_prompt(self, df: pd.DataFrame, instruction: str) -> str:
        return f"""你是一个数据处理助手。用户有一份 Excel 数据，结构如下：

## 数据结构
{self._build_schema_desc(df)}

## 可用操作（Skills）
{self._build_skill_list()}

## 用户指令
{instruction}

## 要求
请将用户指令分解为一个或多个 skill 调用序列，返回严格的 JSON 格式：

```json
{{
  "plan": [
    {{"skill": "技能名", "params": {{...}}, "reason": "这一步的用途"}},
    ...
  ]
}}
```

注意：
1. 只能使用上面列出的 skill，不能发明新 skill
2. params 中的列名必须严格使用数据结构中存在的列名
3. 如果需要先聚合再画图，聚合后的列名格式为 "原列名_聚合函数"，如 "销售额_sum"
4. 只返回 JSON，不要额外说明"""

    def plan(self, df: pd.DataFrame, instruction: str) -> list[PlanStep]:
        """
        主入口：接收 df 和自然语言指令，返回 PlanStep 列表。
        """
        prompt = self._build_prompt(df, instruction)
        msg = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        return plan_to_steps(raw)
```

**Step 4: 运行测试**

```bash
python -m pytest tests/test_ai_planner.py -v --tb=short
```
期望：全部 PASS（mock 测试不需要真实 API key）

**Step 5: Commit**

```bash
git add core/ai_planner.py tests/test_ai_planner.py
git commit -m "feat: add AI planner — LLM translates instructions to skill call sequences"
```

---

## Task 3: 创建 Skill Executor（`core/skill_executor.py`）

**Files:**
- Create: `core/skill_executor.py`
- Create: `tests/test_skill_executor.py`

Executor 顺序执行 plan，df 沿 pipeline 传递，聚合所有 summary。

**Step 1: 写失败测试**

创建 `tests/test_skill_executor.py`：

```python
"""Skill Executor 测试"""
import pytest
import pandas as pd
from core.skills import PlanStep
from core.skill_executor import execute_plan, ExecutionResult


@pytest.fixture
def df():
    return pd.DataFrame({
        "姓名": ["张三", "李四", "王五"],
        "年龄": [25, None, 30],
        "部门": ["销售", "研发", "销售"],
        "销售额": [1200, 800, 1500],
    })


class TestExecutePlan:
    def test_single_step(self, df):
        plan = [PlanStep("filter_rows", {"condition": "销售额 > 1000"}, "筛选")]
        result = execute_plan(df, plan)
        assert result.success
        assert len(result.df) == 2
        assert len(result.steps_summary) == 1

    def test_multi_step_pipeline(self, df):
        plan = [
            PlanStep("fill_missing", {"method": "mean"}, "填充"),
            PlanStep("filter_rows", {"condition": "销售额 > 1000"}, "筛选"),
            PlanStep("sort", {"column": "销售额", "ascending": False}, "排序"),
        ]
        result = execute_plan(df, plan)
        assert result.success
        assert len(result.df) == 2
        assert result.df["销售额"].iloc[0] == 1500  # 降序

    def test_skill_error_stops_execution(self, df):
        plan = [
            PlanStep("delete_columns", {"columns": ["不存在列"]}, "删除"),
            PlanStep("sort", {"column": "销售额"}, "排序"),
        ]
        result = execute_plan(df, plan)
        assert not result.success
        assert result.error is not None
        assert "不存在列" in result.error

    def test_chart_step_returns_option(self, df):
        plan = [PlanStep("build_chart", {"chart_type": "bar", "x": "部门", "y": ["销售额"]}, "画图")]
        result = execute_plan(df, plan)
        assert result.success
        assert result.chart_option is not None

    def test_empty_plan_returns_original(self, df):
        result = execute_plan(df, [])
        assert result.success
        assert len(result.df) == len(df)

    def test_steps_summary_collected(self, df):
        plan = [
            PlanStep("fill_missing", {"method": "zero"}, "填0"),
            PlanStep("deduplicate", {}, "去重"),
        ]
        result = execute_plan(df, plan)
        assert len(result.steps_summary) == 2


class TestExecutionResult:
    def test_full_summary_joins_steps(self, df):
        plan = [
            PlanStep("filter_rows", {"condition": "销售额 > 1000"}, "筛选"),
        ]
        result = execute_plan(df, plan)
        assert "筛选" in result.full_summary or "1000" in result.full_summary
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_skill_executor.py -v --tb=short
```

**Step 3: 实现 `core/skill_executor.py`**

```python
"""
Skill Executor — 顺序执行 plan，df 沿 pipeline 传递。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
from core.skills import SKILL_REGISTRY, SkillResult
from core.ai_planner import PlanStep


@dataclass
class ExecutionResult:
    df: pd.DataFrame
    success: bool
    steps_summary: list[str] = field(default_factory=list)
    chart_option: Optional[dict] = None
    error: Optional[str] = None

    @property
    def full_summary(self) -> str:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.steps_summary))


def execute_plan(df: pd.DataFrame, plan: list[PlanStep]) -> ExecutionResult:
    """
    顺序执行 plan 中的每个 step。
    - df 沿 pipeline 传递（每步输出是下一步输入）
    - 遇到 error 立即停止，返回失败结果
    - chart_option 由最后一个 build_chart step 决定
    """
    current_df = df.copy()
    summaries = []
    chart_option = None

    if not plan:
        return ExecutionResult(df=current_df, success=True)

    for step in plan:
        skill_meta = SKILL_REGISTRY.get(step.skill)
        if not skill_meta:
            return ExecutionResult(
                df=current_df, success=False,
                steps_summary=summaries,
                error=f"未知 skill：{step.skill!r}",
            )
        fn = skill_meta["fn"]
        try:
            skill_result: SkillResult = fn(current_df, **step.params)
        except TypeError as e:
            return ExecutionResult(
                df=current_df, success=False,
                steps_summary=summaries,
                error=f"Skill {step.skill!r} 参数错误：{e}",
            )

        if skill_result.error:
            return ExecutionResult(
                df=current_df, success=False,
                steps_summary=summaries,
                error=skill_result.error,
            )

        current_df = skill_result.df
        summaries.append(skill_result.summary)
        if skill_result.chart_option:
            chart_option = skill_result.chart_option

    return ExecutionResult(
        df=current_df,
        success=True,
        steps_summary=summaries,
        chart_option=chart_option,
    )
```

**Step 4: 运行测试**

```bash
python -m pytest tests/test_skill_executor.py -v --tb=short
```
期望：全部 PASS

**Step 5: Commit**

```bash
git add core/skill_executor.py tests/test_skill_executor.py
git commit -m "feat: add skill executor — pipeline execution with error handling"
```

---

## Task 4: 重构 API 路由（`api/routes.py`）

**Files:**
- Modify: `api/routes.py`
- Create: `tests/test_routes_new.py`

用一个干净的 `/api/chat` 端点替换 `api_analyze` 中 300 行的 if/elif 链。

> ⚠️ **不删除旧端点** `/api/analyze`，保持兼容，只是新增 `/api/chat`。删除旧代码在所有测试通过后进行。

**Step 1: 写失败测试**

创建 `tests/test_routes_new.py`：

```python
"""新 /api/chat 端点集成测试（用 mock 替换 LLM）"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import io

from app import app  # 主 FastAPI 应用

client = TestClient(app)


@pytest.fixture
def uploaded_file_id(tmp_path):
    """上传一个测试 Excel 文件，返回 file_id"""
    df = pd.DataFrame({
        "部门": ["销售", "研发", "销售"],
        "销售额": [1200, 800, 1500],
        "年龄": [25, 30, None],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    resp = client.post("/api/upload", files={"file": ("test.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 200
    return resp.json()["file_id"]


@patch("api.routes.AIPLanner")
def test_chat_filter_rows(mock_planner_cls, uploaded_file_id):
    mock_planner = MagicMock()
    mock_planner_cls.return_value = mock_planner
    from core.ai_planner import PlanStep
    mock_planner.plan.return_value = [
        PlanStep("filter_rows", {"condition": "销售额 > 1000"}, "筛选")
    ]

    resp = client.post("/api/chat", json={
        "file_id": uploaded_file_id,
        "instruction": "筛选销售额大于1000的行",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["table_data"]) == 2


@patch("api.routes.AIPLanner")
def test_chat_delete_column(mock_planner_cls, uploaded_file_id):
    mock_planner = MagicMock()
    mock_planner_cls.return_value = mock_planner
    from core.ai_planner import PlanStep
    mock_planner.plan.return_value = [
        PlanStep("delete_columns", {"columns": ["年龄"]}, "删除年龄")
    ]

    resp = client.post("/api/chat", json={
        "file_id": uploaded_file_id,
        "instruction": "删除年龄列",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    cols = [row.keys() for row in data["table_data"]]
    if cols:
        assert "年龄" not in list(cols)[0]


@patch("api.routes.AIPLanner")
def test_chat_returns_chart_option(mock_planner_cls, uploaded_file_id):
    mock_planner = MagicMock()
    mock_planner_cls.return_value = mock_planner
    from core.ai_planner import PlanStep
    mock_planner.plan.return_value = [
        PlanStep("build_chart", {"chart_type": "bar", "x": "部门", "y": ["销售额"]}, "画图")
    ]

    resp = client.post("/api/chat", json={
        "file_id": uploaded_file_id,
        "instruction": "画柱状图",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["chart_option"] is not None


@patch("api.routes.AIPLanner")
def test_chat_skill_error_returns_friendly_message(mock_planner_cls, uploaded_file_id):
    mock_planner = MagicMock()
    mock_planner_cls.return_value = mock_planner
    from core.ai_planner import PlanStep
    mock_planner.plan.return_value = [
        PlanStep("delete_columns", {"columns": ["不存在列"]}, "删列")
    ]

    resp = client.post("/api/chat", json={
        "file_id": uploaded_file_id,
        "instruction": "删除不存在列",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "不存在列" in data["error"]


def test_chat_invalid_file_id():
    resp = client.post("/api/chat", json={
        "file_id": "nonexistent_id",
        "instruction": "删除年龄列",
    })
    assert resp.status_code == 400
```

**Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_routes_new.py -v --tb=short
```

**Step 3: 在 `api/routes.py` 添加新端点**

在 `routes.py` 底部（`api_chart` 函数之后）添加：

```python
# ── 新架构：/api/chat ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    file_id: str
    instruction: str


@router.post("/api/chat")
async def api_chat(req: ChatRequest):
    """
    新架构入口：AI 大脑 + Skills 工具。
    接收自然语言指令 → AI 规划 skill 序列 → 执行 → 返回结果。
    """
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")

    file_path = _uploaded_files[req.file_id]
    try:
        df_raw, _ = load_excel(str(file_path))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    # 1. AI 规划
    try:
        from core.ai_planner import AIPLanner
        from core.skill_executor import execute_plan
        planner = AIPLanner()
        plan = planner.plan(df_raw, req.instruction)
    except Exception as e:
        return JSONResponse({"success": False, "error": f"AI 规划失败：{str(e)}"})

    # 2. 执行 plan
    from core.skill_executor import execute_plan
    exec_result = execute_plan(df_raw, plan)

    if not exec_result.success:
        return JSONResponse({"success": False, "error": exec_result.error})

    # 3. 保存结果文件（复用现有逻辑）
    report_filename = None
    try:
        from core.process_result import ProcessResult
        proc_result = ProcessResult(
            operation="chat",
            summary_text=exec_result.full_summary,
            result_df=exec_result.df,
        )
        proc_result.raw_row_count = len(df_raw)
        proc_result.valid_row_count = len(exec_result.df)
        out_path = get_output_path(file_path.name, "chat")
        _write_process_excel(proc_result, out_path)
        cleanup_old_reports()
        report_filename = out_path.name
    except Exception:
        pass

    table_rows = []
    if exec_result.df is not None and not exec_result.df.empty:
        table_rows = exec_result.df.head(200).to_dict("records")

    return JSONResponse({
        "success": True,
        "summary_text": exec_result.full_summary,
        "steps": exec_result.steps_summary,
        "table_data": table_rows,
        "chart_option": exec_result.chart_option,
        "report_filename": report_filename,
        "data_info": {
            "raw": len(df_raw),
            "valid": len(exec_result.df),
        },
    })
```

**Step 4: 运行测试**

```bash
python -m pytest tests/test_routes_new.py -v --tb=short
```
期望：全部 PASS

**Step 5: Commit**

```bash
git add api/routes.py tests/test_routes_new.py
git commit -m "feat: add /api/chat endpoint — AI-as-brain entry point"
```

---

## Task 5: 更新前端（`templates/analyze.html`）

**Files:**
- Modify: `templates/analyze.html`

前端改造：保留现有 UI 结构，将聊天发送接口从 `/api/analyze` 切换到 `/api/chat`，并渲染 `steps`（每步执行摘要）。

**Step 1: 定位当前发送逻辑**

在 `analyze.html` 中找到 `sendMessage` 或发送聊天的函数（搜索 `/api/analyze`），通常在 Alpine.js 组件的 `methods` 或直接在 `x-data` 中。

**Step 2: 修改 API 调用**

将调用端点从 `/api/analyze` 改为 `/api/chat`，精简请求体：

```javascript
// 旧：
const resp = await fetch('/api/analyze', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    file_id: this.fileId,
    instruction: this.input,
    use_ai: true,
    manual_mode: '',
    manual_y: '',
    manual_x_cols: [],
  })
})

// 新：
const resp = await fetch('/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    file_id: this.fileId,
    instruction: this.input,
  })
})
```

**Step 3: 渲染 steps（执行步骤追踪）**

在消息气泡模板中，当 `msg.steps` 存在时，在 `summary_text` 下方显示步骤列表：

```html
<!-- 在消息气泡内，summary_text 之后 -->
<template x-if="msg.steps && msg.steps.length > 1">
  <div class="mt-2 text-xs text-gray-400 border-t border-gray-600 pt-2">
    <div class="font-medium mb-1">执行步骤：</div>
    <template x-for="(step, i) in msg.steps" :key="i">
      <div class="flex items-start gap-1">
        <span class="text-blue-400" x-text="`${i+1}.`"></span>
        <span x-text="step"></span>
      </div>
    </template>
  </div>
</template>
```

**Step 4: 保存后手动测试**

1. 启动服务：`python app.py`
2. 上传一个 Excel 文件
3. 输入"删除年龄列"——期望：返回不含年龄列的表格
4. 输入"筛选销售额大于1000的行"——期望：返回筛选后行
5. 输入"按部门统计销售额总和，画柱状图"——期望：聚合表 + 柱状图

**Step 5: Commit**

```bash
git add templates/analyze.html
git commit -m "feat: switch frontend to /api/chat, show execution steps"
```

---

## Task 6: 全量回归测试 + 清理

**Step 1: 运行所有测试**

```bash
python -m pytest tests/ -v --tb=short
```
期望：所有测试（原有 69 个 + 新增 ~40 个）全部 PASS

**Step 2: 如有失败，逐一修复**

常见失败原因：
- `ANTHROPIC_API_KEY` 未设置（mock 测试不需要真实 key，检查 mock 是否正确）
- `ProcessResult` 字段不兼容（检查 `result_df` 是否可以为 None）
- ECharts option 格式不匹配（检查 `chart_builder.py` 返回结构）

**Step 3: 可选清理（仅在确认新端点工作正常后）**

`api/routes.py` 中的 `_parse_process_instruction`、`_extract_col`、`_clean_lookup_condition` 函数以及 `api_analyze` 中的关键词匹配分支（前150行）可以删除。
**谨慎**：`/api/analyze` 端点本身暂时保留（前端 fallback）。

**Step 4: 最终推送**

```bash
git add -A
git commit -m "refactor: complete AI-as-brain architecture — skills, planner, executor"
git push
```

---

## 架构对比总结

| 维度 | 重构前 | 重构后 |
|------|--------|--------|
| 添加新操作 | 改 3 个文件（routes + processor + 关键词列表） | 只需在 `skills.py` 加一个函数 + 注册到 `SKILL_REGISTRY` |
| "删除年龄列" | ❌ 关键词匹配失败 | ✅ LLM 识别 → `delete_columns(columns=["年龄"])` |
| 组合操作 | ❌ 不支持 | ✅ AI 返回多步 plan，pipeline 执行 |
| 可测试性 | 集成测试困难（端点依赖整个 NLP 链） | ✅ 每个 skill 是纯函数，独立单测 |
| LLM 使用方式 | 仅做 mode 分类 | 完整理解意图 + 编排操作序列 |
