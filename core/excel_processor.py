"""
Excel处理 + 数据运算引擎
支持 10 种操作：数据清洗、查找、计数、合并、拆分、求和、均值、极值、逻辑、合并计算
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import re

from core.process_result import ProcessResult


# ──────────────────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────────────────

# 中文数字映射
_CN_NUM = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "二十": 20, "三十": 30, "五十": 50, "百": 100,
}


def _cn_num(s: str) -> int:
    """中文或阿拉伯数字字符串 → int，解析失败返回 1"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    return _CN_NUM.get(s, 1)


def _match_col(col_hint: str, df: pd.DataFrame) -> Optional[str]:
    """大小写不敏感地在 df.columns 中查找列名"""
    if col_hint in df.columns:
        return col_hint
    col_hint_l = col_hint.lower()
    return next((c for c in df.columns if c.lower() == col_hint_l), None)


# ──────────────────────────────────────────────────────────
# 条件解析辅助函数
# ──────────────────────────────────────────────────────────

# 中文比较运算符（降序：先匹配长词，避免"大于等于"被"大于"提前截断）
_ZH_OPS = [
    ("大于等于", ">="), ("小于等于", "<="), ("不等于", "!="), ("不是", "!="),
    ("大于", ">"), ("小于", "<"), ("等于", "=="), ("为", "=="),
]


def _apply_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    将自然语言 / 简单表达式条件应用到 DataFrame，返回过滤后子集。
    支持格式：
      - "列名 > 100" / "列名大于100"
      - "列名 == 值" / "列名等于值" / "列名为值"
      - "列名 包含 关键词"
      - "列名最大" / "列名最小"          → 返回极值所在行（可多行）
      - "列名最大的3" / "列名最大的三"   → Top-N 行
    若解析失败，返回原始 DataFrame（不过滤），同时在 name 属性中标记"未匹配"。
    """
    if not condition or not condition.strip():
        return df

    cond = condition.strip()

    # ── 1. 中文"包含" → str.contains ──────────────────────
    m = re.match(r"^(.+?)\s*包含\s*(.+)$", cond)
    if m:
        col = _match_col(m.group(1).strip(), df)
        if col:
            return df[df[col].astype(str).str.contains(m.group(2).strip(), na=False)]
        return df

    # ── 2. Top-N 极值：列名最大/最小的N（必须在单极值之前检测）──
    m = re.match(
        r"^(.+?)(?:中)?(?:(最大)|(最小))的?(\d+|[一两二三四五六七八九十百]+)(?:行|条|个)?$",
        cond,
    )
    if m:
        col = _match_col(m.group(1).strip(), df)
        if col is not None:
            is_max = m.group(2) is not None
            n = _cn_num(m.group(4))
            return df.nlargest(n, col) if is_max else df.nsmallest(n, col)

    # ── 3. 单极值：列名最大 / 列名最小 ─────────────────────
    m = re.match(r"^(.+?)(?:中)?(?:最大|最小)(?:的(?:行|记录|数据|值)?)?$", cond)
    if m:
        col = _match_col(m.group(1).strip(), df)
        if col is not None:
            extreme_val = df[col].max() if "最大" in cond else df[col].min()
            return df[df[col] == extreme_val]

    # ── 4. 中文比较运算符 ───────────────────────────────────
    for zh_op, sym in _ZH_OPS:
        m = re.match(rf"^(.+?)\s*{zh_op}\s*(.+)$", cond)
        if m:
            col = _match_col(m.group(1).strip(), df)
            val_str = m.group(2).strip()
            if col is not None:
                try:
                    val = float(val_str)
                    ops = {">=": "__ge__", "<=": "__le__", "!=": "__ne__",
                           "==": "__eq__", ">": "__gt__", "<": "__lt__"}
                    return df[getattr(df[col], ops[sym])(val)]
                except ValueError:
                    if sym in ("==", "!="):
                        fn = (lambda s, v: s == v) if sym == "==" else (lambda s, v: s != v)
                        return df[fn(df[col].astype(str), val_str)]
            break

    # ── 5. 标准数值/字符串比较运算符 ────────────────────────
    for op in [">=", "<=", "!=", "==", ">", "<"]:
        m = re.match(rf"^(.+?)\s*{re.escape(op)}\s*(.+)$", cond)
        if m:
            col = _match_col(m.group(1).strip(), df)
            val_str = m.group(2).strip()
            if col is None:
                break
            try:
                val = float(val_str)
                ops = {">=": "__ge__", "<=": "__le__", "!=": "__ne__",
                       "==": "__eq__", ">": "__gt__", "<": "__lt__"}
                return df[getattr(df[col], ops[op])(val)]
            except ValueError:
                if op in ("==", "!="):
                    fn = (lambda s, v: s == v) if op == "==" else (lambda s, v: s != v)
                    return df[fn(df[col].astype(str), val_str)]
            break

    # 解析失败，返回原 df（调用方可据 len 判断是否有效）
    return df


# ──────────────────────────────────────────────────────────
# Excel 处理类
# ──────────────────────────────────────────────────────────

class ExcelProcessor:
    """Excel处理与数据运算操作集合"""

    # ── 1. 数据清洗 ──────────────────────────────────────

    @staticmethod
    def process_clean(df: pd.DataFrame,
                      drop_duplicates: bool = True,
                      fill_missing: Optional[str] = "mean",
                      normalize: bool = False,
                      drop_empty_cols: bool = False) -> ProcessResult:
        """
        数据清洗：
        - drop_empty_cols : 删除全为空的列
        - drop_duplicates : 去除重复行
        - fill_missing    : "mean"/"median"/"mode"/"zero"/"drop"/None
        - normalize       : Z-score 标准化数值列
        """
        raw = len(df)
        result = df.copy()
        stats: dict = {}

        # 删除全空列
        if drop_empty_cols:
            before_cols = set(result.columns)
            result = result.dropna(axis=1, how="all")
            dropped = [c for c in before_cols if c not in result.columns]
            stats["删除全空列"] = dropped if dropped else []
            if dropped:
                stats["删除列数"] = len(dropped)

        # 去重
        if drop_duplicates:
            before = len(result)
            result = result.drop_duplicates()
            stats["去重删除行数"] = before - len(result)

        # 缺失值统计
        missing_before = result.isnull().sum().sum()
        stats["填充前缺失值总数"] = int(missing_before)

        # 填充缺失值（数值列）
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if fill_missing == "mean" and num_cols:
            result[num_cols] = result[num_cols].fillna(result[num_cols].mean())
        elif fill_missing == "median" and num_cols:
            result[num_cols] = result[num_cols].fillna(result[num_cols].median())
        elif fill_missing == "mode":
            for col in result.columns:
                mode_val = result[col].mode()
                if not mode_val.empty:
                    result[col] = result[col].fillna(mode_val.iloc[0])
        elif fill_missing in ("zero", "0") and num_cols:
            result[num_cols] = result[num_cols].fillna(0)
        elif fill_missing == "drop":
            result = result.dropna()

        # 标准化（Z-score）
        if normalize and num_cols:
            for col in num_cols:
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - result[col].mean()) / std
            stats["已标准化列数"] = len(num_cols)

        valid = len(result)
        lines = [
            "## 🧹 数据清洗完成",
            f"- 原始行数：**{raw}**，清洗后：**{valid}**（删除 {raw - valid} 行）",
        ]
        if drop_empty_cols:
            dropped_list = stats.get("删除全空列", [])
            lines.append(f"- 删除全空列：**{len(dropped_list)}** 列"
                         + (f"（{', '.join(dropped_list[:5])}）" if dropped_list else "（无）"))
        for k, v in stats.items():
            if k not in ("删除全空列", "删除列数"):
                lines.append(f"- {k}：**{v}**")

        return ProcessResult(
            operation="clean",
            summary_text="\n".join(lines),
            result_df=result,
            raw_row_count=raw,
            valid_row_count=valid,
            extra=stats,
        )

    # ── 2. 数据查找 ──────────────────────────────────────

    @staticmethod
    def process_lookup(df: pd.DataFrame, condition: str) -> ProcessResult:
        """
        按条件筛选行。condition 格式见 _apply_condition。
        若条件解析失败（返回行数 == 原始行数 且条件非空），给出警告。
        """
        raw = len(df)
        result = _apply_condition(df, condition)
        valid = len(result)

        if condition.strip() and valid == raw:
            # 可能条件未被识别，添加提示
            warn = f"\n> ⚠️ 条件 `{condition}` 未能成功解析，返回全量数据。支持格式：`列名 > 值`、`列名包含关键词`、`列名最大的3` 等"
        else:
            warn = ""

        return ProcessResult(
            operation="lookup",
            summary_text=(
                f"## 🔍 数据查找结果\n"
                f"- 条件：`{condition}`\n"
                f"- 共找到 **{valid}** 行数据（原始 {raw} 行）"
                + warn
            ),
            result_df=result,
            raw_row_count=raw,
            valid_row_count=valid,
        )

    # ── 3. 数据计数 ──────────────────────────────────────

    @staticmethod
    def process_count(df: pd.DataFrame,
                      group_col: Optional[str] = None,
                      condition: str = "") -> ProcessResult:
        """
        计数：若指定 group_col，则分组计数；否则统计总行数或条件行数。
        """
        sub = _apply_condition(df, condition) if condition else df
        raw = len(df)

        if group_col and group_col in df.columns:
            cnt = sub.groupby(group_col).size().reset_index(name="计数")
            cnt = cnt.sort_values("计数", ascending=False)
            summary = (
                f"## 📊 分组计数（按 {group_col}）\n"
                f"- 共 **{len(cnt)}** 个分组，总计 **{len(sub)}** 行\n"
                f"- 最大组：**{cnt.iloc[0][group_col]}**（{cnt.iloc[0]['计数']} 条）"
            )
            return ProcessResult(
                operation="count",
                summary_text=summary,
                result_df=cnt,
                raw_row_count=raw,
                valid_row_count=len(sub),
            )
        else:
            count = len(sub)
            cond_desc = f"（条件：`{condition}`）" if condition else ""
            return ProcessResult(
                operation="count",
                summary_text=f"## 📊 数据计数\n- 数据总行数{cond_desc}：**{count}**",
                result_df=pd.DataFrame({"统计项": ["行数"], "值": [count]}),
                raw_row_count=raw,
                valid_row_count=count,
            )

    # ── 4. 多表合并 ──────────────────────────────────────

    @staticmethod
    def process_merge(dfs: list,
                      how: str = "concat",
                      on: Optional[str] = None) -> ProcessResult:
        """
        多表合并：
        - how="concat"：纵向拼接（堆叠）
        - how="left"/"inner"/"outer"：按 on 列横向合并
        """
        raw_total = sum(len(d) for d in dfs)
        if not dfs:
            return ProcessResult(operation="merge", summary_text="❌ 无可合并的数据表")

        if how == "concat":
            result = pd.concat(dfs, ignore_index=True)
            summary = (
                f"## 🔗 多表合并完成（纵向拼接）\n"
                f"- 合并 **{len(dfs)}** 张表，共 **{len(result)}** 行"
            )
        else:
            if len(dfs) < 2 or not on:
                return ProcessResult(operation="merge",
                                     summary_text="❌ 横向合并需要至少 2 张表且指定关联列")
            result = dfs[0]
            for d in dfs[1:]:
                result = pd.merge(result, d, on=on, how=how)
            summary = (
                f"## 🔗 多表合并完成（{how} join on `{on}`）\n"
                f"- 结果 **{len(result)}** 行 × {len(result.columns)} 列"
            )

        return ProcessResult(
            operation="merge",
            summary_text=summary,
            result_df=result,
            raw_row_count=raw_total,
            valid_row_count=len(result),
        )

    # ── 5. 多表拆分 ──────────────────────────────────────

    @staticmethod
    def process_split(df: pd.DataFrame, split_col: str) -> ProcessResult:
        """
        按 split_col 列的唯一值拆分为多个子表（每个值一张 Sheet）。
        """
        if split_col not in df.columns:
            return ProcessResult(
                operation="split",
                summary_text=f"❌ 列 `{split_col}` 不存在，无法拆分",
            )
        raw = len(df)
        groups = df.groupby(split_col)
        sheets = {str(name): grp.reset_index(drop=True) for name, grp in groups}

        summary_lines = [
            f"## ✂️ 数据拆分完成（按 {split_col}）",
            f"- 共拆分为 **{len(sheets)}** 个分组",
        ]
        for k, v in list(sheets.items())[:5]:
            summary_lines.append(f"  - `{k}`：{len(v)} 行")
        if len(sheets) > 5:
            summary_lines.append(f"  - ...（共 {len(sheets)} 组）")

        return ProcessResult(
            operation="split",
            summary_text="\n".join(summary_lines),
            result_sheets=sheets,
            raw_row_count=raw,
            valid_row_count=raw,
        )

    # ── 6. 求和 ──────────────────────────────────────────

    @staticmethod
    def calc_sum(df: pd.DataFrame,
                 value_col: str,
                 group_col: Optional[str] = None,
                 condition: str = "") -> ProcessResult:
        """
        求和：可指定分组列和过滤条件。
        """
        sub = _apply_condition(df, condition) if condition else df
        if value_col not in sub.columns:
            return ProcessResult(operation="calc_sum",
                                 summary_text=f"❌ 列 `{value_col}` 不存在")

        if group_col and group_col in sub.columns:
            result = sub.groupby(group_col)[value_col].sum().reset_index()
            result.columns = [group_col, f"{value_col}_求和"]
            result = result.sort_values(f"{value_col}_求和", ascending=False)
            total = result[f"{value_col}_求和"].sum()
            summary = (
                f"## ➕ 分组求和（{value_col} 按 {group_col}）\n"
                f"- 总计：**{total:,.4g}**，共 {len(result)} 组"
            )
        else:
            total = sub[value_col].sum()
            result = pd.DataFrame({"列": [value_col], "求和": [total]})
            summary = f"## ➕ 求和结果\n- `{value_col}` 求和：**{total:,.4g}**"
            if condition:
                summary += f"\n- 过滤条件：`{condition}`（{len(sub)} 行参与计算）"

        return ProcessResult(
            operation="calc_sum",
            summary_text=summary,
            result_df=result,
            raw_row_count=len(df),
            valid_row_count=len(sub),
        )

    # ── 7. 求平均值 ──────────────────────────────────────

    @staticmethod
    def calc_mean(df: pd.DataFrame,
                  value_col: str,
                  group_col: Optional[str] = None,
                  condition: str = "") -> ProcessResult:
        """求均值（可分组 / 可过滤）"""
        sub = _apply_condition(df, condition) if condition else df
        if value_col not in sub.columns:
            return ProcessResult(operation="calc_mean",
                                 summary_text=f"❌ 列 `{value_col}` 不存在")

        if group_col and group_col in sub.columns:
            result = sub.groupby(group_col)[value_col].mean().reset_index()
            result.columns = [group_col, f"{value_col}_均值"]
            result = result.sort_values(f"{value_col}_均值", ascending=False)
            overall = sub[value_col].mean()
            summary = (
                f"## ➗ 分组求平均（{value_col} 按 {group_col}）\n"
                f"- 全局均值：**{overall:,.4g}**，共 {len(result)} 组"
            )
        else:
            avg = sub[value_col].mean()
            result = pd.DataFrame({"列": [value_col], "均值": [avg]})
            summary = f"## ➗ 均值结果\n- `{value_col}` 均值：**{avg:,.4g}**"

        return ProcessResult(
            operation="calc_mean",
            summary_text=summary,
            result_df=result,
            raw_row_count=len(df),
            valid_row_count=len(sub),
        )

    # ── 8. 求极值 ──────────────────────────────────────

    @staticmethod
    def calc_extremes(df: pd.DataFrame,
                      value_col: str,
                      group_col: Optional[str] = None) -> ProcessResult:
        """求最大值、最小值、极差"""
        if value_col not in df.columns:
            return ProcessResult(operation="calc_extremes",
                                 summary_text=f"❌ 列 `{value_col}` 不存在")

        if group_col and group_col in df.columns:
            agg = df.groupby(group_col)[value_col].agg(["max", "min"]).reset_index()
            agg.columns = [group_col, "最大值", "最小值"]
            agg["极差"] = agg["最大值"] - agg["最小值"]
            summary = (
                f"## 📈 分组极值（{value_col} 按 {group_col}）\n"
                f"- 全局最大：**{df[value_col].max():,.4g}**，"
                f"全局最小：**{df[value_col].min():,.4g}**"
            )
            result = agg
        else:
            vmax = df[value_col].max()
            vmin = df[value_col].min()
            result = pd.DataFrame({
                "列": [value_col],
                "最大值": [vmax],
                "最小值": [vmin],
                "极差": [vmax - vmin],
            })
            summary = (
                f"## 📈 极值结果\n"
                f"- `{value_col}` 最大值：**{vmax:,.4g}**\n"
                f"- `{value_col}` 最小值：**{vmin:,.4g}**\n"
                f"- 极差：**{vmax - vmin:,.4g}**"
            )

        return ProcessResult(
            operation="calc_extremes",
            summary_text=summary,
            result_df=result,
            raw_row_count=len(df),
            valid_row_count=len(df),
        )

    # ── 9. 逻辑计算（IF-style 标签）──────────────────────

    @staticmethod
    def calc_logic(df: pd.DataFrame,
                   value_col: str,
                   condition: str,
                   true_label: str = "是",
                   false_label: str = "否",
                   new_col: str = "逻辑结果") -> ProcessResult:
        """
        根据条件为每行打标签（类似 Excel IF 函数）。
        condition 格式："> 100" / "== 0" / "包含 关键词"
        """
        result = df.copy()
        full_cond = f"{value_col} {condition}"
        matched = _apply_condition(df, full_cond).index
        result[new_col] = false_label
        result.loc[matched, new_col] = true_label

        true_cnt = (result[new_col] == true_label).sum()
        return ProcessResult(
            operation="calc_logic",
            summary_text=(
                f"## 🔀 逻辑计算完成\n"
                f"- 条件：`{value_col} {condition}`\n"
                f"- 满足条件（{true_label}）：**{true_cnt}** 行\n"
                f"- 不满足（{false_label}）：**{len(df) - true_cnt}** 行\n"
                f"- 结果写入新列 `{new_col}`"
            ),
            result_df=result,
            raw_row_count=len(df),
            valid_row_count=len(df),
        )

    # ── 10. 合并计算（多指标一次性聚合）────────────────

    @staticmethod
    def calc_aggregate(df: pd.DataFrame,
                       value_cols: list,
                       group_col: Optional[str] = None,
                       agg_funcs: Optional[list] = None) -> ProcessResult:
        """
        合并计算：对多个数值列同时计算多种聚合指标（sum/mean/max/min/count）。
        """
        if agg_funcs is None:
            agg_funcs = ["sum", "mean", "max", "min", "count"]

        valid_cols = [c for c in value_cols if c in df.columns]
        if not valid_cols:
            return ProcessResult(operation="calc_aggregate",
                                 summary_text="❌ 指定的值列均不存在")

        func_map = {"sum": "sum", "mean": "mean", "max": "max",
                    "min": "min", "count": "count"}
        funcs = [func_map[f] for f in agg_funcs if f in func_map]

        if group_col and group_col in df.columns:
            result = df.groupby(group_col)[valid_cols].agg(funcs)
            result.columns = ["_".join(c) for c in result.columns]
            result = result.reset_index()
        else:
            agg_dict = {}
            for col in valid_cols:
                for fn in funcs:
                    agg_dict[f"{col}_{fn}"] = getattr(df[col], fn)()
            result = pd.DataFrame([agg_dict])

        summary = (
            f"## 📋 合并计算结果\n"
            f"- 计算列：{', '.join(f'`{c}`' for c in valid_cols)}\n"
            f"- 聚合指标：{', '.join(agg_funcs)}\n"
            f"- {'按 `' + group_col + '` 分组' if group_col else '全量汇总'}"
        )

        return ProcessResult(
            operation="calc_aggregate",
            summary_text=summary,
            result_df=result,
            raw_row_count=len(df),
            valid_row_count=len(df),
        )
