# Full Feature Expansion: Excel处理 + 数据运算 + 分析整合 + ECharts图表

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 对标 ChatExcel，完整实现四大能力模块（Excel处理、数据运算、数据分析整合、ECharts交互图表），并将首页和分析页改造为 ChatExcel 风格的能力展示面板。

**Architecture:**
- 新增 `core/excel_processor.py`（Excel处理 + 数据运算，基于 pandas/openpyxl）
- 新增 `core/chart_builder.py`（ECharts JSON 生成，前端用 ECharts.js 渲染）
- 新增 `/api/process` 接口（处理/运算类操作，返回表格数据 + 可下载 Excel）
- 新增 `/api/chart` 接口（返回 ECharts option JSON）
- 扩展 `/api/analyze`：新增对比分析、交叉分析模式
- 前端 `home.html` 改为四列能力面板，`analyze.html` 聊天流支持所有操作

**Tech Stack:** pandas, openpyxl（已有）；ECharts.js 5.x CDN（前端渲染，替代 matplotlib）；Alpine.js（已有）；FastAPI（已有）

---

## 模块总览

```
四大能力模块
├── Excel处理          → core/excel_processor.py  + /api/process
│   ├── 数据清洗       deduplicate + fill_missing + normalize
│   ├── 数据查找       filter rows by condition
│   ├── 数据计数       count / groupby count
│   ├── 多表合并       concat / merge multiple files or sheets
│   └── 多表拆分       split by column value → multiple sheets
│
├── 数据运算          → core/excel_processor.py  + /api/process
│   ├── 求和           sum / sumif by group
│   ├── 求平均值       mean / averageif by group
│   ├── 求极值         min / max
│   ├── 逻辑计算       IF-style condition labeling
│   └── 合并计算       multi-aggregate in one pass
│
├── 数据分析          → core/analysis_engine.py（扩展）+ /api/analyze
│   ├── 对比分析       compare two groups/periods  [新增]
│   ├── 交叉分析       pivot table / crosstab       [新增]
│   ├── 关联分析       correlation matrix           [已有: y_vs_all/multi_x_vs_y]
│   ├── 相关性分析     Pearson/Spearman             [已有: two_column]
│   └── 线性回归       Linear regression            [已有: multi_x_vs_y]
│
└── 图表生成          → core/chart_builder.py      + /api/chart
    ├── 柱状图         ECharts bar
    ├── 饼图           ECharts pie
    ├── 折线图         ECharts line
    ├── 面积图         ECharts line(area)
    └── 组合图         ECharts bar+line combo
```

---

## Task 1：新增 ProcessResult 数据类 + 依赖检查

**Files:**
- Create: `core/process_result.py`
- Verify: `requirements.txt`（openpyxl 已有，无需新增依赖）

### core/process_result.py 完整代码

```python
"""
Excel 处理 / 数据运算的统一结果容器
与 AnalysisResult 分开，因为处理类操作返回的是变换后的 DataFrame
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class ProcessResult:
    """Excel处理 / 数据运算 结果"""
    operation: str              # 操作类型标识符
    summary_text: str = ""      # Markdown 摘要（显示在聊天气泡）
    result_df: Optional[pd.DataFrame] = None   # 处理后的数据表
    download_path: Optional[str] = None        # 可下载 Excel 路径
    row_in: int = 0             # 输入行数
    row_out: int = 0            # 输出行数
    meta: dict = field(default_factory=dict)   # 附加信息（如合并文件数等）
```

**验证：**
```bash
python -c "from core.process_result import ProcessResult; print('OK')"
```

**提交：**
```bash
git add core/process_result.py
git commit -m "feat: add ProcessResult dataclass for excel processing operations"
```

---

## Task 2：新建 core/excel_processor.py（Excel处理 + 数据运算，共 10 个函数）

**Files:**
- Create: `core/excel_processor.py`

### 完整代码

```python
"""
Excel 处理与数据运算模块
实现 ChatExcel 四大能力中的前两列：Excel处理 + 数据运算
所有函数返回 ProcessResult
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.process_result import ProcessResult


# ══════════════════════════════════════════════════════
#  Excel 处理（5 个功能）
# ══════════════════════════════════════════════════════

def process_clean(df: pd.DataFrame) -> ProcessResult:
    """
    数据清洗：
    1. 删除完全重复行
    2. 统计并填充缺失值（数值列用中位数，文字列用众数）
    3. 去除列名首尾空格
    """
    row_in = len(df)
    result = df.copy()

    # 删除重复行
    dup_count = result.duplicated().sum()
    result = result.drop_duplicates()

    # 填充缺失值
    filled = {}
    for col in result.columns:
        missing = result[col].isna().sum()
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            fill_val = result[col].median()
            result[col] = result[col].fillna(fill_val)
            filled[col] = f"中位数 {fill_val:.4g}"
        else:
            fill_val = result[col].mode().iloc[0] if not result[col].mode().empty else ""
            result[col] = result[col].fillna(fill_val)
            filled[col] = f"众数 '{fill_val}'"

    row_out = len(result)
    lines = [
        "## 数据清洗结果",
        f"- 原始行数：{row_in}",
        f"- 删除重复行：{dup_count} 行",
        f"- 清洗后行数：{row_out}",
    ]
    if filled:
        lines.append("\n**缺失值填充：**")
        for col, info in filled.items():
            lines.append(f"- `{col}` → {info}")
    else:
        lines.append("- 无缺失值需要填充 ✅")

    return ProcessResult(
        operation="clean",
        summary_text="\n".join(lines),
        result_df=result,
        row_in=row_in, row_out=row_out,
        meta={"dup_removed": int(dup_count), "filled": filled},
    )


def process_lookup(df: pd.DataFrame, condition: str) -> ProcessResult:
    """
    数据查找：根据自然语言条件筛选行
    支持简单条件：列名 > 值, 列名 == 值, 列名 包含 文字
    condition 示例：'销售额 > 100', '地区 == 北京', '姓名 包含 张'
    """
    row_in = len(df)
    result = _apply_condition(df, condition)
    row_out = len(result)

    return ProcessResult(
        operation="lookup",
        summary_text=(
            f"## 数据查找结果\n\n"
            f"条件：`{condition}`\n\n"
            f"共找到 **{row_out}** 条记录（原始 {row_in} 行）"
        ),
        result_df=result,
        row_in=row_in, row_out=row_out,
    )


def process_count(df: pd.DataFrame, group_col: Optional[str] = None,
                  condition: Optional[str] = None) -> ProcessResult:
    """
    数据计数：
    - group_col 指定时：按列分组计数（类似 COUNTIF）
    - condition 指定时：统计满足条件的行数
    - 两者都不指定：统计各列非空数量
    """
    row_in = len(df)
    if group_col and group_col in df.columns:
        result = df.groupby(group_col).size().reset_index(name="计数")
        result = result.sort_values("计数", ascending=False)
        summary = (
            f"## 数据计数结果\n\n"
            f"按 `{group_col}` 分组，共 **{len(result)}** 个类别：\n\n"
            + result.head(20).to_markdown(index=False)
        )
    elif condition:
        filtered = _apply_condition(df, condition)
        result = pd.DataFrame({"条件": [condition], "计数": [len(filtered)]})
        summary = (
            f"## 数据计数结果\n\n"
            f"条件 `{condition}` 满足 **{len(filtered)}** 行 / 共 {row_in} 行"
        )
    else:
        result = pd.DataFrame({
            "列名": df.columns,
            "非空数量": [df[c].notna().sum() for c in df.columns],
            "缺失数量": [df[c].isna().sum() for c in df.columns],
            "唯一值数": [df[c].nunique() for c in df.columns],
        })
        summary = f"## 各列计数统计\n\n" + result.to_markdown(index=False)

    return ProcessResult(
        operation="count",
        summary_text=summary,
        result_df=result,
        row_in=row_in, row_out=len(result),
    )


def process_merge(dfs: list[pd.DataFrame], names: list[str],
                  how: str = "vertical") -> ProcessResult:
    """
    多表合并：
    - how='vertical'  纵向合并（行追加，列名相同时对齐）
    - how='horizontal' 横向合并（列追加，按索引对齐）
    """
    row_in = sum(len(d) for d in dfs)
    if how == "vertical":
        result = pd.concat(dfs, ignore_index=True)
    else:
        result = pd.concat(dfs, axis=1)
    row_out = len(result)

    return ProcessResult(
        operation="merge",
        summary_text=(
            f"## 多表合并结果（{how}）\n\n"
            f"- 合并表数：{len(dfs)} 张（{', '.join(names)}）\n"
            f"- 合并前总行数：{row_in}\n"
            f"- 合并后行数：{row_out}，列数：{len(result.columns)}"
        ),
        result_df=result,
        row_in=row_in, row_out=row_out,
        meta={"tables": names, "how": how},
    )


def process_split(df: pd.DataFrame, split_col: str) -> ProcessResult:
    """
    多表拆分：按 split_col 列的不同值拆分为多个 DataFrame
    返回 meta['sheets'] = {值: df}，调用方负责写入多 Sheet Excel
    """
    if split_col not in df.columns:
        raise ValueError(f"列 '{split_col}' 不存在")
    groups = {str(k): v.reset_index(drop=True) for k, v in df.groupby(split_col)}
    summary = (
        f"## 多表拆分结果\n\n"
        f"按 `{split_col}` 列拆分为 **{len(groups)}** 个 Sheet：\n\n"
        + "\n".join(f"- `{k}`：{len(v)} 行" for k, v in groups.items())
    )
    return ProcessResult(
        operation="split",
        summary_text=summary,
        result_df=df,
        row_in=len(df), row_out=len(df),
        meta={"split_col": split_col, "sheets": groups},
    )


# ══════════════════════════════════════════════════════
#  数据运算（5 个功能）
# ══════════════════════════════════════════════════════

def calc_sum(df: pd.DataFrame, value_col: str,
             group_col: Optional[str] = None) -> ProcessResult:
    """求和（可分组）"""
    if group_col and group_col in df.columns:
        result = df.groupby(group_col)[value_col].sum().reset_index()
        result.columns = [group_col, f"{value_col}_求和"]
        result = result.sort_values(f"{value_col}_求和", ascending=False)
        total = df[value_col].sum()
        summary = (
            f"## 分组求和：{value_col}\n\n"
            f"按 `{group_col}` 分组，总计 **{total:,.4g}**\n\n"
            + result.head(20).to_markdown(index=False)
        )
    else:
        total = df[value_col].sum()
        result = pd.DataFrame({"列": [value_col], "求和": [total]})
        summary = f"## 求和结果\n\n`{value_col}` 列总计：**{total:,.4g}**"

    return ProcessResult(
        operation="sum",
        summary_text=summary,
        result_df=result,
        row_in=len(df), row_out=len(result),
        meta={"total": float(total)},
    )


def calc_mean(df: pd.DataFrame, value_col: str,
              group_col: Optional[str] = None) -> ProcessResult:
    """求平均值（可分组）"""
    if group_col and group_col in df.columns:
        result = df.groupby(group_col)[value_col].mean().reset_index()
        result.columns = [group_col, f"{value_col}_均值"]
        result = result.sort_values(f"{value_col}_均值", ascending=False)
        summary = (
            f"## 分组均值：{value_col}\n\n"
            f"全局均值：**{df[value_col].mean():,.4g}**\n\n"
            + result.head(20).to_markdown(index=False)
        )
    else:
        mean_val = df[value_col].mean()
        result = pd.DataFrame({"列": [value_col], "均值": [mean_val]})
        summary = f"## 均值结果\n\n`{value_col}` 列平均值：**{mean_val:,.4g}**"

    return ProcessResult(
        operation="mean",
        summary_text=summary,
        result_df=result,
        row_in=len(df), row_out=len(result),
    )


def calc_extremes(df: pd.DataFrame, value_col: str,
                  group_col: Optional[str] = None) -> ProcessResult:
    """求极值（最大值、最小值，可分组）"""
    if group_col and group_col in df.columns:
        result = df.groupby(group_col)[value_col].agg(["min", "max", "mean"]).reset_index()
        result.columns = [group_col, "最小值", "最大值", "均值"]
        summary = (
            f"## 分组极值：{value_col}\n\n"
            f"全局最大：**{df[value_col].max():,.4g}**，"
            f"全局最小：**{df[value_col].min():,.4g}**\n\n"
            + result.head(20).to_markdown(index=False)
        )
    else:
        result = pd.DataFrame({
            "列": [value_col],
            "最大值": [df[value_col].max()],
            "最小值": [df[value_col].min()],
            "极差": [df[value_col].max() - df[value_col].min()],
        })
        summary = (
            f"## 极值结果\n\n"
            f"`{value_col}` 列：最大 **{result['最大值'][0]:,.4g}**，"
            f"最小 **{result['最小值'][0]:,.4g}**，"
            f"极差 **{result['极差'][0]:,.4g}**"
        )

    return ProcessResult(
        operation="extremes",
        summary_text=summary,
        result_df=result,
        row_in=len(df), row_out=len(result),
    )


def calc_logic(df: pd.DataFrame, condition: str,
               label_true: str = "是", label_false: str = "否",
               new_col: str = "判断结果") -> ProcessResult:
    """
    逻辑计算：根据条件给每行打标签
    condition: '销售额 > 100', '地区 == 北京'
    """
    result = df.copy()
    mask = _apply_condition(df, condition).index
    result[new_col] = label_false
    result.loc[mask, new_col] = label_true
    true_count = (result[new_col] == label_true).sum()

    return ProcessResult(
        operation="logic",
        summary_text=(
            f"## 逻辑计算结果\n\n"
            f"条件：`{condition}`\n"
            f"- 满足（{label_true}）：**{true_count}** 行\n"
            f"- 不满足（{label_false}）：**{len(df) - true_count}** 行\n\n"
            f"新列 `{new_col}` 已追加到数据末尾"
        ),
        result_df=result,
        row_in=len(df), row_out=len(result),
        meta={"true_count": int(true_count), "new_col": new_col},
    )


def calc_aggregate(df: pd.DataFrame, value_cols: list[str],
                   group_col: Optional[str] = None) -> ProcessResult:
    """
    合并计算：对多列同时做 sum/mean/min/max 聚合
    """
    agg_funcs = {"sum": "求和", "mean": "均值", "min": "最小值", "max": "最大值"}
    valid_cols = [c for c in value_cols if c in df.columns]
    if not valid_cols:
        raise ValueError("没有可用的数值列")

    if group_col and group_col in df.columns:
        result = df.groupby(group_col)[valid_cols].agg(["sum", "mean", "min", "max"])
        result.columns = [f"{c}_{agg_funcs[f]}" for c, f in result.columns]
        result = result.reset_index()
    else:
        rows = []
        for col in valid_cols:
            rows.append({
                "列名": col,
                "求和": df[col].sum(),
                "均值": df[col].mean(),
                "最小值": df[col].min(),
                "最大值": df[col].max(),
            })
        result = pd.DataFrame(rows)

    summary = (
        f"## 合并计算结果\n\n"
        f"分析列：{', '.join(f'`{c}`' for c in valid_cols)}\n"
        + (f"分组：`{group_col}`\n" if group_col else "") + "\n"
        + result.head(20).to_markdown(index=False)
    )

    return ProcessResult(
        operation="aggregate",
        summary_text=summary,
        result_df=result,
        row_in=len(df), row_out=len(result),
    )


# ══════════════════════════════════════════════════════
#  内部工具
# ══════════════════════════════════════════════════════

def _apply_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    解析简单条件字符串，返回筛选后的 DataFrame
    支持：col > val, col < val, col >= val, col <= val,
          col == val, col != val, col 包含 text, col 不包含 text
    """
    condition = condition.strip()

    # 中文包含/不包含
    m = re.match(r"(.+?)\s*(包含|不包含)\s*(.+)", condition)
    if m:
        col, op, val = m.group(1).strip(), m.group(2), m.group(3).strip()
        if col in df.columns:
            mask = df[col].astype(str).str.contains(val, na=False)
            return df[~mask] if op == "不包含" else df[mask]

    # 数值比较
    m = re.match(r"(.+?)\s*(>=|<=|!=|==|>|<)\s*(.+)", condition)
    if m:
        col, op, val = m.group(1).strip(), m.group(2), m.group(3).strip()
        if col in df.columns:
            try:
                val_num = float(val)
                ops = {">": df[col] > val_num, "<": df[col] < val_num,
                       ">=": df[col] >= val_num, "<=": df[col] <= val_num,
                       "==": df[col] == val_num, "!=": df[col] != val_num}
                return df[ops[op]]
            except ValueError:
                # 字符串比较
                ops = {"==": df[col] == val, "!=": df[col] != val,
                       ">": df[col] > val, "<": df[col] < val}
                return df[ops.get(op, df[col] == val)]

    # 无法解析时返回原表
    return df
```

**验证：**
```bash
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from core.excel_processor import (process_clean, process_lookup, process_count,
                                   calc_sum, calc_mean, calc_extremes,
                                   calc_logic, calc_aggregate)
df = pd.DataFrame({'区域':['北京','上海','北京','广州'], '销售额':[100,200,150,80], '利润':[20,40,30,10]})
r = process_clean(df); assert r.operation == 'clean'
r = process_lookup(df, '销售额 > 100'); assert r.row_out == 2
r = calc_sum(df, '销售额', '区域'); assert '北京' in r.result_df['区域'].values
r = calc_logic(df, '销售额 > 100'); assert '判断结果' in r.result_df.columns
print('excel_processor OK')
"
```

**提交：**
```bash
git add core/excel_processor.py
git commit -m "feat: add excel_processor with 10 operations (5 excel + 5 calc)"
```

---

## Task 3：扩展 analysis_engine.py — 新增对比分析、交叉分析

**Files:**
- Modify: `core/analysis_engine.py`（末尾追加两个函数）

### 追加到 analysis_engine.py 末尾

```python
# ──────────────────────────────────────────────────────────
# 对比分析（compare two groups or periods）
# ──────────────────────────────────────────────────────────

def analyze_compare(df: pd.DataFrame, value_col: str,
                    group_col: str) -> AnalysisResult:
    """
    对比分析：比较两个组/多个时间段的均值、分布差异
    使用 t 检验（2组）或 ANOVA（多组）
    """
    from scipy import stats as sp_stats

    groups = {k: v[value_col].dropna() for k, v in df.groupby(group_col)}
    group_names = list(groups.keys())
    n_groups = len(group_names)

    # 描述统计
    desc = []
    for name, vals in groups.items():
        desc.append({
            "组别": name,
            "样本量": len(vals),
            "均值": round(vals.mean(), 4),
            "中位数": round(vals.median(), 4),
            "标准差": round(vals.std(), 4),
        })
    desc_df = pd.DataFrame(desc)

    # 显著性检验
    if n_groups == 2:
        t_stat, p_val = sp_stats.ttest_ind(*[list(groups[g]) for g in group_names])
        test_name = "独立样本 t 检验"
        stat_label, stat_val = "t 统计量", t_stat
    else:
        f_stat, p_val = sp_stats.f_oneway(*[list(groups[g]) for g in group_names])
        test_name = "单因素 ANOVA"
        stat_label, stat_val = "F 统计量", f_stat

    sig = "**显著**（p<0.05）" if p_val < 0.05 else "不显著（p≥0.05）"

    lines = [
        f"## 对比分析：{value_col}（按 {group_col} 分组）",
        "",
        desc_df.to_markdown(index=False),
        "",
        f"**{test_name}**：{stat_label} = {stat_val:.4f}，p = {p_val:.4f}",
        f"→ 组间差异{sig}",
    ]
    summary = "\n".join(lines)

    result = AnalysisResult(mode="compare", target_y=value_col, summary_text=summary)
    result.anova_group_col = group_col
    result.anova_f_stat = stat_val
    result.anova_p_value = p_val
    result.anova_group_stats_df = desc_df
    return result


# ──────────────────────────────────────────────────────────
# 交叉分析（pivot table / crosstab）
# ──────────────────────────────────────────────────────────

def analyze_crosstab(df: pd.DataFrame, row_col: str,
                     col_col: str, value_col: Optional[str] = None,
                     aggfunc: str = "mean") -> AnalysisResult:
    """
    交叉分析：生成透视表
    - value_col 为空时：计数交叉表
    - value_col 有值时：对数值列按 aggfunc 聚合
    """
    if value_col and value_col in df.columns:
        agg_map = {"mean": np.mean, "sum": np.sum, "count": len,
                   "max": np.max, "min": np.min}
        pivot = pd.pivot_table(df, values=value_col,
                               index=row_col, columns=col_col,
                               aggfunc=agg_map.get(aggfunc, np.mean),
                               fill_value=0)
        title = f"{value_col} 交叉表（{aggfunc}）"
    else:
        pivot = pd.crosstab(df[row_col], df[col_col])
        title = f"{row_col} × {col_col} 频次交叉表"

    summary = (
        f"## {title}\n\n"
        f"行：`{row_col}`（{pivot.shape[0]} 个类别）  "
        f"列：`{col_col}`（{pivot.shape[1]} 个类别）\n\n"
        + pivot.to_markdown()
    )

    result = AnalysisResult(mode="crosstab", target_y=value_col or row_col,
                            summary_text=summary)
    result.stats_df = pivot.reset_index()
    return result
```

**验证：**
```bash
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
from core.analysis_engine import analyze_compare, analyze_crosstab
df = pd.DataFrame({'区域':['北京','上海','北京','广州','上海'], '销售额':[100,200,150,80,180], '季度':['Q1','Q1','Q2','Q2','Q2']})
r = analyze_compare(df, '销售额', '区域'); assert 'p =' in r.summary_text
r = analyze_crosstab(df, '区域', '季度'); assert r.mode == 'crosstab'
print('compare + crosstab OK')
"
```

**提交：**
```bash
git add core/analysis_engine.py
git commit -m "feat: add analyze_compare and analyze_crosstab to analysis_engine"
```

---

## Task 4：新建 core/chart_builder.py（ECharts JSON 生成）

**Files:**
- Create: `core/chart_builder.py`

> **注意：** 使用 ECharts 而非 matplotlib，后端只生成 JSON option，前端用 ECharts.js 渲染交互图表。

```python
"""
图表生成器 — 生成 ECharts option JSON，由前端渲染
支持：bar（柱状图）、pie（饼图）、line（折线图）、
      area（面积图）、combo（组合图柱+线）
"""
from typing import Optional
import pandas as pd
import numpy as np

SUPPORTED_CHART_TYPES = {
    "bar":   "柱状图",
    "pie":   "饼图",
    "line":  "折线图",
    "area":  "面积图",
    "combo": "组合图（柱+线）",
}

# 默认颜色
COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#C00000", "#7030A0", "#00B0F0", "#FFC000"]


def _tooltip(trigger="axis"):
    return {"trigger": trigger, "axisPointer": {"type": "shadow"}}


def _legend(data: list):
    return {"data": data, "top": 8}


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    y2_col: Optional[str] = None,
    y_cols: Optional[list[str]] = None,
    title: str = "",
    color_scheme: str = "default",
) -> dict:
    """
    生成 ECharts option 字典。
    前端通过 echarts.init(dom).setOption(option) 渲染。
    """
    if chart_type == "bar":
        return _bar(df, x_col, y_col, title)
    elif chart_type == "pie":
        return _pie(df, x_col, y_col, title)
    elif chart_type == "line":
        return _line(df, x_col, y_cols or ([y_col] if y_col else []), title, area=False)
    elif chart_type == "area":
        return _line(df, x_col, y_cols or ([y_col] if y_col else []), title, area=True)
    elif chart_type == "combo":
        return _combo(df, x_col, y_col, y2_col, title)
    else:
        raise ValueError(f"不支持的图表类型：{chart_type}")


def _x_data(df, x_col):
    if x_col and x_col in df.columns:
        return [str(v) for v in df[x_col].tolist()]
    return [str(i) for i in range(len(df))]


def _bar(df, x_col, y_col, title):
    if not y_col:
        raise ValueError("柱状图需要指定 y_col")
    return {
        "title": {"text": title or f"{y_col} 柱状图", "left": "center"},
        "tooltip": _tooltip("axis"),
        "color": COLORS,
        "grid": {"left": "5%", "right": "5%", "bottom": "15%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": _x_data(df, x_col),
            "axisLabel": {"rotate": 30, "interval": 0 if len(df) <= 20 else "auto"},
        },
        "yAxis": {"type": "value", "name": y_col},
        "series": [{
            "name": y_col,
            "type": "bar",
            "data": df[y_col].round(4).tolist(),
            "itemStyle": {"color": COLORS[0]},
            "label": {"show": len(df) <= 15, "position": "top"},
        }],
    }


def _pie(df, x_col, y_col, title):
    if not y_col:
        raise ValueError("饼图需要指定 y_col（数值列）")
    labels = _x_data(df, x_col)
    values = df[y_col].abs().round(4).tolist()
    data = [{"name": l, "value": v} for l, v in zip(labels, values)]
    return {
        "title": {"text": title or f"{y_col} 饼图", "left": "center"},
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
        "legend": {"orient": "vertical", "left": "left", "top": "middle"},
        "color": COLORS,
        "series": [{
            "name": y_col,
            "type": "pie",
            "radius": ["35%", "65%"],
            "center": ["60%", "55%"],
            "data": data,
            "label": {"formatter": "{b}\n{d}%"},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowOffsetX": 0,
                                        "shadowColor": "rgba(0,0,0,0.5)"}},
        }],
    }


def _line(df, x_col, y_col_list, title, area=False):
    if not y_col_list:
        raise ValueError("折线/面积图需要指定 y_col 或 y_cols")
    series = []
    for i, ycol in enumerate(y_col_list):
        s = {
            "name": ycol,
            "type": "line",
            "data": df[ycol].round(4).tolist(),
            "smooth": True,
            "symbol": "circle",
            "symbolSize": 5,
            "lineStyle": {"width": 2, "color": COLORS[i % len(COLORS)]},
            "itemStyle": {"color": COLORS[i % len(COLORS)]},
        }
        if area:
            s["areaStyle"] = {"opacity": 0.15, "color": COLORS[i % len(COLORS)]}
        series.append(s)
    chart_name = "面积图" if area else "折线图"
    return {
        "title": {"text": title or f"{', '.join(y_col_list)} {chart_name}", "left": "center"},
        "tooltip": _tooltip("axis"),
        "legend": _legend(y_col_list) if len(y_col_list) > 1 else {"show": False},
        "color": COLORS,
        "grid": {"left": "5%", "right": "5%", "bottom": "15%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": _x_data(df, x_col),
            "boundaryGap": False,
            "axisLabel": {"rotate": 30, "interval": "auto"},
        },
        "yAxis": {"type": "value", "name": y_col_list[0] if len(y_col_list) == 1 else ""},
        "series": series,
    }


def _combo(df, x_col, bar_col, line_col, title):
    if not bar_col or not line_col:
        raise ValueError("组合图需要指定 y_col（柱）和 y2_col（线）")
    return {
        "title": {"text": title or f"{bar_col} & {line_col} 组合图", "left": "center"},
        "tooltip": _tooltip("axis"),
        "legend": _legend([bar_col, line_col]),
        "color": COLORS,
        "grid": {"left": "5%", "right": "8%", "bottom": "15%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": _x_data(df, x_col),
            "axisLabel": {"rotate": 30, "interval": "auto"},
        },
        "yAxis": [
            {"type": "value", "name": bar_col, "position": "left"},
            {"type": "value", "name": line_col, "position": "right",
             "splitLine": {"show": False}},
        ],
        "series": [
            {
                "name": bar_col,
                "type": "bar",
                "yAxisIndex": 0,
                "data": df[bar_col].round(4).tolist(),
                "itemStyle": {"color": COLORS[0]},
            },
            {
                "name": line_col,
                "type": "line",
                "yAxisIndex": 1,
                "data": df[line_col].round(4).tolist(),
                "smooth": True,
                "symbol": "circle",
                "lineStyle": {"color": COLORS[1], "width": 2.5},
                "itemStyle": {"color": COLORS[1]},
            },
        ],
    }
```

**验证：**
```bash
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from core.chart_builder import generate_chart, SUPPORTED_CHART_TYPES
df = pd.DataFrame({'月份':['1月','2月','3月'], '销售额':[100,150,120], '利润':[20,35,28]})
for ct in SUPPORTED_CHART_TYPES:
    kw = {'y_col': '销售额', 'x_col': '月份'}
    if ct == 'combo': kw['y2_col'] = '利润'
    opt = generate_chart(df, ct, **kw)
    assert 'series' in opt, f'{ct} missing series'
print('chart_builder OK types:', list(SUPPORTED_CHART_TYPES.keys()))
"
```

**提交：**
```bash
git add core/chart_builder.py
git commit -m "feat: add ECharts-based chart_builder (bar/pie/line/area/combo)"
```

---

## Task 5：result_formatter.py 扩展对比/交叉格式化器

**Files:**
- Modify: `core/result_formatter.py`

在 `detect_focus` 末尾的 `elif mode in (...)` 前追加：

```python
elif mode == "compare":
    return "default"

elif mode == "crosstab":
    return "default"
```

在 `format_result` 的 `formatters` 字典追加：

```python
"compare": {
    "default": lambda a: a.summary_text,
},
"crosstab": {
    "default": lambda a: a.summary_text,
},
```

**验证：**
```bash
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from core.analysis_engine import analyze_compare
from core.result_formatter import format_result
df = pd.DataFrame({'区域':['北京','上海','北京'], '销售额':[100,200,150]})
r = analyze_compare(df, '销售额', '区域')
out = format_result(r, 'default')
assert '均值' in out or '对比' in out
print('formatter OK')
"
```

**提交：**
```bash
git add core/result_formatter.py
git commit -m "feat: add compare/crosstab formatters to result_formatter"
```

---

## Task 6：新增 /api/process 和 /api/chart 接口

**Files:**
- Modify: `api/routes.py`（在末尾追加两个接口）

### 6.1 /api/process 接口

```python
# ──────────────────────────────────────────────────────────
# API：Excel处理 + 数据运算
# ──────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    file_id: str
    operation: str          # clean/lookup/count/merge/split/sum/mean/extremes/logic/aggregate
    value_col: Optional[str] = None
    group_col: Optional[str] = None
    x_col: Optional[str] = None      # split 用 split_col
    condition: Optional[str] = None  # lookup/count/logic 用
    label_true: str = "是"
    label_false: str = "否"
    new_col: str = "判断结果"
    y_cols: list[str] = []           # aggregate 多列
    aggfunc: str = "mean"
    file_ids: list[str] = []         # merge 多文件
    how: str = "vertical"
    use_ai: bool = False
    instruction: str = ""


@router.post("/api/process")
async def api_process(req: ProcessRequest):
    """执行 Excel 处理 / 数据运算，返回结果摘要 + 可下载 Excel"""
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")

    file_path = _uploaded_files[req.file_id]
    try:
        df_raw, all_cols = load_excel(str(file_path))
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    # AI 解析操作类型和参数
    operation = req.operation
    value_col = req.value_col
    group_col = req.group_col
    condition = req.condition

    if req.use_ai and req.instruction.strip():
        inst = req.instruction.lower()
        numeric_cols = get_numeric_columns(df_raw)
        # 自动推断 operation
        _op_map = [
            (["清洗", "去重", "缺失", "脏数据", "clean"],                 "clean"),
            (["查找", "筛选", "过滤", "找出", "lookup"],                  "lookup"),
            (["计数", "count", "数量", "多少个"],                          "count"),
            (["合并", "merge", "纵向", "横向", "拼接"],                    "merge"),
            (["拆分", "split", "分表", "按.*拆"],                          "split"),
            (["求和", "sum", "总和", "合计"],                              "sum"),
            (["均值", "平均", "mean", "average"],                          "mean"),
            (["极值", "最大", "最小", "max", "min"],                       "extremes"),
            (["逻辑", "判断", "标记", "if ", "条件"],                      "logic"),
            (["聚合", "汇总", "综合统计", "aggregate"],                    "aggregate"),
        ]
        for keywords, op in _op_map:
            if any(k in inst for k in keywords):
                operation = op
                break

        # 从列名列表中尝试匹配 value_col / group_col
        if not value_col:
            for col in numeric_cols:
                if col.lower() in inst or inst.find(col) >= 0:
                    value_col = col
                    break
        if not group_col:
            text_cols = [c for c in all_cols if c not in numeric_cols]
            for col in text_cols:
                if col in inst:
                    group_col = col
                    break

    from core.excel_processor import (
        process_clean, process_lookup, process_count, process_split,
        calc_sum, calc_mean, calc_extremes, calc_logic, calc_aggregate,
    )
    try:
        if operation == "clean":
            pr = process_clean(df_raw)
        elif operation == "lookup":
            if not condition:
                return JSONResponse({"success": False, "error": "lookup 需要指定 condition"})
            pr = process_lookup(df_raw, condition)
        elif operation == "count":
            pr = process_count(df_raw, group_col, condition)
        elif operation == "split":
            split_col = req.x_col or group_col
            if not split_col:
                return JSONResponse({"success": False, "error": "split 需要指定拆分列"})
            pr = process_split(df_raw, split_col)
        elif operation == "sum":
            if not value_col:
                return JSONResponse({"success": False, "error": "求和需要指定 value_col"})
            pr = calc_sum(df_raw, value_col, group_col)
        elif operation == "mean":
            if not value_col:
                return JSONResponse({"success": False, "error": "均值需要指定 value_col"})
            pr = calc_mean(df_raw, value_col, group_col)
        elif operation == "extremes":
            if not value_col:
                return JSONResponse({"success": False, "error": "极值需要指定 value_col"})
            pr = calc_extremes(df_raw, value_col, group_col)
        elif operation == "logic":
            if not condition:
                return JSONResponse({"success": False, "error": "逻辑计算需要指定 condition"})
            pr = calc_logic(df_raw, condition, req.label_true, req.label_false, req.new_col)
        elif operation == "aggregate":
            cols = req.y_cols or [v for v in [value_col] if v]
            if not cols:
                cols = get_numeric_columns(df_raw)
            pr = calc_aggregate(df_raw, cols, group_col)
        else:
            return JSONResponse({"success": False, "error": f"未知操作：{operation}"})
    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": f"操作失败：{str(e)}\n{traceback.format_exc()}"})

    # 将结果写成 Excel
    download_filename = None
    try:
        out_path = get_output_path(file_path.name, operation)
        if operation == "split" and pr.meta.get("sheets"):
            with pd.ExcelWriter(str(out_path), engine="openpyxl") as writer:
                for sheet_name, sheet_df in pr.meta["sheets"].items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        elif pr.result_df is not None:
            pr.result_df.to_excel(str(out_path), index=False)
        cleanup_old_reports()
        download_filename = out_path.name
    except Exception:
        pass

    return {
        "success": True,
        "summary_text": pr.summary_text,
        "operation": operation,
        "row_in": pr.row_in,
        "row_out": pr.row_out,
        "report_filename": download_filename,
    }
```

### 6.2 /api/chart 接口

```python
# ──────────────────────────────────────────────────────────
# API：图表生成（返回 ECharts option JSON）
# ──────────────────────────────────────────────────────────

class ChartRequest(BaseModel):
    file_id: str
    chart_type: str          # bar/pie/line/area/combo
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    y2_col: Optional[str] = None
    y_cols: list[str] = []
    title: str = ""
    color_scheme: str = "default"
    row_limit: int = 100     # 最多取前 N 行，防止图表过密
    use_ai: bool = False
    instruction: str = ""


@router.post("/api/chart")
async def api_chart(req: ChartRequest):
    """生成 ECharts option JSON"""
    if req.file_id not in _uploaded_files:
        raise HTTPException(400, "文件不存在，请重新上传")
    file_path = _uploaded_files[req.file_id]
    try:
        df_raw, all_cols = load_excel(str(file_path))
        numeric_cols = get_numeric_columns(df_raw)
    except Exception as e:
        raise HTTPException(400, f"文件读取失败：{str(e)}")

    chart_type = req.chart_type
    x_col, y_col, y2_col = req.x_col, req.y_col, req.y2_col

    # AI 推断
    if req.use_ai and req.instruction.strip():
        inst = req.instruction.lower()
        _ct_map = [
            (["饼", "pie", "占比", "比例"],      "pie"),
            (["面积", "area"],                   "area"),
            (["折线", "line", "趋势"],           "line"),
            (["组合", "combo", "双轴", "柱线"],  "combo"),
            (["柱", "bar", "直方"],              "bar"),
        ]
        for kws, ct in _ct_map:
            if any(k in inst for k in kws):
                chart_type = ct
                break
        try:
            from core.nlp_parser import IntentParser
            parsed = IntentParser().parse(req.instruction, all_cols)
            if not parsed.get("error"):
                y_col = y_col or parsed.get("target_y")
                xcols = parsed.get("x_columns", [])
                x_col = x_col or (xcols[0] if xcols else None)
                y2_col = y2_col or (xcols[1] if len(xcols) > 1 else None)
        except Exception:
            pass

    # 准备绘图数据（数值列清洗，文字 X 列保留）
    all_y = [c for c in ([y_col, y2_col] + req.y_cols) if c and c in df_raw.columns]
    numeric_y = [c for c in all_y if c in numeric_cols]
    if not numeric_y:
        return JSONResponse({"success": False, "error": "请至少指定一个数值列作为 Y 轴"})
    try:
        clean_df, _, _ = preprocess_for_analysis(df_raw, numeric_y)
        if x_col and x_col not in numeric_cols and x_col in df_raw.columns:
            clean_df[x_col] = df_raw.loc[clean_df.index, x_col].values
    except Exception as e:
        return JSONResponse({"success": False, "error": f"数据处理失败：{str(e)}"})

    clean_df = clean_df.head(req.row_limit)

    try:
        from core.chart_builder import generate_chart, SUPPORTED_CHART_TYPES
        option = generate_chart(
            clean_df, chart_type,
            x_col=x_col if x_col and x_col in clean_df.columns else None,
            y_col=y_col if y_col and y_col in clean_df.columns else None,
            y2_col=y2_col if y2_col and y2_col in clean_df.columns else None,
            y_cols=[c for c in req.y_cols if c in clean_df.columns] or None,
            title=req.title,
        )
    except Exception as e:
        return JSONResponse({"success": False, "error": f"图表生成失败：{str(e)}"})

    from core.chart_builder import SUPPORTED_CHART_TYPES
    return {
        "success": True,
        "option": option,
        "chart_type": chart_type,
        "chart_label": SUPPORTED_CHART_TYPES.get(chart_type, chart_type),
        "data_rows": len(clean_df),
    }
```

**验证（路由存在）：**
```bash
python -c "
import sys; sys.path.insert(0, '.')
from api.routes import router
paths = [r.path for r in router.routes]
assert '/api/process' in paths
assert '/api/chart' in paths
print('routes OK:', [p for p in paths if p.startswith('/api/')])
"
```

**提交：**
```bash
git add api/routes.py
git commit -m "feat: add /api/process and /api/chart endpoints"
```

---

## Task 7：扩展 /api/analyze 支持 compare 和 crosstab 模式

**Files:**
- Modify: `api/routes.py`

### 7.1 在 NLP 关键词覆盖区追加

```python
_compare_kw = ["对比分析", "比较", "组间差异", "compare", "差异显著"]
if mode == "y_vs_all" and any(k in inst_lower for k in _compare_kw):
    mode = "compare"

_crosstab_kw = ["交叉分析", "透视表", "交叉表", "crosstab", "pivot", "分类统计"]
if mode == "y_vs_all" and any(k in inst_lower for k in _crosstab_kw):
    mode = "crosstab"
```

### 7.2 在分析执行 elif 链追加

```python
elif mode == "compare":
    group_c = x_cols[0] if x_cols else None
    if not group_c:
        return JSONResponse({"success": False, "error": "对比分析需要指定分组列（x_columns）"})
    from core.analysis_engine import analyze_compare
    analysis = analyze_compare(clean_df, target_y, group_c)

elif mode == "crosstab":
    col_c = x_cols[0] if x_cols else None
    if not col_c:
        return JSONResponse({"success": False, "error": "交叉分析需要指定交叉列（x_columns）"})
    val_c = x_cols[1] if len(x_cols) > 1 else None
    from core.analysis_engine import analyze_crosstab
    analysis = analyze_crosstab(clean_df, target_y, col_c, val_c)
```

### 7.3 在 _build_table_data 追加

```python
elif analysis.mode == "compare" and analysis.anova_group_stats_df is not None:
    for _, r in analysis.anova_group_stats_df.iterrows():
        rows.append({"组别": str(r["组别"]), "样本量": r["样本量"],
                     "均值": f"{r['均值']:.4f}", "标准差": f"{r['标准差']:.4f}"})
elif analysis.mode == "crosstab" and analysis.stats_df is not None:
    rows = analysis.stats_df.head(10).to_dict("records")
```

### 7.4 在 modeLabel 映射中追加（routes.py 不涉及，在前端做）

**提交：**
```bash
git add api/routes.py
git commit -m "feat: add compare/crosstab modes to /api/analyze routing"
```

---

## Task 8：base.html 引入 ECharts.js CDN

**Files:**
- Modify: `templates/base.html`

在 `marked.js` CDN script 标签后追加：

```html
<!-- ECharts 5.x -->
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
```

**提交：**
```bash
git add templates/base.html
git commit -m "feat: add ECharts 5.x CDN to base template"
```

---

## Task 9：home.html 改造为四列能力面板

**Files:**
- Modify: `templates/home.html`

将「官方能力演示」部分（`.cards-grid`）替换为四列面板，对标截图样式：

```html
<!-- 官方能力演示 -->
<div class="section-title">📋 官方能力演示</div>
<div class="capability-grid">

  <!-- 列1：Excel处理 -->
  <div class="cap-card">
    <div class="cap-card-header">
      <span class="cap-icon">📋</span> Excel处理
    </div>
    <a href="/analyze?instruction=对数据进行清洗，去除重复行并填充缺失值" class="cap-btn">数据清洗</a>
    <a href="/analyze?instruction=数据查找" class="cap-btn">数据查找</a>
    <a href="/analyze?instruction=统计各列数据计数" class="cap-btn">数据计数</a>
    <a href="/analyze?instruction=多表合并" class="cap-btn">多表合并</a>
    <a href="/analyze?instruction=按列拆分为多个Sheet" class="cap-btn">多表/sheet拆分</a>
  </div>

  <!-- 列2：数据运算 -->
  <div class="cap-card">
    <div class="cap-card-header">
      <span class="cap-icon">⚙️</span> 数据运算
    </div>
    <a href="/analyze?instruction=对目标列求和" class="cap-btn">求和</a>
    <a href="/analyze?instruction=计算各列的平均值" class="cap-btn">求平均值</a>
    <a href="/analyze?instruction=找出各列最大值和最小值" class="cap-btn">求极值</a>
    <a href="/analyze?instruction=对数据进行逻辑判断并打标签" class="cap-btn">逻辑计算</a>
    <a href="/analyze?instruction=对多列做汇总统计（求和均值极值）" class="cap-btn">合并计算</a>
  </div>

  <!-- 列3：数据分析 -->
  <div class="cap-card">
    <div class="cap-card-header">
      <span class="cap-icon">📊</span> 数据分析
    </div>
    <a href="/analyze?instruction=对比分析两组数据的差异" class="cap-btn">对比分析</a>
    <a href="/analyze?instruction=生成交叉分析透视表" class="cap-btn">交叉分析</a>
    <a href="/analyze?instruction=分析所有因素之间的关联关系" class="cap-btn">关联分析</a>
    <a href="/analyze?instruction=计算Pearson和Spearman相关系数" class="cap-btn">相关性分析</a>
    <a href="/analyze?instruction=多元线性回归分析" class="cap-btn">线性回归</a>
  </div>

  <!-- 列4：图表生成 -->
  <div class="cap-card">
    <div class="cap-card-header">
      <span class="cap-icon">📈</span> 图表生成
    </div>
    <a href="/analyze?instruction=生成柱状图" class="cap-btn">柱状图</a>
    <a href="/analyze?instruction=生成饼图展示占比" class="cap-btn">饼图</a>
    <a href="/analyze?instruction=生成折线图展示趋势" class="cap-btn">折线图</a>
    <a href="/analyze?instruction=生成面积图" class="cap-btn">面积图</a>
    <a href="/analyze?instruction=生成组合图（柱状+折线双轴）" class="cap-btn">组合图</a>
  </div>

</div>
```

同时在 `static/css/main.css` 追加样式：

```css
/* ── 四列能力面板 ── */
.capability-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}
.cap-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.cap-card-header {
  font-weight: 700;
  font-size: 14px;
  color: var(--text-primary);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--accent);
  display: flex;
  align-items: center;
  gap: 6px;
}
.cap-icon { font-size: 16px; }
.cap-btn {
  display: block;
  padding: 7px 10px;
  margin-bottom: 6px;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 13px;
  color: var(--text-primary);
  text-decoration: none;
  transition: all 0.15s;
}
.cap-btn:hover {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}
@media (max-width: 900px) {
  .capability-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
  .capability-grid { grid-template-columns: 1fr; }
}
```

**验证：**
```bash
python -c "
html = open('templates/home.html', encoding='utf-8').read()
assert 'cap-card' in html
assert 'Excel处理' in html
assert '数据运算' in html
assert '图表生成' in html
print('home.html OK')
"
```

**提交：**
```bash
git add templates/home.html static/css/main.css
git commit -m "feat: redesign home.html with ChatExcel-style 4-column capability panels"
```

---

## Task 10：analyze.html 前端改造（图表面板 + ECharts 渲染 + 处理面板）

**Files:**
- Modify: `templates/analyze.html`

### 10.1 在左侧面板新增"图表生成"和"快捷操作"区域

在 `</div><!-- file-panel -->` 之前插入：

```html
<!-- ── 图表生成面板 ── -->
<div style="border-top:1px solid var(--border);margin:12px 0;"></div>
<div class="ai-toggle">
  <span>📈 图表生成</span>
  <label class="toggle-switch">
    <input type="checkbox" x-model="showChartPanel">
    <span class="toggle-slider"></span>
  </label>
</div>

<template x-if="showChartPanel && fileId">
  <div style="margin-top:8px;">
    <div class="form-group" style="margin-bottom:7px;">
      <label class="form-label">图表类型</label>
      <select x-model="chartType" class="form-input">
        <option value="bar">柱状图</option>
        <option value="pie">饼图</option>
        <option value="line">折线图</option>
        <option value="area">面积图</option>
        <option value="combo">组合图（柱+线）</option>
      </select>
    </div>
    <div class="form-group" style="margin-bottom:7px;">
      <label class="form-label">X 轴列 <span style="color:var(--text-muted)">(可选)</span></label>
      <select x-model="chartX" class="form-input">
        <option value="">— 用序号 —</option>
        <template x-for="col in allColumns" :key="col">
          <option :value="col" x-text="col"></option>
        </template>
      </select>
    </div>
    <div class="form-group" style="margin-bottom:7px;">
      <label class="form-label">Y 轴列（主）</label>
      <select x-model="chartY" class="form-input">
        <option value="">— 请选择 —</option>
        <template x-for="col in columns" :key="col">
          <option :value="col" x-text="col"></option>
        </template>
      </select>
    </div>
    <template x-if="chartType === 'combo'">
      <div class="form-group" style="margin-bottom:7px;">
        <label class="form-label">Y2 轴列（折线）</label>
        <select x-model="chartY2" class="form-input">
          <option value="">— 请选择 —</option>
          <template x-for="col in columns" :key="col">
            <option :value="col" x-text="col"></option>
          </template>
        </select>
      </div>
    </template>
    <div class="form-group" style="margin-bottom:7px;">
      <label class="form-label">标题 <span style="color:var(--text-muted)">(可选)</span></label>
      <input type="text" x-model="chartTitle" class="form-input" placeholder="留空自动生成">
    </div>
    <button class="send-btn" style="width:100%;background:#4472C4;margin-top:4px;"
            :disabled="!chartY || isLoading"
            @click="generateChart()">📈 生成图表</button>
  </div>
</template>
```

### 10.2 在消息气泡中渲染 ECharts

找到「下载按钮」模板之后追加：

```html
<!-- ECharts 图表容器 -->
<template x-if="msg.chartOption">
  <div>
    <div :id="`chart-${idx}`"
         style="width:100%;height:360px;margin-top:12px;border-radius:8px;border:1px solid var(--border);"
         x-init="$nextTick(() => renderChart(`chart-${idx}`, msg.chartOption))">
    </div>
    <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">
      💡 图表支持鼠标缩放、悬停查看数值
    </div>
  </div>
</template>
```

### 10.3 Alpine.js 数据对象追加状态变量

```javascript
// 图表
showChartPanel: false,
chartType: 'bar',
chartX: '',
chartY: '',
chartY2: '',
chartTitle: '',
allColumns: [],
```

### 10.4 uploadFile 中同步 allColumns

在 `this.columns = data.numeric_columns;` 后追加：
```javascript
this.allColumns = data.all_columns || data.numeric_columns;
```

### 10.5 modeLabel 补充新模式

```javascript
compare: '对比分析', crosstab: '交叉分析',
model_comparison: '多模型对比',
```

### 10.6 追加 JS 方法

```javascript
// ECharts 渲染
renderChart(domId, option) {
  const dom = document.getElementById(domId);
  if (!dom || !window.echarts) return;
  const chart = echarts.init(dom, null, { renderer: 'canvas' });
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
},

// 生成图表
async generateChart() {
  if (!this.fileId || !this.chartY || this.isLoading) return;
  const labels = { bar:'柱状图', pie:'饼图', line:'折线图', area:'面积图', combo:'组合图' };
  const label = labels[this.chartType] || this.chartType;
  this.messages.push({ role:'user', content:`📈 生成${label}：${this.chartY}${this.chartX?' × '+this.chartX:''}` });
  this.isLoading = true; this.startThinking(); this.scrollToBottom();
  try {
    const res = await fetch('/api/chart', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_id: this.fileId,
        chart_type: this.chartType,
        x_col: this.chartX || null,
        y_col: this.chartY,
        y2_col: this.chartY2 || null,
        title: this.chartTitle,
      }),
    });
    const data = await res.json();
    if (data.success) {
      this.messages.push({
        role: 'assistant',
        content: `✅ **${data.chart_label}** 已生成（${data.data_rows} 行数据）`,
        chartOption: data.option,
      });
    } else {
      this.messages.push({ role:'assistant', content:'❌ 图表生成失败\n\n'+(data.error||'') });
    }
  } catch(e) {
    this.messages.push({ role:'assistant', content:'❌ 请求失败\n\n'+e.message });
  } finally {
    this.stopThinking(); this.isLoading = false; this.scrollToBottom();
  }
},

// 多模型对比（已有，补充快捷按钮）
runModelComparison() {
  this.instruction = '对比所有模型，找出预测效果最佳的模型';
  const prevMode = this.manualMode;
  this.manualMode = 'model_comparison';
  this.sendMessage().finally(() => { if (!this.useAI) this.manualMode = prevMode; });
},
```

**验证：**
```bash
python -c "
html = open('templates/analyze.html', encoding='utf-8').read()
assert 'renderChart' in html
assert 'generateChart' in html
assert 'showChartPanel' in html
assert 'chartOption' in html
assert '/api/chart' in html
print('analyze.html OK')
"
```

**提交：**
```bash
git add templates/analyze.html
git commit -m "feat: add ECharts chart panel and interactive chart rendering to analyze page"
```

---

## Task 11：NLP 提示词扩展（支持所有新操作的意图识别）

**Files:**
- Modify: `prompts/intent_parser.txt`

在分析模式说明末尾追加：

```
- **compare**: 对比分析两个群体/时间段的差异，使用 t 检验或 ANOVA。target_y 为数值列，x_columns[0] 为分组列。
- **crosstab**: 交叉分析 / 透视表，统计两个分类列的频次或数值聚合。target_y 为行列，x_columns[0] 为列维度，x_columns[1] 可选数值列。
```

在输出格式 `analysis_mode` 枚举中追加 `compare | crosstab`。

**提交：**
```bash
git add prompts/intent_parser.txt
git commit -m "feat: extend NLP prompt with compare/crosstab/chart modes"
```

---

## Task 12：集成测试

**Files:**
- Create: `tests/test_all_new_features.py`

```python
"""全功能集成测试"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

def make_df():
    np.random.seed(42)
    return pd.DataFrame({
        '区域': ['北京','上海','北京','广州','上海','北京'],
        '季度': ['Q1','Q1','Q2','Q2','Q1','Q2'],
        '销售额': [100, 200, 150, 80, 180, 130],
        '利润':   [20,  40,  30,  15, 35,  25],
    })

# ── Excel处理 ──
def test_process_clean():
    from core.excel_processor import process_clean
    df = make_df(); df.loc[0] = df.loc[1]  # 制造重复行
    r = process_clean(df)
    assert r.row_out < r.row_in

def test_process_lookup():
    from core.excel_processor import process_lookup
    r = process_lookup(make_df(), '销售额 > 100')
    assert r.row_out == 4

def test_process_count():
    from core.excel_processor import process_count
    r = process_count(make_df(), group_col='区域')
    assert '北京' in r.result_df['区域'].values

def test_process_split():
    from core.excel_processor import process_split
    r = process_split(make_df(), '区域')
    assert len(r.meta['sheets']) == 3

# ── 数据运算 ──
def test_calc_sum():
    from core.excel_processor import calc_sum
    r = calc_sum(make_df(), '销售额')
    assert r.meta['total'] == 840.0

def test_calc_mean():
    from core.excel_processor import calc_mean
    r = calc_mean(make_df(), '销售额', '区域')
    assert len(r.result_df) == 3

def test_calc_logic():
    from core.excel_processor import calc_logic
    r = calc_logic(make_df(), '销售额 > 100')
    assert '判断结果' in r.result_df.columns
    assert (r.result_df['判断结果'] == '是').sum() == 4

# ── 数据分析 ──
def test_analyze_compare():
    from core.analysis_engine import analyze_compare
    r = analyze_compare(make_df(), '销售额', '区域')
    assert 'p =' in r.summary_text

def test_analyze_crosstab():
    from core.analysis_engine import analyze_crosstab
    r = analyze_crosstab(make_df(), '区域', '季度')
    assert r.mode == 'crosstab'

# ── 图表生成 ──
def test_chart_all_types():
    from core.chart_builder import generate_chart, SUPPORTED_CHART_TYPES
    df = make_df()
    for ct in SUPPORTED_CHART_TYPES:
        kw = {'y_col': '销售额', 'x_col': '区域'}
        if ct == 'combo': kw['y2_col'] = '利润'
        opt = generate_chart(df, ct, **kw)
        assert 'series' in opt, f'{ct} missing series'
    assert len(SUPPORTED_CHART_TYPES) == 5

if __name__ == '__main__':
    test_process_clean();  print('clean OK')
    test_process_lookup(); print('lookup OK')
    test_process_count();  print('count OK')
    test_process_split();  print('split OK')
    test_calc_sum();       print('sum OK')
    test_calc_mean();      print('mean OK')
    test_calc_logic();     print('logic OK')
    test_analyze_compare();  print('compare OK')
    test_analyze_crosstab(); print('crosstab OK')
    test_chart_all_types();  print('charts OK')
    print('\nAll tests passed!')
```

**运行：**
```bash
python tests/test_all_new_features.py
```

预期输出：
```
clean OK
lookup OK
...
All tests passed!
```

**提交：**
```bash
git add tests/test_all_new_features.py
git commit -m "test: full integration test for all new features"
```

---

## Task 13：推送到 GitHub/ModelScope

```bash
git push origin master
```

---

## 验收标准汇总

| 模块 | 功能点 | 验收 |
|------|--------|------|
| Excel处理 | 数据清洗（去重+填充） | ✅ |
| Excel处理 | 数据查找（条件筛选） | ✅ |
| Excel处理 | 数据计数（分组计数） | ✅ |
| Excel处理 | 多表合并（纵/横向） | ✅ |
| Excel处理 | 多表拆分（按列分Sheet） | ✅ |
| 数据运算 | 求和 / 分组求和 | ✅ |
| 数据运算 | 均值 / 分组均值 | ✅ |
| 数据运算 | 极值（min/max） | ✅ |
| 数据运算 | 逻辑计算（打标签） | ✅ |
| 数据运算 | 合并计算（多列聚合） | ✅ |
| 数据分析 | 对比分析（t检验/ANOVA） | ✅ |
| 数据分析 | 交叉分析（透视表） | ✅ |
| 数据分析 | 关联/相关/回归 | ✅（已有） |
| 图表生成 | 柱状图 ECharts | ✅ |
| 图表生成 | 饼图 ECharts | ✅ |
| 图表生成 | 折线图 ECharts | ✅ |
| 图表生成 | 面积图 ECharts | ✅ |
| 图表生成 | 组合图 ECharts | ✅ |
| 前端 | 首页四列能力面板 | ✅ |
| 前端 | 图表面板 + ECharts 渲染 | ✅ |
| 前端 | 图表可交互（缩放/悬停） | ✅ |
| API | /api/process 接口 | ✅ |
| API | /api/chart 接口 | ✅ |
| API | /api/analyze 新增两模式 | ✅ |
