"""
专项结果格式化器
根据用户指令的分析意图（analysis_focus），将同一份 AnalysisResult 格式化为不同侧重的 Markdown 报告。
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from core.analysis_engine import AnalysisResult


# ──────────────────────────────────────────────────────────
# Focus 检测
# ──────────────────────────────────────────────────────────

def detect_focus(instruction: str, mode: str) -> str:
    """从指令文本关键词检测分析焦点"""
    inst = instruction.lower()

    if mode == "y_vs_all":
        if any(k in inst for k in ["最重要", "最大", "最强", "最主要", "第一", "top1", "最影响"]):
            return "top_factor"
        if any(k in inst for k in ["重要性报告", "重要性排名", "排名", "importance", "因子重要"]):
            return "importance_ranking"
        if any(k in inst for k in ["正负", "正相关", "负相关", "拆分", "方向"]):
            return "pos_neg_split"
        if any(k in inst for k in ["全量", "所有相关", "相关性分析", "相关系数", "全部因素", "完整"]):
            return "full_correlation"
        # 默认 y_vs_all 走标准输出
        return "default"

    elif mode == "two_column":
        if any(k in inst for k in ["显著性", "p值", "p-value", "显著"]):
            return "significance_test"
        if any(k in inst for k in ["回归", "线性关系", "方程", "斜率", "截距"]):
            return "regression_eq"
        if any(k in inst for k in ["pearson", "spearman", "相关系数"]):
            return "pearson_spearman"
        if any(k in inst for k in ["分布", "均值", "描述统计", "对比"]):
            return "distribution"
        # 默认 two_column 走相关性检验
        return "correlation_test"

    elif mode == "multi_x_vs_y":
        if any(k in inst for k in ["共线性", "多重共线", "相关矩阵", "vif", "collinear"]):
            return "collinearity"
        if any(k in inst for k in ["rf", "随机森林", "特征重要性", "feature importance"]):
            return "rf_importance"
        if any(k in inst for k in ["系数", "p值", "p-value", "回归系数", "显著性"]):
            return "coef_table"
        if any(k in inst for k in ["多元线性", "回归方程", "r²", "r2", "拟合", "explained"]):
            return "regression_eq"
        # 默认 multi_x_vs_y 走综合输出
        return "default"

    return "default"


# ──────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────

def format_result(analysis: "AnalysisResult", instruction: str) -> str:
    """根据指令意图，将 AnalysisResult 格式化为专项 Markdown"""
    focus = detect_focus(instruction, analysis.mode)

    formatters = {
        "y_vs_all": {
            "top_factor":         _fmt_top_factor,
            "importance_ranking": _fmt_importance_ranking,
            "pos_neg_split":      _fmt_pos_neg_split,
            "full_correlation":   _fmt_full_correlation,
            "default":            lambda a: a.summary_text,
        },
        "two_column": {
            "correlation_test":  _fmt_correlation_test,
            "pearson_spearman":  _fmt_pearson_spearman,
            "regression_eq":     _fmt_two_col_regression,
            "significance_test": _fmt_significance_test,
            "distribution":      _fmt_distribution,
            "default":           _fmt_correlation_test,
        },
        "multi_x_vs_y": {
            "collinearity":   _fmt_collinearity,
            "rf_importance":  _fmt_rf_importance,
            "coef_table":     _fmt_coef_table,
            "regression_eq":  _fmt_multi_regression,
            "default":        lambda a: a.summary_text,
        },
    }

    mode_map = formatters.get(analysis.mode, {})
    fn = mode_map.get(focus, lambda a: a.summary_text)
    return fn(analysis)


# ──────────────────────────────────────────────────────────
# y_vs_all 专项格式
# ──────────────────────────────────────────────────────────

def _fmt_top_factor(a: "AnalysisResult") -> str:
    """找出最重要因素 —— 聚焦 #1 因素的深度解析"""
    if a.feature_importance_df is None or a.feature_importance_df.empty:
        return a.summary_text

    top = a.feature_importance_df.iloc[0]
    corr_df = a.correlation_df
    reg_df = a.regression_df

    direction = "正相关 ↑" if top["Pearson"] >= 0 else "负相关 ↓"
    strength_val = abs(top["Pearson"])
    if strength_val > 0.8:
        strength = "极强"
    elif strength_val > 0.6:
        strength = "强"
    elif strength_val > 0.3:
        strength = "中等"
    else:
        strength = "弱"

    lines = [
        f"## 🏆 最重要影响因素分析",
        f"**目标变量**：`{a.target_y}`  **有效样本**：{a.valid_row_count} 行（共 {len(a.x_columns)} 个候选因素）",
        f"",
        f"### 🥇 第一名：`{top['Feature']}`",
        f"| 指标 | 值 | 说明 |",
        f"|------|-----|------|",
        f"| 随机森林权重 | **{top['Importance']:.1%}** | 综合影响力排名第一 |",
        f"| Pearson 相关系数 | **{top['Pearson']:+.4f}** | {direction}，{strength}相关 |",
    ]

    # 如果有单变量回归结果
    if reg_df is not None and not reg_df.empty:
        row = reg_df[reg_df["Feature"] == top["Feature"]]
        if not row.empty:
            r = row.iloc[0]
            lines += [
                f"| 线性回归 R² | **{r['R_Squared']:.4f}** | 可解释 {r['R_Squared']:.1%} 的方差 |",
                f"| 回归方程 | `{a.target_y} = {r['Coefficient']:.4f}·{top['Feature']} + {r['Intercept']:.4f}` | — |",
            ]

    lines += [
        f"",
        f"### 📊 影响力排名（Top 10）",
        f"| 排名 | 因素 | 权重 | Pearson r | 方向 | 强度 |",
        f"|------|------|------|-----------|------|------|",
    ]

    medals = ["🥇", "🥈", "🥉"] + ["  "] * 20
    for i, (_, row) in enumerate(a.feature_importance_df.head(10).iterrows()):
        d = "↑ 正" if row["Pearson"] >= 0 else "↓ 负"
        v = abs(row["Pearson"])
        s = "极强" if v > 0.8 else ("强" if v > 0.6 else ("中" if v > 0.3 else "弱"))
        lines.append(f"| {medals[i]} {i+1} | `{row['Feature']}` | {row['Importance']:.1%} | {row['Pearson']:+.3f} | {d} | {s} |")

    return "\n".join(lines)


def _fmt_importance_ranking(a: "AnalysisResult") -> str:
    """因子重要性报告 —— 全量因素的重要性排名"""
    if a.feature_importance_df is None:
        return a.summary_text

    lines = [
        f"## 📊 因子重要性完整报告",
        f"**目标变量**：`{a.target_y}`  **有效样本**：{a.valid_row_count} 行",
        f"**共分析 {len(a.x_columns)} 个影响因素**",
        f"",
        f"### 🏆 完整因子重要性排名",
        f"| 排名 | 因子 | RF 权重 | Pearson r | 方向 | 相关强度 |",
        f"|------|------|---------|-----------|------|---------|",
    ]

    for i, (_, row) in enumerate(a.feature_importance_df.iterrows()):
        d = "↑ 正相关" if row["Pearson"] >= 0 else "↓ 负相关"
        v = abs(row["Pearson"])
        s = "极强" if v > 0.8 else ("强" if v > 0.6 else ("中等" if v > 0.3 else "弱"))
        lines.append(f"| {i+1} | `{row['Feature']}` | {row['Importance']:.1%} | {row['Pearson']:+.3f} | {d} | {s} |")

    # 汇总信息
    df = a.feature_importance_df
    strong_pos = df[(df["Pearson"] > 0.6)]["Feature"].tolist()
    strong_neg = df[(df["Pearson"] < -0.6)]["Feature"].tolist()
    top3_imp = df.head(3)["Feature"].tolist()

    lines += [
        f"",
        f"### 📌 关键洞察",
        f"- **最重要因素（Top 3）**：{', '.join(f'`{c}`' for c in top3_imp)}",
        f"- **强正相关因素**：{', '.join(f'`{c}`' for c in strong_pos) if strong_pos else '无'}",
        f"- **强负相关因素**：{', '.join(f'`{c}`' for c in strong_neg) if strong_neg else '无'}",
        f"- 前 3 因素合计权重：{df.head(3)['Importance'].sum():.1%}",
    ]

    return "\n".join(lines)


def _fmt_pos_neg_split(a: "AnalysisResult") -> str:
    """正负相关因素拆分 —— 分组展示正/负相关因素"""
    if a.correlation_df is None:
        return a.summary_text

    corr = a.correlation_df.copy()
    pos_df = corr[corr["Pearson"] > 0].sort_values("Pearson", ascending=False)
    neg_df = corr[corr["Pearson"] < 0].sort_values("Pearson")
    neu_df = corr[corr["Pearson"] == 0]

    lines = [
        f"## ↕️ 正负相关因素拆分",
        f"**目标变量**：`{a.target_y}`  **有效样本**：{a.valid_row_count} 行",
        f"**正相关因素 {len(pos_df)} 个 ｜ 负相关因素 {len(neg_df)} 个**",
        f"",
        f"### ↑ 正相关因素（共 {len(pos_df)} 个）",
        f"> 这些因素增大时，`{a.target_y}` 也倾向于增大",
        f"",
        f"| 因素 | Pearson r | 强度 |",
        f"|------|-----------|------|",
    ]

    for _, row in pos_df.iterrows():
        v = abs(row["Pearson"])
        s = "极强" if v > 0.8 else ("强" if v > 0.6 else ("中等" if v > 0.3 else "弱"))
        lines.append(f"| `{row['Feature']}` | +{row['Pearson']:.4f} | {s} |")

    lines += [
        f"",
        f"### ↓ 负相关因素（共 {len(neg_df)} 个）",
        f"> 这些因素增大时，`{a.target_y}` 倾向于减小",
        f"",
        f"| 因素 | Pearson r | 强度 |",
        f"|------|-----------|------|",
    ]

    for _, row in neg_df.iterrows():
        v = abs(row["Pearson"])
        s = "极强" if v > 0.8 else ("强" if v > 0.6 else ("中等" if v > 0.3 else "弱"))
        lines.append(f"| `{row['Feature']}` | {row['Pearson']:.4f} | {s} |")

    if not neu_df.empty:
        neu_names = ", ".join(f"`{r['Feature']}`" for _, r in neu_df.iterrows())
        lines += [
            f"",
            f"### ➡️ 零相关因素（共 {len(neu_df)} 个）",
            neu_names,
        ]

    return "\n".join(lines)


def _fmt_full_correlation(a: "AnalysisResult") -> str:
    """全量相关性分析 —— 完整的 Pearson + Spearman 相关性表格"""
    if a.correlation_df is None:
        return a.summary_text

    lines = [
        f"## 🔬 全量相关性分析",
        f"**目标变量**：`{a.target_y}`  **有效样本**：{a.valid_row_count} 行",
        f"",
        f"| 因素 | Pearson r | Pearson p | Spearman ρ | Spearman p | 强度 | 方向 |",
        f"|------|-----------|-----------|------------|------------|------|------|",
    ]

    # correlation_df 可能包含 Pearson_p 和 Spearman_p
    for _, row in a.correlation_df.sort_values("AbsPearson", ascending=False).iterrows():
        v = abs(row["Pearson"])
        s = "极强" if v > 0.8 else ("强" if v > 0.6 else ("中" if v > 0.3 else ("弱" if v > 0.1 else "无")))
        d = "↑" if row["Pearson"] > 0 else "↓"
        pval = row.get("Pearson_p", float("nan"))
        spval = row.get("Spearman_p", float("nan"))
        spear = row.get("Spearman", float("nan"))
        p_str = "p<0.001" if pval < 0.001 else f"p={pval:.3f}" if pval == pval else "—"
        sp_str = "p<0.001" if spval < 0.001 else f"p={spval:.3f}" if spval == spval else "—"
        spear_str = f"{spear:+.3f}" if spear == spear else "—"
        lines.append(
            f"| `{row['Feature']}` | {row['Pearson']:+.4f} | {p_str} | {spear_str} | {sp_str} | {s} | {d} |"
        )

    # 显著性统计
    sig_count = sum(
        1 for _, row in a.correlation_df.iterrows()
        if row.get("Pearson_p", 1.0) < 0.05
    )
    lines += [
        f"",
        f"### 📌 统计摘要",
        f"- 共 **{len(a.correlation_df)}** 个因素，其中 **{sig_count}** 个与 `{a.target_y}` 显著相关（p<0.05）",
        f"- 按相关强度排名第一：`{a.correlation_df.iloc[0]['Feature']}` (|r| = {a.correlation_df.iloc[0]['AbsPearson']:.4f})",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# two_column 专项格式
# ──────────────────────────────────────────────────────────

def _fmt_correlation_test(a: "AnalysisResult") -> str:
    """双列相关性检验 —— 综合相关系数检验结果"""
    abs_r = abs(a.pearson_r)
    if abs_r > 0.8:
        strength = "极强"
    elif abs_r > 0.6:
        strength = "强"
    elif abs_r > 0.4:
        strength = "中等"
    elif abs_r > 0.2:
        strength = "弱"
    else:
        strength = "几乎无"

    direction = "正" if a.pearson_r > 0 else "负"
    sig_str = "**显著相关**（p<0.05）" if a.pearson_p < 0.05 else "**不显著**（p≥0.05）"
    p_str = "p<0.001" if a.pearson_p < 0.001 else f"p={a.pearson_p:.4f}"
    sp_str = "p<0.001" if a.spearman_p < 0.001 else f"p={a.spearman_p:.4f}"

    lines = [
        f"## 🔗 双列相关性检验",
        f"**分析列对**：`{a.col_a}` ↔ `{a.col_b}`  **有效样本**：{a.valid_row_count} 行",
        f"",
        f"### 📊 相关系数汇总",
        f"| 检验方法 | 系数值 | p 值 | 结论 |",
        f"|---------|--------|------|------|",
        f"| **Pearson r** | **{a.pearson_r:+.4f}** | {p_str} | {'显著' if a.pearson_p < 0.05 else '不显著'} |",
        f"| **Spearman ρ** | **{a.spearman_r:+.4f}** | {sp_str} | {'显著' if a.spearman_p < 0.05 else '不显著'} |",
        f"",
        f"### 📝 综合结论",
        f"- `{a.col_a}` 与 `{a.col_b}` 存在 **{strength}{direction}相关**，统计上{sig_str}",
        f"- Pearson r = {a.pearson_r:+.4f}，Spearman ρ = {a.spearman_r:+.4f}",
    ]

    if abs(a.pearson_r - a.spearman_r) > 0.1:
        lines.append(f"- ⚠️ Pearson 与 Spearman 差异较大（Δ={abs(a.pearson_r - a.spearman_r):.3f}），数据可能存在非线性关系或异常值")

    return "\n".join(lines)


def _fmt_pearson_spearman(a: "AnalysisResult") -> str:
    """Pearson + Spearman 详细解读"""
    p_str = "p<0.001" if a.pearson_p < 0.001 else f"p={a.pearson_p:.4f}"
    sp_str = "p<0.001" if a.spearman_p < 0.001 else f"p={a.spearman_p:.4f}"

    def _interp(r):
        a = abs(r)
        base = ("极强" if a > 0.8 else ("强" if a > 0.6 else ("中等" if a > 0.4 else ("弱" if a > 0.2 else "几乎无"))))
        d = "正" if r > 0 else ("负" if r < 0 else "零")
        return f"{base}{d}相关"

    lines = [
        f"## 📐 Pearson + Spearman 相关分析",
        f"**分析列对**：`{a.col_a}` ↔ `{a.col_b}`  **有效样本**：{a.valid_row_count} 行",
        f"",
        f"### Pearson 线性相关系数",
        f"- **r = {a.pearson_r:+.4f}**，{p_str}",
        f"- 解读：{_interp(a.pearson_r)}（适用于正态分布、线性关系）",
        f"",
        f"### Spearman 秩相关系数",
        f"- **ρ = {a.spearman_r:+.4f}**，{sp_str}",
        f"- 解读：{_interp(a.spearman_r)}（适用于非正态、单调关系）",
        f"",
        f"### 🔍 两种方法对比",
        f"| 方法 | 系数 | 适用场景 | 结论 |",
        f"|------|------|---------|------|",
        f"| Pearson | {a.pearson_r:+.4f} | 线性关系 | {'显著' if a.pearson_p < 0.05 else '不显著'} ({p_str}) |",
        f"| Spearman | {a.spearman_r:+.4f} | 单调关系 | {'显著' if a.spearman_p < 0.05 else '不显著'} ({sp_str}) |",
    ]

    diff = abs(a.pearson_r - a.spearman_r)
    if diff > 0.1:
        lines += [
            f"",
            f"⚠️ **注意**：两系数相差 {diff:.3f}，说明 `{a.col_a}` 与 `{a.col_b}` 之间可能存在**非线性关系**，建议结合散点图判断。",
        ]
    else:
        lines += [
            f"",
            f"✅ 两系数高度一致（差值 {diff:.3f}），关系较为稳定。",
        ]

    return "\n".join(lines)


def _fmt_two_col_regression(a: "AnalysisResult") -> str:
    """双向线性回归 —— 展示 A→B 和 B→A 两个回归方程"""
    ab = a.reg_a_to_b or {}
    ba = a.reg_b_to_a or {}

    lines = [
        f"## 📏 双向线性回归分析",
        f"**分析列对**：`{a.col_a}` ↔ `{a.col_b}`  **有效样本**：{a.valid_row_count} 行",
        f"",
        f"### ➡️ 方向 A：以 `{a.col_a}` 预测 `{a.col_b}`",
        f"| 参数 | 值 |",
        f"|------|----|",
        f"| **回归方程** | `{a.col_b} = {ab.get('slope', 0):.4f} × {a.col_a} + {ab.get('intercept', 0):.4f}` |",
        f"| **R²（决定系数）** | {ab.get('r2', 0):.4f}（模型解释 {ab.get('r2', 0):.1%} 的方差） |",
        f"| **斜率** | {ab.get('slope', 0):.4f}（`{a.col_a}` 每增加 1，`{a.col_b}` 变化 {ab.get('slope', 0):.4f}） |",
        f"| **截距** | {ab.get('intercept', 0):.4f} |",
        f"",
        f"### ⬅️ 方向 B：以 `{a.col_b}` 预测 `{a.col_a}`",
        f"| 参数 | 值 |",
        f"|------|----|",
        f"| **回归方程** | `{a.col_a} = {ba.get('slope', 0):.4f} × {a.col_b} + {ba.get('intercept', 0):.4f}` |",
        f"| **R²（决定系数）** | {ba.get('r2', 0):.4f}（模型解释 {ba.get('r2', 0):.1%} 的方差） |",
        f"| **斜率** | {ba.get('slope', 0):.4f} |",
        f"",
        f"### 📌 回归分析总结",
        f"- **Pearson 相关系数**：r = {a.pearson_r:+.4f}（{' 显著' if a.pearson_p < 0.05 else '不显著'}）",
    ]

    better = "A→B" if ab.get("r2", 0) >= ba.get("r2", 0) else "B→A"
    lines.append(f"- **预测方向建议**：{better} 方向的 R² 更高，预测效果更好")

    return "\n".join(lines)


def _fmt_significance_test(a: "AnalysisResult") -> str:
    """显著性检验 —— 聚焦统计显著性"""
    p_str = "p<0.001" if a.pearson_p < 0.001 else f"p={a.pearson_p:.4f}"
    sp_str = "p<0.001" if a.spearman_p < 0.001 else f"p={a.spearman_p:.4f}"

    is_sig_p = a.pearson_p < 0.05
    is_sig_s = a.spearman_p < 0.05

    lines = [
        f"## 🔬 统计显著性检验",
        f"**分析列对**：`{a.col_a}` ↔ `{a.col_b}`  **有效样本**：{a.valid_row_count} 行",
        f"",
        f"### 假设检验",
        f"- **H₀（零假设）**：`{a.col_a}` 与 `{a.col_b}` 之间不存在线性相关（ρ = 0）",
        f"- **H₁（备择假设）**：两者存在显著线性相关（ρ ≠ 0）",
        f"",
        f"### 检验结果",
        f"| 方法 | 统计量 | p 值 | 结论（α=0.05） |",
        f"|------|--------|------|----------------|",
        f"| Pearson | r = {a.pearson_r:+.4f} | **{p_str}** | {'✅ **拒绝 H₀**，显著相关' if is_sig_p else '❌ 无法拒绝 H₀，不显著'} |",
        f"| Spearman | ρ = {a.spearman_r:+.4f} | **{sp_str}** | {'✅ **拒绝 H₀**，显著相关' if is_sig_s else '❌ 无法拒绝 H₀，不显著'} |",
        f"",
        f"### 📌 结论",
    ]

    if is_sig_p and is_sig_s:
        lines.append(f"✅ **Pearson 和 Spearman 双重检验均显著**，`{a.col_a}` 与 `{a.col_b}` 存在统计上显著的相关关系（{abs(a.pearson_r):.1%} 强度）。")
    elif is_sig_p:
        lines.append(f"⚠️ **Pearson 显著但 Spearman 不显著**，两变量可能存在线性关联但不具有单调稳健性，建议检查异常值。")
    elif is_sig_s:
        lines.append(f"⚠️ **Spearman 显著但 Pearson 不显著**，两变量可能存在非线性单调关系。")
    else:
        lines.append(f"❌ **双重检验均不显著**，在当前样本量（n={a.valid_row_count}）下，无法证明 `{a.col_a}` 与 `{a.col_b}` 存在统计显著的相关关系。")

    lines += [
        f"",
        f"> **注意**：显著性受样本量影响，大样本下极弱的相关也可能显著，需结合效应量（|r| 值）综合判断。",
    ]

    return "\n".join(lines)


def _fmt_distribution(a: "AnalysisResult") -> str:
    """分布对比统计 —— 描述性统计对比"""
    if a.stats_df is None:
        return a.summary_text

    lines = [
        f"## 📊 分布对比统计",
        f"**分析列对**：`{a.col_a}` ↔ `{a.col_b}`  **有效样本**：{a.valid_row_count} 行",
        f"",
        f"### 描述性统计对比",
        f"| 统计量 | `{a.col_a}` | `{a.col_b}` |",
        f"|--------|------------|------------|",
    ]

    stat_labels = {
        "样本数": "样本数", "均值": "均值", "标准差": "标准差",
        "最小值": "最小值", "中位数": "中位数", "最大值": "最大值",
        "偏度": "偏度", "峰度": "峰度",
    }

    for col_label in stat_labels:
        try:
            va = a.stats_df.loc[a.col_a, col_label]
            vb = a.stats_df.loc[a.col_b, col_label]
            va_str = f"{va:.4f}" if isinstance(va, float) else str(va)
            vb_str = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
            lines.append(f"| {col_label} | {va_str} | {vb_str} |")
        except (KeyError, TypeError):
            continue

    lines += [
        f"",
        f"### 🔗 相关性摘要",
        f"- Pearson r = {a.pearson_r:+.4f}（{'显著' if a.pearson_p < 0.05 else '不显著'}）",
        f"- Spearman ρ = {a.spearman_r:+.4f}（{'显著' if a.spearman_p < 0.05 else '不显著'}）",
    ]

    # 偏度解读
    try:
        skew_a = a.stats_df.loc[a.col_a, "偏度"]
        skew_b = a.stats_df.loc[a.col_b, "偏度"]
        if abs(skew_a) > 1 or abs(skew_b) > 1:
            lines.append(f"- ⚠️ 数据存在明显偏态（偏度 `{a.col_a}`={skew_a:.2f}, `{a.col_b}`={skew_b:.2f}），建议考虑 Spearman 系数更为可靠")
    except (KeyError, TypeError):
        pass

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# multi_x_vs_y 专项格式
# ──────────────────────────────────────────────────────────

def _fmt_collinearity(a: "AnalysisResult") -> str:
    """多重共线性检测 —— 聚焦 X 间相关矩阵与警告"""
    lines = [
        f"## ⚠️ 多重共线性检测",
        f"**目标变量**：`{a.target_y}`  **自变量个数**：{len(a.x_columns)} 个",
        f"**有效样本**：{a.valid_row_count} 行",
        f"",
    ]

    if a.collinearity_warnings:
        lines += [
            f"### 🚨 检测到以下共线性问题（|r| ≥ 0.8）",
            f"",
        ]
        for w in a.collinearity_warnings:
            lines.append(f"- {w}")
        lines += [
            f"",
            f"> **建议**：共线性过强的变量会导致回归系数不稳定，建议：",
            f"> 1. 删除其中一个高度相关的变量",
            f"> 2. 使用 PCA 降维后再回归",
            f"> 3. 使用岭回归（Ridge）代替普通最小二乘",
        ]
    else:
        lines += [
            f"### ✅ 未检测到严重共线性问题",
            f"所有自变量对之间的相关系数均低于 0.8，多重共线性风险较低。",
        ]

    if a.x_corr_matrix is not None:
        lines += [
            f"",
            f"### 📋 自变量相关矩阵",
            f"",
        ]
        # 表头
        cols = a.x_columns
        header = "| 变量 | " + " | ".join(f"`{c}`" for c in cols) + " |"
        sep = "|------|" + "------|" * len(cols)
        lines += [header, sep]
        for i, c in enumerate(cols):
            row_vals = []
            for j, c2 in enumerate(cols):
                v = a.x_corr_matrix.iloc[i, j]
                if i == j:
                    row_vals.append("—")
                else:
                    flag = " ⚠️" if abs(v) >= 0.8 else ""
                    row_vals.append(f"{v:+.3f}{flag}")
            lines.append(f"| `{c}` | " + " | ".join(row_vals) + " |")

    # 多元回归整体质量
    lines += [
        f"",
        f"### 📐 多元回归整体拟合",
        f"- **R²** = {a.multi_reg_r2:.4f}（模型解释 {a.multi_reg_r2:.1%} 的 `{a.target_y}` 方差）",
    ]

    return "\n".join(lines)


def _fmt_rf_importance(a: "AnalysisResult") -> str:
    """RF 特征重要性 —— 专注随机森林结果"""
    if a.feature_importance_df is None:
        return a.summary_text

    lines = [
        f"## 🌲 随机森林特征重要性",
        f"**目标变量**：`{a.target_y}`  **自变量**：{', '.join(f'`{c}`' for c in a.x_columns)}",
        f"**有效样本**：{a.valid_row_count} 行",
        f"",
        f"### 📊 特征重要性排名",
        f"| 排名 | 特征 | RF 重要性 | Pearson r | 方向 |",
        f"|------|------|-----------|-----------|------|",
    ]

    medals = ["🥇", "🥈", "🥉"] + ["  "] * 20
    for i, (_, row) in enumerate(a.feature_importance_df.iterrows()):
        d = "↑ 正" if row["Pearson"] >= 0 else "↓ 负"
        lines.append(f"| {medals[i]} {i+1} | `{row['Feature']}` | **{row['Importance']:.1%}** | {row['Pearson']:+.3f} | {d} |")

    total = a.feature_importance_df["Importance"].sum()
    top3_sum = a.feature_importance_df.head(3)["Importance"].sum()
    top1 = a.feature_importance_df.iloc[0]

    lines += [
        f"",
        f"### 📌 重要性摘要",
        f"- **最重要特征**：`{top1['Feature']}`（重要性 {top1['Importance']:.1%}）",
        f"- **Top 3 合计**：{top3_sum:.1%}（占总重要性的 {top3_sum/total:.0%}）",
        f"- **总和**：{total:.1%}（随机森林权重之和，约等于 100%）",
        f"",
        f"> RF 重要性反映了每个特征对降低预测误差的贡献度，值越高表示该特征在预测 `{a.target_y}` 中越关键。",
    ]

    return "\n".join(lines)


def _fmt_coef_table(a: "AnalysisResult") -> str:
    """回归系数 + P 值 —— 聚焦系数显著性"""
    if a.multi_reg_coef_df is None:
        return a.summary_text

    lines = [
        f"## 📋 回归系数与显著性检验",
        f"**目标变量**：`{a.target_y}`  **自变量**：{', '.join(f'`{c}`' for c in a.x_columns)}",
        f"**有效样本**：{a.valid_row_count} 行",
        f"",
        f"### 多元线性回归系数表",
        f"| 特征 | 系数 | p 值 | 显著性 | 解读 |",
        f"|------|------|------|--------|------|",
    ]

    for _, row in a.multi_reg_coef_df.iterrows():
        p = row["p_value"]
        p_str = "p<0.001" if p < 0.001 else f"p={p:.4f}" if p == p else "N/A"
        if p < 0.001:
            sig = "★★★ 极显著"
        elif p < 0.01:
            sig = "★★ 高度显著"
        elif p < 0.05:
            sig = "★ 显著"
        elif p < 0.1:
            sig = "· 边缘显著"
        else:
            sig = "— 不显著"

        direction = "↑ 正向" if row["Coefficient"] > 0 else "↓ 负向"
        lines.append(f"| `{row['Feature']}` | **{row['Coefficient']:.4f}** | {p_str} | {sig} | {direction} |")

    n = a.valid_row_count
    k = len(a.x_columns)
    adj_r2 = 1 - (1 - a.multi_reg_r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    adj_str = f"{adj_r2:.4f}" if adj_r2 == adj_r2 else "N/A"

    sig_features = [
        row["Feature"] for _, row in a.multi_reg_coef_df.iterrows()
        if row["p_value"] < 0.05
    ]

    lines += [
        f"",
        f"### 📐 模型整体质量",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| R² | {a.multi_reg_r2:.4f} |",
        f"| 调整后 R² | {adj_str} |",
        f"| 显著特征（p<0.05） | {', '.join(f'`{c}`' for c in sig_features) if sig_features else '无'} |",
        f"",
        f"> **显著性标准**：★★★ p<0.001 ｜ ★★ p<0.01 ｜ ★ p<0.05 ｜ · p<0.1 ｜ — p≥0.1",
    ]

    if a.collinearity_warnings:
        lines += ["", "### ⚠️ 共线性警告"] + [f"- {w}" for w in a.collinearity_warnings]

    return "\n".join(lines)


def _fmt_multi_regression(a: "AnalysisResult") -> str:
    """多元线性回归方程 —— 展示完整回归方程与 R²"""
    if a.multi_reg_coef_df is None:
        return a.summary_text

    # 构建回归方程字符串
    terms = []
    for _, row in a.multi_reg_coef_df.iterrows():
        sign = "+" if row["Coefficient"] >= 0 else ""
        terms.append(f"{sign}{row['Coefficient']:.4f}·{row['Feature']}")
    equation = f"{a.target_y} = " + " ".join(terms)

    n = a.valid_row_count
    k = len(a.x_columns)
    adj_r2 = 1 - (1 - a.multi_reg_r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")

    lines = [
        f"## 📐 多元线性回归分析",
        f"**目标变量**：`{a.target_y}`  **自变量**：{', '.join(f'`{c}`' for c in a.x_columns)}",
        f"**有效样本**：{a.valid_row_count} 行",
        f"",
        f"### 📌 回归方程",
        f"```",
        equation,
        f"```",
        f"",
        f"### 模型拟合质量",
        f"| 指标 | 值 | 说明 |",
        f"|------|----|------|",
        f"| **R²** | {a.multi_reg_r2:.4f} | 模型可解释 {a.multi_reg_r2:.1%} 的 `{a.target_y}` 方差 |",
        f"| **调整后 R²** | {adj_r2:.4f} | 修正自变量个数后的 R² |",
        f"| 样本量 | {n} | |",
        f"| 自变量个数 | {k} | |",
        f"",
        f"### 各项系数",
        f"| 特征 | 系数 | 方向解读 |",
        f"|------|------|---------|",
    ]

    for _, row in a.multi_reg_coef_df.iterrows():
        d = f"每增加 1，`{a.target_y}` {'增加' if row['Coefficient'] > 0 else '减少'} {abs(row['Coefficient']):.4f}"
        lines.append(f"| `{row['Feature']}` | {row['Coefficient']:+.4f} | {d} |")

    if a.collinearity_warnings:
        lines += ["", "### ⚠️ 共线性警告"] + [f"- {w}" for w in a.collinearity_warnings]

    return "\n".join(lines)
