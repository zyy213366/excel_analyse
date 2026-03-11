"""
三模式分析引擎
- y_vs_all       : 分析所有因素对目标 Y 的影响（随机森林 + Pearson + 线性回归）
- two_column     : 分析两列之间的关系（双向回归 + Pearson + Spearman + 分布）
- multi_x_vs_y   : 多 X 对 Y 的影响（多元回归 + RF 特征重要性 + 共线性检测）
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import RF_N_ESTIMATORS, RF_RANDOM_STATE, COLLINEARITY_THRESHOLD


# ──────────────────────────────────────────────────────────
# 统一结果容器
# ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    mode: str                          # y_vs_all / two_column / multi_x_vs_y
    target_y: str
    x_columns: list[str]
    valid_row_count: int
    raw_row_count: int

    # 通用统计
    summary_text: str = ""             # 最终展示给用户的 Markdown 结论

    # mode: y_vs_all / multi_x_vs_y 共用
    feature_importance_df: Optional[pd.DataFrame] = None   # Feature, Importance, Pearson, Imp_Pos, Imp_Neg
    correlation_df: Optional[pd.DataFrame] = None          # Feature, Pearson, Spearman, AbsPearson
    regression_df: Optional[pd.DataFrame] = None           # Feature, R_Squared, Intercept, Coefficient
    plot_df: Optional[pd.DataFrame] = None                 # 散点图数据源
    stats_df: Optional[pd.DataFrame] = None                # 描述性统计

    # mode: two_column 专用
    col_a: str = ""
    col_b: str = ""
    pearson_r: float = 0.0
    pearson_p: float = 0.0
    spearman_r: float = 0.0
    spearman_p: float = 0.0
    reg_a_to_b: Optional[dict] = None  # {r2, slope, intercept}
    reg_b_to_a: Optional[dict] = None

    # mode: multi_x_vs_y 专用
    multi_reg_r2: float = 0.0
    multi_reg_coef_df: Optional[pd.DataFrame] = None       # Feature, Coefficient, p_value
    collinearity_warnings: list[str] = field(default_factory=list)
    x_corr_matrix: Optional[pd.DataFrame] = None          # X 间相关矩阵


# ──────────────────────────────────────────────────────────
# 模式1：Y vs All
# ──────────────────────────────────────────────────────────

def analyze_y_vs_all(df: pd.DataFrame, target_y: str) -> AnalysisResult:
    """
    分析所有 X 对 target_y 的影响。
    核心算法：随机森林特征重要性 + Pearson/Spearman + 单变量线性回归
    """
    X = df.drop(columns=[target_y])
    y = df[target_y]

    if X.shape[1] == 0:
        return AnalysisResult(
            mode="y_vs_all", target_y=target_y, x_columns=[],
            valid_row_count=len(df), raw_row_count=len(df),
            summary_text="⚠ 数据集仅含目标变量，无法进行相关性分析。"
        )

    # 相关性计算
    corr_rows = []
    for col in X.columns:
        p_r, p_p = stats.pearsonr(X[col], y)
        s_r, s_p = stats.spearmanr(X[col], y)
        p_r = 0.0 if np.isnan(p_r) else p_r
        corr_rows.append({"Feature": col, "Pearson": p_r, "Pearson_p": p_p,
                          "Spearman": s_r, "Spearman_p": s_p, "AbsPearson": abs(p_r)})
    corr_df = pd.DataFrame(corr_rows).sort_values("AbsPearson", ascending=False)

    # 随机森林
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=1)
    rf.fit(X, y)
    importances = rf.feature_importances_

    # 合并
    imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    merged = imp_df.merge(corr_df[["Feature", "Pearson"]], on="Feature").sort_values("Importance", ascending=False)
    merged["Imp_Pos"] = merged.apply(lambda r: r["Importance"] if r["Pearson"] >= 0 else 0, axis=1)
    merged["Imp_Neg"] = merged.apply(lambda r: r["Importance"] if r["Pearson"] < 0 else 0, axis=1)

    # 单变量线性回归（Top 3）
    top3 = corr_df["Feature"].head(3).tolist()
    reg_rows = []
    for feat in top3:
        lr = LinearRegression()
        lr.fit(X[[feat]], y)
        reg_rows.append({
            "Feature": feat,
            "R_Squared": round(lr.score(X[[feat]], y), 4),
            "Intercept": round(lr.intercept_, 4),
            "Coefficient": round(lr.coef_[0], 4),
        })
    reg_df = pd.DataFrame(reg_rows)

    # 描述性统计
    stats_tbl = df.describe().T
    stats_tbl["skew"] = df.skew()
    stats_tbl["kurt"] = df.kurt()
    stats_tbl = stats_tbl[["count", "mean", "std", "min", "50%", "max", "skew", "kurt"]]
    stats_tbl.columns = ["样本数", "均值", "标准差", "最小值", "中位数", "最大值", "偏度", "峰度"]

    # 散点图数据源（Target + Top 6 特征）
    top6 = merged["Feature"].head(6).tolist()
    plot_data = pd.DataFrame({target_y: y})
    for f in top6:
        plot_data[f] = X[f].values
    plot_data = plot_data.reset_index(drop=True)

    # 生成总结文本
    top_feat = corr_df.iloc[0]
    pos_feat = corr_df[corr_df["Pearson"] > 0]["Feature"].iloc[0] if not corr_df[corr_df["Pearson"] > 0].empty else "无"
    neg_feat = corr_df[corr_df["Pearson"] < 0]["Feature"].iloc[0] if not corr_df[corr_df["Pearson"] < 0].empty else "无"

    lines = [
        f"## 📊 分析模式：全因素影响分析",
        f"**目标变量**：`{target_y}`",
        f"**有效样本**：{len(df)} 行（共分析 {X.shape[1]} 个影响因素）",
        f"",
        f"### 🏆 核心发现",
        f"- **最强相关因素**：`{top_feat['Feature']}` (Pearson r = {top_feat['Pearson']:.3f})",
        f"- **最强正相关**：`{pos_feat}`",
        f"- **最强负相关**：`{neg_feat}`",
        f"",
        f"### 📈 Top 3 影响因素",
    ]
    for _, row in merged.head(3).iterrows():
        direction = "正相关↑" if row["Pearson"] >= 0 else "负相关↓"
        strength = "强" if abs(row["Pearson"]) > 0.6 else ("中" if abs(row["Pearson"]) > 0.3 else "弱")
        lines.append(f"- `{row['Feature']}` — 权重 {row['Importance']:.1%}，{direction}（{strength}）")

    return AnalysisResult(
        mode="y_vs_all",
        target_y=target_y,
        x_columns=list(X.columns),
        valid_row_count=len(df),
        raw_row_count=len(df),
        summary_text="\n".join(lines),
        feature_importance_df=merged,
        correlation_df=corr_df,
        regression_df=reg_df,
        plot_df=plot_data,
        stats_df=stats_tbl,
    )


# ──────────────────────────────────────────────────────────
# 模式2：Two Column
# ──────────────────────────────────────────────────────────

def analyze_two_column(df: pd.DataFrame, col_a: str, col_b: str) -> AnalysisResult:
    """
    分析两列之间的关系。
    核心：Pearson + Spearman + 双向线性回归 + 描述性统计
    """
    a = df[col_a]
    b = df[col_b]

    # 相关性
    p_r, p_p = stats.pearsonr(a, b)
    s_r, s_p = stats.spearmanr(a, b)

    # A → B 回归
    lr_ab = LinearRegression().fit(a.values.reshape(-1, 1), b)
    r2_ab = lr_ab.score(a.values.reshape(-1, 1), b)

    # B → A 回归
    lr_ba = LinearRegression().fit(b.values.reshape(-1, 1), a)
    r2_ba = lr_ba.score(b.values.reshape(-1, 1), a)

    # 描述性统计对比
    stats_tbl = df[[col_a, col_b]].describe().T
    stats_tbl["skew"] = df[[col_a, col_b]].skew()
    stats_tbl["kurt"] = df[[col_a, col_b]].kurt()
    stats_tbl = stats_tbl[["count", "mean", "std", "min", "50%", "max", "skew", "kurt"]]
    stats_tbl.columns = ["样本数", "均值", "标准差", "最小值", "中位数", "最大值", "偏度", "峰度"]

    # 散点图数据
    plot_data = df[[col_a, col_b]].copy().reset_index(drop=True)

    # 相关程度描述
    abs_r = abs(p_r)
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

    direction = "正" if p_r > 0 else "负"
    sig_str = "显著（p<0.05）" if p_p < 0.05 else "不显著（p≥0.05）"

    lines = [
        f"## 📊 分析模式：双列关系分析",
        f"**分析列**：`{col_a}` ↔ `{col_b}`",
        f"**有效样本**：{len(df)} 行",
        f"",
        f"### 🔗 相关性检验",
        f"| 指标 | 值 | 显著性 |",
        f"|------|-----|--------|",
        f"| Pearson r | {p_r:.4f} | {'p<0.001' if p_p < 0.001 else f'p={p_p:.3f}'} |",
        f"| Spearman ρ | {s_r:.4f} | {'p<0.001' if s_p < 0.001 else f'p={s_p:.3f}'} |",
        f"",
        f"### 📐 结论",
        f"- `{col_a}` 与 `{col_b}` 存在 **{strength}{direction}相关**，统计上{sig_str}",
        f"- {col_a} → {col_b} 单变量回归：R² = {r2_ab:.4f}，斜率 = {lr_ab.coef_[0]:.4f}",
        f"- {col_b} → {col_a} 单变量回归：R² = {r2_ba:.4f}，斜率 = {lr_ba.coef_[0]:.4f}",
    ]

    return AnalysisResult(
        mode="two_column",
        target_y=col_b,
        x_columns=[col_a],
        valid_row_count=len(df),
        raw_row_count=len(df),
        summary_text="\n".join(lines),
        col_a=col_a,
        col_b=col_b,
        pearson_r=p_r,
        pearson_p=p_p,
        spearman_r=s_r,
        spearman_p=s_p,
        reg_a_to_b={"r2": r2_ab, "slope": lr_ab.coef_[0], "intercept": lr_ab.intercept_},
        reg_b_to_a={"r2": r2_ba, "slope": lr_ba.coef_[0], "intercept": lr_ba.intercept_},
        plot_df=plot_data,
        stats_df=stats_tbl,
    )


# ──────────────────────────────────────────────────────────
# 模式3：Multi X vs Y
# ──────────────────────────────────────────────────────────

def analyze_multi_x_vs_y(df: pd.DataFrame, target_y: str, x_columns: list[str]) -> AnalysisResult:
    """
    多个 X 对目标 Y 的影响。
    核心：多元线性回归 + 随机森林特征重要性 + X 间共线性检测
    """
    from scipy.stats import t as t_dist

    X = df[x_columns]
    y = df[target_y]

    # 多元线性回归
    lr = LinearRegression()
    lr.fit(X, y)
    r2 = lr.score(X, y)
    y_pred = lr.predict(X)
    n, k = len(y), len(x_columns)

    # 计算 p 值（t 检验）
    residuals = y - y_pred
    mse = (residuals ** 2).sum() / (n - k - 1)
    X_arr = np.column_stack([np.ones(n), X.values])
    try:
        cov_matrix = mse * np.linalg.inv(X_arr.T @ X_arr)
        se = np.sqrt(np.diag(cov_matrix)[1:])  # 去掉截距项的 SE
        t_stats = lr.coef_ / se
        p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n - k - 1))
    except np.linalg.LinAlgError:
        p_vals = [float("nan")] * k

    coef_df = pd.DataFrame({
        "Feature": x_columns,
        "Coefficient": lr.coef_,
        "p_value": p_vals,
        "Significant": ["*" if p < 0.05 else "" for p in p_vals],
    })

    # 随机森林特征重要性
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=1)
    rf.fit(X, y)
    imp_df = pd.DataFrame({
        "Feature": x_columns,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Pearson 相关
    corr_rows = []
    for col in x_columns:
        p_r, _ = stats.pearsonr(X[col], y)
        p_r = 0.0 if np.isnan(p_r) else p_r
        corr_rows.append({"Feature": col, "Pearson": p_r, "AbsPearson": abs(p_r)})
    corr_df = pd.DataFrame(corr_rows).sort_values("AbsPearson", ascending=False)

    merged = imp_df.merge(corr_df[["Feature", "Pearson"]], on="Feature")
    merged["Imp_Pos"] = merged.apply(lambda r: r["Importance"] if r["Pearson"] >= 0 else 0, axis=1)
    merged["Imp_Neg"] = merged.apply(lambda r: r["Importance"] if r["Pearson"] < 0 else 0, axis=1)

    # X 间相关矩阵 + 共线性警告
    x_corr = X.corr()
    warnings = []
    for i in range(len(x_columns)):
        for j in range(i + 1, len(x_columns)):
            r = x_corr.iloc[i, j]
            if abs(r) >= COLLINEARITY_THRESHOLD:
                warnings.append(
                    f"⚠ `{x_columns[i]}` 与 `{x_columns[j]}` 高度相关（r={r:.2f}），存在多重共线性风险"
                )

    # 描述性统计
    stats_tbl = df[[target_y] + x_columns].describe().T
    stats_tbl["skew"] = df[[target_y] + x_columns].skew()
    stats_tbl["kurt"] = df[[target_y] + x_columns].kurt()
    stats_tbl = stats_tbl[["count", "mean", "std", "min", "50%", "max", "skew", "kurt"]]
    stats_tbl.columns = ["样本数", "均值", "标准差", "最小值", "中位数", "最大值", "偏度", "峰度"]

    # 散点图数据
    plot_data = df[[target_y] + x_columns].copy().reset_index(drop=True)

    # 总结文本
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    lines = [
        f"## 📊 分析模式：多因素影响分析",
        f"**目标变量**：`{target_y}`",
        f"**自变量**：{', '.join([f'`{c}`' for c in x_columns])}",
        f"**有效样本**：{len(df)} 行",
        f"",
        f"### 📐 多元回归拟合",
        f"- **R²**：{r2:.4f}（调整后 R² = {adj_r2:.4f}）",
        f"- 模型可解释 **{r2:.1%}** 的 `{target_y}` 方差",
        f"",
        f"### 🏆 特征重要性 (随机森林)",
    ]
    for _, row in merged.head(5).iterrows():
        d = "↑" if row["Pearson"] >= 0 else "↓"
        lines.append(f"- `{row['Feature']}` — {row['Importance']:.1%} {d}")

    if warnings:
        lines += ["", "### ⚠ 共线性警告"] + warnings

    return AnalysisResult(
        mode="multi_x_vs_y",
        target_y=target_y,
        x_columns=x_columns,
        valid_row_count=len(df),
        raw_row_count=len(df),
        summary_text="\n".join(lines),
        feature_importance_df=merged,
        correlation_df=corr_df,
        multi_reg_r2=r2,
        multi_reg_coef_df=coef_df,
        collinearity_warnings=warnings,
        x_corr_matrix=x_corr,
        plot_df=plot_data,
        stats_df=stats_tbl,
    )
