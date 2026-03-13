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

    # ── 时序分析 (time_series) ──────────────────────────────
    time_col: str = ""
    value_col: str = ""
    ts_trend_slope: float = 0.0           # 趋势斜率（线性）
    ts_trend_r2: float = 0.0              # 趋势线 R²
    ts_adf_stat: float = 0.0              # ADF 检验统计量
    ts_adf_pvalue: float = 1.0            # ADF p 值（<0.05 则平稳）
    ts_decompose_df: Optional[pd.DataFrame] = None   # trend/seasonal/residual
    ts_forecast_df: Optional[pd.DataFrame] = None    # 预测值（若 ARIMA 成功）
    ts_arima_order: tuple = ()             # (p,d,q)

    # ── PCA 分析 ────────────────────────────────────────────
    pca_n_components: int = 0
    pca_explained_ratio: list[float] = field(default_factory=list)  # 每个主成分方差贡献率
    pca_cumulative_ratio: list[float] = field(default_factory=list)  # 累计方差贡献率
    pca_loadings_df: Optional[pd.DataFrame] = None  # 载荷矩阵（列=主成分，行=原始变量）
    pca_scores_df: Optional[pd.DataFrame] = None    # 主成分得分

    # ── ANOVA 分析 ──────────────────────────────────────────
    anova_group_col: str = ""
    anova_f_stat: float = 0.0
    anova_p_value: float = 1.0
    anova_eta_squared: float = 0.0        # 效应量 η²
    anova_group_stats_df: Optional[pd.DataFrame] = None  # 各组 n/mean/std
    anova_tukey_df: Optional[pd.DataFrame] = None        # Tukey HSD 两两比较

    # ── 逻辑回归 (logistic) ──────────────────────────────────
    logistic_coef_df: Optional[pd.DataFrame] = None  # Feature / Coef / OddsRatio / p_value
    logistic_accuracy: float = 0.0
    logistic_auc: float = 0.0
    logistic_classes: list = field(default_factory=list)

    # ── 聚类分析 (cluster) ───────────────────────────────────
    cluster_n: int = 0
    cluster_inertia_list: list[float] = field(default_factory=list)  # 肘部法则 SSE
    cluster_labels: Optional[pd.Series] = None       # 每行的簇标签
    cluster_centers_df: Optional[pd.DataFrame] = None  # 簇中心
    cluster_stats_df: Optional[pd.DataFrame] = None    # 各簇样本数/均值

    # ── 神经网络回归 (neural_reg) ────────────────────────────
    neural_r2_train: float = 0.0
    neural_r2_test: float = 0.0
    neural_mae_test: float = 0.0
    neural_vs_linear_r2: float = 0.0   # 同数据线性回归 R²（对比用）
    neural_hidden_layers: tuple = ()

    # ── 岭/套索回归 (ridge_lasso) ────────────────────────────
    ridge_lasso_type: str = ""          # "ridge" / "lasso" / "both"
    ridge_best_alpha: float = 0.0
    lasso_best_alpha: float = 0.0
    ridge_coef_df: Optional[pd.DataFrame] = None  # Feature / Ridge_Coef / Lasso_Coef / OLS_Coef
    ridge_r2: float = 0.0
    lasso_r2: float = 0.0
    lasso_selected_features: list[str] = field(default_factory=list)  # Lasso 保留的非零特征

    # ── 模型对比 (model_comparison) ─────────────────────────
    mc_x_columns: list[str] = field(default_factory=list)   # 参与建模的 X 列
    mc_comparison_df: Optional[pd.DataFrame] = None          # 模型对比汇总表
    mc_best_model_name: str = ""                              # 最优模型名称
    mc_best_model_r2: float = 0.0                            # 最优模型 R²
    mc_predictions: dict = field(default_factory=dict)        # {模型名: {"actual": [...], "predicted": [...], "residual": [...]}}
    mc_feature_types: dict = field(default_factory=dict)      # {列名: "continuous" / "categorical"}
    mc_cv_scores: dict = field(default_factory=dict)          # {模型名: [cv_r2_fold1, fold2, ...]}

    # ── 对比分析 (compare) ───────────────────────────────────
    compare_group_col: str = ""
    compare_value_col: str = ""
    compare_group_stats_df: Optional[pd.DataFrame] = None   # 各组 n/mean/std/median
    compare_test_name: str = ""                              # "t-test" / "ANOVA"
    compare_stat: float = 0.0
    compare_p_value: float = 1.0

    # ── 交叉分析 (crosstab) ──────────────────────────────────
    crosstab_row_col: str = ""
    crosstab_col_col: str = ""
    crosstab_value_col: str = ""                             # 空则计数，非空则聚合
    crosstab_agg: str = "count"                              # count / sum / mean
    crosstab_df: Optional[pd.DataFrame] = None               # 透视表结果
    crosstab_row_pct_df: Optional[pd.DataFrame] = None       # 行百分比表


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


# ──────────────────────────────────────────────────────────
# 模式4：时间序列分析
# ──────────────────────────────────────────────────────────

def analyze_time_series(df: pd.DataFrame, time_col: str, value_col: str) -> AnalysisResult:
    """
    时间序列分析：线性趋势 + ADF平稳性检验 + 季节分解（若数据量足够）+ ARIMA预测（可选）
    time_col：时间列（日期字符串或数值序号皆可）
    value_col：分析的数值列
    """
    result = AnalysisResult(
        mode="time_series",
        target_y=value_col,
        x_columns=[time_col],
        valid_row_count=len(df),
        raw_row_count=len(df),
        time_col=time_col,
        value_col=value_col,
    )

    y = df[value_col].values
    n = len(y)
    x_idx = np.arange(n)

    # 线性趋势
    slope, intercept, r, p, _ = stats.linregress(x_idx, y)
    result.ts_trend_slope = float(slope)
    result.ts_trend_r2 = float(r ** 2)

    # ADF 平稳性检验
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(y, autolag="AIC")
        result.ts_adf_stat = float(adf_result[0])
        result.ts_adf_pvalue = float(adf_result[1])
    except Exception:
        result.ts_adf_stat = 0.0
        result.ts_adf_pvalue = 1.0

    # 季节分解（需要至少 2 个周期，即 n >= 2*freq）
    if n >= 24:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            freq = 12 if n >= 24 else (7 if n >= 14 else None)
            if freq:
                decomp = seasonal_decompose(y, model="additive", period=freq, extrapolate_trend="freq")
                result.ts_decompose_df = pd.DataFrame({
                    "trend":    decomp.trend,
                    "seasonal": decomp.seasonal,
                    "residual": decomp.resid,
                })
        except Exception:
            pass

    # ARIMA 预测（若 statsmodels 可用且数据量 >= 30）
    if n >= 30:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(y, order=(1, 1, 1))
            fit = model.fit()
            forecast_steps = max(5, n // 10)
            forecast = fit.forecast(steps=forecast_steps)
            result.ts_forecast_df = pd.DataFrame({
                "step":  range(1, forecast_steps + 1),
                "value": forecast,
            })
            result.ts_arima_order = (1, 1, 1)
        except Exception:
            pass

    # 生成 summary_text
    stationary = result.ts_adf_pvalue < 0.05
    trend_dir = "上升" if slope > 0 else "下降"
    lines = [
        f"## 📈 时间序列分析",
        f"**分析列**：`{value_col}`  **时间列**：`{time_col}`  **样本量**：{n} 行",
        f"",
        f"### 趋势分析",
        f"- 整体趋势：**{trend_dir}**（斜率 = {slope:.4f}，趋势 R² = {r**2:.4f}）",
        f"",
        f"### ADF 平稳性检验",
        f"- ADF 统计量 = {result.ts_adf_stat:.4f}，p = {result.ts_adf_pvalue:.4f}",
        f"- 序列{'**平稳**' if stationary else '**非平稳**'}（{'无需差分' if stationary else '建议一阶差分'}）",
    ]
    if result.ts_forecast_df is not None:
        lines += [
            f"",
            f"### ARIMA(1,1,1) 预测",
            f"- 已生成未来 {len(result.ts_forecast_df)} 期预测值",
            f"- 预测下一期：{result.ts_forecast_df['value'].iloc[0]:.4f}",
        ]
    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 模式5：PCA 主成分分析
# ──────────────────────────────────────────────────────────

def analyze_pca(df: pd.DataFrame, columns: list[str], n_components: int = 0) -> AnalysisResult:
    """
    PCA 主成分分析：解释方差比 + 载荷矩阵 + 主成分得分
    columns：参与 PCA 的数值列（至少 2 列）
    n_components：主成分数，0 = 自动（保留解释方差 ≥ 80% 所需的最少成分数）
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = df[columns].dropna()
    n, p = X.shape

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 先用全量 PCA 得到解释方差
    pca_full = PCA()
    pca_full.fit(X_scaled)
    explained = pca_full.explained_variance_ratio_

    # 确定主成分数
    if n_components <= 0:
        cumsum = np.cumsum(explained)
        n_components = int(np.searchsorted(cumsum, 0.80) + 1)
        n_components = min(n_components, p)

    n_components = min(n_components, p, n)

    # 拟合目标维度
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_).tolist()
    pc_names = [f"PC{i+1}" for i in range(n_components)]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=columns,
        columns=pc_names,
    )
    scores_df = pd.DataFrame(scores, columns=pc_names)

    result = AnalysisResult(
        mode="pca",
        target_y="",
        x_columns=columns,
        valid_row_count=n,
        raw_row_count=len(df),
        pca_n_components=n_components,
        pca_explained_ratio=pca.explained_variance_ratio_.tolist(),
        pca_cumulative_ratio=cumulative,
        pca_loadings_df=loadings_df,
        pca_scores_df=scores_df,
    )

    lines = [
        f"## 🔬 PCA 主成分分析",
        f"**输入变量**：{len(columns)} 个  **保留主成分**：{n_components} 个  **有效样本**：{n} 行",
        f"",
        f"### 方差贡献率",
        f"| 主成分 | 个体方差贡献 | 累计贡献 |",
        f"|--------|------------|---------|",
    ]
    for i, (r, c) in enumerate(zip(pca.explained_variance_ratio_, cumulative)):
        lines.append(f"| PC{i+1} | {r:.1%} | {c:.1%} |")

    lines += [
        f"",
        f"### 主要载荷（PC1 ~ PC{min(3, n_components)}）",
        f"| 变量 | " + " | ".join(f"PC{i+1}" for i in range(min(3, n_components))) + " |",
        f"|------| " + "------|" * min(3, n_components),
    ]
    for var in columns:
        vals = [f"{loadings_df.loc[var, f'PC{i+1}']:+.3f}" for i in range(min(3, n_components))]
        lines.append(f"| `{var}` | " + " | ".join(vals) + " |")

    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 模式6：单因素 ANOVA + Tukey HSD
# ──────────────────────────────────────────────────────────

def analyze_anova(df: pd.DataFrame, target_y: str, group_col: str) -> AnalysisResult:
    """
    单因素 ANOVA：组间均值差异检验 + Tukey HSD 事后检验 + 效应量 η²
    target_y：数值型因变量
    group_col：分组变量（分类列或低基数数值列）
    """
    groups = {}
    for gname, gdf in df.groupby(group_col):
        groups[gname] = gdf[target_y].dropna().values

    group_names = list(groups.keys())
    group_arrays = list(groups.values())

    # 单因素 ANOVA
    f_stat, p_value = stats.f_oneway(*group_arrays)

    # 效应量 η²（组间 SS / 总 SS）
    grand_mean = df[target_y].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_arrays)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in group_arrays)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # 各组描述统计
    group_stats_rows = []
    for gname, g in groups.items():
        group_stats_rows.append({
            "组别": str(gname),
            "样本量": len(g),
            "均值": round(float(g.mean()), 4),
            "标准差": round(float(g.std(ddof=1)), 4),
            "最小值": round(float(g.min()), 4),
            "最大值": round(float(g.max()), 4),
        })
    group_stats_df = pd.DataFrame(group_stats_rows)

    # Tukey HSD（需要 statsmodels）
    tukey_df = None
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        all_vals = np.concatenate(group_arrays)
        all_labels = np.concatenate([[str(gn)] * len(g) for gn, g in groups.items()])
        tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
    except Exception:
        pass

    result = AnalysisResult(
        mode="anova",
        target_y=target_y,
        x_columns=[group_col],
        valid_row_count=len(df),
        raw_row_count=len(df),
        anova_group_col=group_col,
        anova_f_stat=float(f_stat),
        anova_p_value=float(p_value),
        anova_eta_squared=float(eta_squared),
        anova_group_stats_df=group_stats_df,
        anova_tukey_df=tukey_df,
    )

    sig_str = "**显著**（p<0.05）" if p_value < 0.05 else "**不显著**（p≥0.05）"
    effect_str = "大" if eta_squared > 0.14 else ("中" if eta_squared > 0.06 else "小")
    lines = [
        f"## 📊 单因素 ANOVA 方差分析",
        f"**因变量**：`{target_y}`  **分组变量**：`{group_col}`  **组数**：{len(groups)}",
        f"",
        f"### 检验结果",
        f"- F 统计量 = {f_stat:.4f}，p = {p_value:.4f}",
        f"- 结论：组间均值差异{sig_str}",
        f"- 效应量 η² = {eta_squared:.4f}（{effect_str}效应）",
    ]
    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 模式7：逻辑回归
# ──────────────────────────────────────────────────────────

def analyze_logistic(df: pd.DataFrame, target_y: str, x_cols: list[str]) -> AnalysisResult:
    """
    逻辑回归：适用于二元目标变量（0/1）
    输出系数、Odds Ratio、准确率、AUC-ROC
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    x_cols = [c for c in x_cols if c in df.columns and c != target_y]
    if not x_cols:
        raise ValueError("没有有效的自变量列，请确认列名存在于数据中")

    X = df[x_cols].values
    y = df[target_y].values
    classes = sorted(set(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) == 2 else None
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    auc = 0.0
    if len(classes) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = 0.0

    coefs = model.coef_[0]
    import math
    coef_rows = []
    for feat, coef in zip(x_cols, coefs):
        coef_rows.append({
            "Feature": feat,
            "Coefficient": float(coef),
            "OddsRatio": float(math.exp(coef)),
        })
    coef_df = pd.DataFrame(coef_rows).sort_values("Coefficient", key=abs, ascending=False)

    result = AnalysisResult(
        mode="logistic",
        target_y=target_y,
        x_columns=x_cols,
        valid_row_count=len(df),
        raw_row_count=len(df),
        logistic_coef_df=coef_df,
        logistic_accuracy=accuracy,
        logistic_auc=auc,
        logistic_classes=classes,
    )

    lines = [
        f"## 🎯 逻辑回归分析",
        f"**目标变量**：`{target_y}`（{'二元' if len(classes)==2 else str(len(classes))+'类'}分类）",
        f"**有效样本**：{len(df)} 行  **测试集准确率**：{accuracy:.1%}",
    ]
    if auc > 0:
        lines.append(f"**AUC-ROC**：{auc:.4f}（{'优秀' if auc>0.9 else '良好' if auc>0.7 else '一般'}）")
    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 模式8：K-Means 聚类分析
# ──────────────────────────────────────────────────────────

def analyze_cluster(df: pd.DataFrame, columns: list[str], n_clusters: int = 0) -> AnalysisResult:
    """
    K-Means 聚类：肘部法则自动选 K + 簇统计
    n_clusters：0 = 自动（肘部法则，测试 k=2..min(8, n//10)）
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = df[columns].dropna()
    n = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 肘部法则
    k_max = min(8, n // 10, len(columns) * 2)
    k_max = max(k_max, 3)
    inertia_list = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia_list.append(float(km.inertia_))

    # 自动选 K：肘部点（最大曲率变化）
    if n_clusters <= 0:
        if len(inertia_list) >= 2:
            diffs = [inertia_list[i] - inertia_list[i+1] for i in range(len(inertia_list)-1)]
            diffs2 = [diffs[i] - diffs[i+1] for i in range(len(diffs)-1)]
            best_idx = int(np.argmax(diffs2)) + 2 if diffs2 else 0
            n_clusters = best_idx + 2
        else:
            n_clusters = 3

    n_clusters = min(n_clusters, k_max)

    # 最终拟合
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)
    centers_scaled = km_final.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_orig, columns=columns)
    centers_df.index.name = "Cluster"

    # 各簇统计
    X_copy = X.copy()
    X_copy["_cluster"] = labels
    stats_rows = []
    for k in range(n_clusters):
        grp = X_copy[X_copy["_cluster"] == k]
        row = {"簇": f"Cluster {k}", "样本量": len(grp)}
        for col in columns:
            row[f"{col}_均值"] = round(float(grp[col].mean()), 4)
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)

    result = AnalysisResult(
        mode="cluster",
        target_y="",
        x_columns=columns,
        valid_row_count=n,
        raw_row_count=len(df),
        cluster_n=n_clusters,
        cluster_inertia_list=inertia_list,
        cluster_labels=pd.Series(labels),
        cluster_centers_df=centers_df,
        cluster_stats_df=stats_df,
    )

    lines = [
        f"## 🔵 K-Means 聚类分析",
        f"**聚类变量**：{len(columns)} 个  **自动选定 K**：{n_clusters} 簇  **有效样本**：{n} 行",
        f"",
        f"### 各簇大小",
    ]
    for row in stats_rows:
        lines.append(f"- **{row['簇']}**：{row['样本量']} 个样本")
    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 模式9：MLP 神经网络回归
# ──────────────────────────────────────────────────────────

def analyze_neural_reg(df: pd.DataFrame, target_y: str, x_cols: list[str]) -> AnalysisResult:
    """
    MLP 神经网络回归 + 与线性回归对比
    使用 sklearn MLPRegressor，默认两层隐藏层
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    # 过滤不存在的列（防止 cols_needed 与 clean_df 不一致）
    x_cols = [c for c in x_cols if c in df.columns and c != target_y]
    if not x_cols:
        raise ValueError(f"没有有效的自变量列，请确认列名存在于数据中")

    X = df[x_cols].values
    y = df[target_y].values
    n = len(y)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 根据样本量决定网络规模
    h1 = min(64, max(8, len(x_cols) * 4))
    h2 = h1 // 2
    hidden_layers = (h1, h2)

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)

    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    r2_train = float(r2_score(y_train, y_pred_train))
    r2_test = float(r2_score(y_test, y_pred_test))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))

    # 线性回归对比基准
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_r2 = float(r2_score(y_test, lr.predict(X_test)))

    result = AnalysisResult(
        mode="neural_reg",
        target_y=target_y,
        x_columns=x_cols,
        valid_row_count=n,
        raw_row_count=n,
        neural_r2_train=r2_train,
        neural_r2_test=r2_test,
        neural_mae_test=mae_test,
        neural_vs_linear_r2=lr_r2,
        neural_hidden_layers=hidden_layers,
    )

    improvement = r2_test - lr_r2
    lines = [
        f"## 🧠 神经网络回归（MLP）",
        f"**目标变量**：`{target_y}`  **输入特征**：{len(x_cols)} 个  **有效样本**：{n} 行",
        f"**网络结构**：{len(x_cols)} → {h1} → {h2} → 1",
        f"",
        f"### 模型性能",
        f"| 指标 | MLP 神经网络 | 线性回归基准 |",
        f"|------|------------|------------|",
        f"| R²（测试集） | **{r2_test:.4f}** | {lr_r2:.4f} |",
        f"| 训练集 R² | {r2_train:.4f} | — |",
        f"| MAE（测试集） | {mae_test:.4f} | — |",
        f"",
        f"### 📌 结论",
        f"- MLP 相比线性回归 R² {'提升' if improvement > 0 else '下降'} {abs(improvement):.4f}",
    ]
    if improvement > 0.05:
        lines.append("- ✅ 数据存在**非线性关系**，神经网络显著优于线性模型")
    elif improvement < -0.02:
        lines.append("- ⚠️ 线性回归表现更好，可能存在过拟合，建议增加数据量")
    else:
        lines.append("- 两模型表现相近，数据关系以线性为主")

    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 模式10：岭回归 + 套索回归
# ──────────────────────────────────────────────────────────

def analyze_ridge_lasso(df: pd.DataFrame, target_y: str, x_cols: list[str],
                         reg_type: str = "both") -> AnalysisResult:
    """
    岭回归（Ridge）和套索回归（Lasso）：解决多重共线性，自动交叉验证选最优 alpha
    reg_type: "ridge" / "lasso" / "both"（默认）
    """
    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    x_cols = [c for c in x_cols if c in df.columns and c != target_y]
    if not x_cols:
        raise ValueError("没有有效的自变量列，请确认列名存在于数据中")

    X = df[x_cols].values
    y = df[target_y].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    ridge_coefs = np.zeros(len(x_cols))
    lasso_coefs = np.zeros(len(x_cols))
    ridge_r2 = 0.0
    lasso_r2 = 0.0
    ridge_alpha = 0.0
    lasso_alpha = 0.0
    lasso_selected = []

    if reg_type in ("ridge", "both"):
        ridge = RidgeCV(alphas=alphas, cv=5)
        ridge.fit(X_scaled, y)
        ridge_coefs = ridge.coef_
        ridge_r2 = float(r2_score(y, ridge.predict(X_scaled)))
        ridge_alpha = float(ridge.alpha_)

    if reg_type in ("lasso", "both"):
        lasso = LassoCV(alphas=alphas, cv=5, max_iter=5000)
        lasso.fit(X_scaled, y)
        lasso_coefs = lasso.coef_
        lasso_r2 = float(r2_score(y, lasso.predict(X_scaled)))
        lasso_alpha = float(lasso.alpha_)
        lasso_selected = [x_cols[i] for i, c in enumerate(lasso_coefs) if abs(c) > 1e-6]

    # OLS 基准
    lr = LinearRegression()
    lr.fit(X_scaled, y)
    ols_coefs = lr.coef_

    coef_rows = []
    for i, feat in enumerate(x_cols):
        coef_rows.append({
            "Feature": feat,
            "OLS_Coef": float(ols_coefs[i]),
            "Ridge_Coef": float(ridge_coefs[i]),
            "Lasso_Coef": float(lasso_coefs[i]),
        })
    coef_df = pd.DataFrame(coef_rows)

    result = AnalysisResult(
        mode="ridge_lasso",
        target_y=target_y,
        x_columns=x_cols,
        valid_row_count=len(df),
        raw_row_count=len(df),
        ridge_lasso_type=reg_type,
        ridge_best_alpha=ridge_alpha,
        lasso_best_alpha=lasso_alpha,
        ridge_coef_df=coef_df,
        ridge_r2=ridge_r2,
        lasso_r2=lasso_r2,
        lasso_selected_features=lasso_selected,
    )

    lines = [
        f"## 🔧 岭回归 + 套索回归",
        f"**目标变量**：`{target_y}`  **自变量**：{len(x_cols)} 个  **有效样本**：{len(df)} 行",
        f"",
        f"### 模型对比",
        f"| 模型 | 最优 α | R²（训练集） |",
        f"|------|-------|------------|",
        f"| 岭回归（Ridge） | {ridge_alpha:.4g} | {ridge_r2:.4f} |",
        f"| 套索回归（Lasso） | {lasso_alpha:.4g} | {lasso_r2:.4f} |",
        f"",
        f"### Lasso 变量选择",
        f"保留 **{len(lasso_selected)}** 个非零系数特征：{', '.join(f'`{c}`' for c in lasso_selected) if lasso_selected else '无（所有系数被压缩至0）'}",
    ]
    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 辅助：变量类型检测
# ──────────────────────────────────────────────────────────

def _detect_feature_types(df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    """
    自动检测每列的变量类型。
    规则：
    - object / category dtype → categorical
    - 整数列且唯一值 ≤ min(10, n*0.05) → categorical（视为分组编码）
    - 其余 → continuous
    """
    n = len(df)
    types = {}
    for col in columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            types[col] = "categorical"
        elif df[col].dtype in (int, "int64", "int32") and df[col].nunique() <= min(10, max(3, n * 0.05)):
            types[col] = "categorical"
        else:
            types[col] = "continuous"
    return types


# ──────────────────────────────────────────────────────────
# 模式11：多模型对比（Model Comparison / AutoML-lite）
# ──────────────────────────────────────────────────────────

def analyze_model_comparison(
    df: pd.DataFrame,
    target_y: str,
    x_cols: list[str],
    top_n: int = 5,
) -> AnalysisResult:
    """
    多模型赛马：对 top_n 个 X 列自动训练多个回归/分类模型，5 折 CV 评估，
    找出最优模型。全流程使用采样数据，保证大数据集响应速度。
    """
    from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.metrics import accuracy_score, f1_score
    import warnings

    # ── 列名防御过滤 ──
    x_cols = [c for c in x_cols if c in df.columns and c != target_y]
    if not x_cols:
        raise ValueError("没有有效的自变量列，请确认列名存在于数据中")

    # ── 变量类型检测 ──
    feature_types = _detect_feature_types(df, x_cols)
    y_type = _detect_feature_types(df, [target_y]).get(target_y, "continuous")

    # ── 预处理：分类 X 列标签编码 ──
    X_raw = df[x_cols].copy()
    for col, ftype in feature_types.items():
        if ftype == "categorical":
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    X_all = X_raw.values.astype(float)
    y_raw = df[target_y].values

    # ── Y 编码（分类任务）──
    y_le = None
    if y_type == "categorical":
        y_le = LabelEncoder()
        y_all = y_le.fit_transform(y_raw.astype(str)).astype(int)
        task = "classification"
    else:
        y_all = y_raw.astype(float)
        task = "regression"

    n = len(y_all)

    # ── 全流程统一采样：CV + 训练 + 预测 都用同一份数据 ──
    MAX_ROWS = 8000
    if n > MAX_ROWS:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, MAX_ROWS, replace=False)
        X, y = X_all[idx], y_all[idx]
        sample_note = f"（数据量 {n} 行，随机抽样 {MAX_ROWS} 行用于训练与评估）"
    else:
        X, y = X_all, y_all
        sample_note = ""

    n_sample = len(y)

    # ── 标准化 ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 候选模型（根据任务类型选择）──
    n_feat = len(x_cols)
    mlp_layers = (min(64, n_feat * 8), min(32, n_feat * 4))

    if task == "regression":
        models = {
            "线性回归":    LinearRegression(),
            "岭回归":      RidgeCV(alphas=[0.1, 1.0, 10.0]),
            "随机森林":    RandomForestRegressor(
                               n_estimators=50, max_depth=8,
                               random_state=RF_RANDOM_STATE, n_jobs=-1),
            "梯度提升":    HistGradientBoostingRegressor(
                               max_iter=80, max_depth=5,
                               random_state=RF_RANDOM_STATE),
            "神经网络MLP": MLPRegressor(
                               hidden_layer_sizes=mlp_layers,
                               max_iter=150, random_state=RF_RANDOM_STATE,
                               early_stopping=True, n_iter_no_change=10),
        }
        scoring = "r2"
        cv_cls = KFold
    else:
        n_classes = len(np.unique(y))
        avg = "binary" if n_classes == 2 else "macro"
        models = {
            "逻辑回归":    LogisticRegression(max_iter=300, random_state=RF_RANDOM_STATE, n_jobs=-1),
            "随机森林":    RandomForestClassifier(
                               n_estimators=50, max_depth=8,
                               random_state=RF_RANDOM_STATE, n_jobs=-1),
            "梯度提升":    HistGradientBoostingClassifier(
                               max_iter=80, max_depth=5,
                               random_state=RF_RANDOM_STATE),
            "神经网络MLP": MLPClassifier(
                               hidden_layer_sizes=mlp_layers,
                               max_iter=150, random_state=RF_RANDOM_STATE,
                               early_stopping=True, n_iter_no_change=10),
        }
        scoring = "f1_weighted"
        cv_cls = StratifiedKFold

    n_splits = max(2, min(5, n_sample // 50))
    kf = cv_cls(n_splits=n_splits, shuffle=True, random_state=42)

    comparison_rows = []
    predictions_store = {}
    cv_scores_store = {}

    for model_name, model in models.items():
        use_scaled = model_name in ("线性回归", "岭回归", "逻辑回归", "神经网络MLP")
        X_use = X_scaled if use_scaled else X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(model, X_use, y, cv=kf, scoring=scoring, n_jobs=-1)
            model.fit(X_use, y)

        y_pred = model.predict(X_use)
        cv_mean = float(np.mean(cv_scores))
        cv_std  = float(np.std(cv_scores))

        if task == "regression":
            r2   = float(r2_score(y, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            mae  = float(mean_absolute_error(y, y_pred))
            comparison_rows.append({
                "模型":      model_name,
                "训练集R²":  round(r2, 4),
                "CV均值R²":  round(cv_mean, 4),
                "CV标准差":  round(cv_std, 4),
                "RMSE":      round(rmse, 4),
                "MAE":       round(mae, 4),
            })
        else:
            acc = float(accuracy_score(y, y_pred))
            f1  = float(f1_score(y, y_pred, average="weighted", zero_division=0))
            comparison_rows.append({
                "模型":        model_name,
                "训练集准确率": round(acc, 4),
                "CV均值F1":    round(cv_mean, 4),
                "CV标准差":    round(cv_std, 4),
            })

        predictions_store[model_name] = {
            "actual":    y.tolist(),
            "predicted": y_pred.tolist(),
            "residual":  (y - y_pred).tolist(),
        }
        cv_scores_store[model_name] = cv_scores.tolist()

    sort_col = "CV均值R²" if task == "regression" else "CV均值F1"
    comparison_df = pd.DataFrame(comparison_rows).sort_values(sort_col, ascending=False)
    best_row  = comparison_df.iloc[0]
    best_name = str(best_row["模型"])
    best_score = float(best_row[sort_col])

    result = AnalysisResult(
        mode="model_comparison",
        target_y=target_y,
        x_columns=x_cols,
        valid_row_count=n_sample,
        raw_row_count=n,
        mc_x_columns=x_cols,
        mc_comparison_df=comparison_df,
        mc_best_model_name=best_name,
        mc_best_model_r2=best_score,
        mc_predictions=predictions_store,
        mc_feature_types=feature_types,
        mc_cv_scores=cv_scores_store,
    )

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    task_label = "分类" if task == "classification" else "回归"
    score_label = "F1（加权）" if task == "classification" else "R²"
    lines = [
        f"## 多模型对比分析（AutoML-lite）",
        f"**目标变量**：`{target_y}`（{'分类变量' if task == 'classification' else '连续变量'}）  "
        f"**输入特征**：{', '.join(f'`{c}`' for c in x_cols)}",
        f"**有效样本**：{n} 行  **评估方式**：{n_splits} 折交叉验证{sample_note}",
        f"**任务类型**：{task_label}",
        f"",
        f"### 最优模型：{best_name}",
        f"- 交叉验证 {score_label} = **{best_score:.4f}**（越高越好）",
        f"",
        f"### 模型对比排名",
    ]

    if task == "regression":
        lines += [
            "| 排名 | 模型 | CV R² | CV σ | RMSE | MAE |",
            "|------|------|-------|------|------|-----|",
        ]
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            medal = medals[i] if i < len(medals) else str(i + 1)
            lines.append(
                f"| {medal} | **{row['模型']}** | {row['CV均值R²']:.4f} | "
                f"±{row['CV标准差']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |"
            )
    else:
        lines += [
            "| 排名 | 模型 | CV F1 | CV σ | 训练准确率 |",
            "|------|------|-------|------|-----------|",
        ]
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            medal = medals[i] if i < len(medals) else str(i + 1)
            lines.append(
                f"| {medal} | **{row['模型']}** | {row['CV均值F1']:.4f} | "
                f"±{row['CV标准差']:.4f} | {row['训练集准确率']:.4f} |"
            )

    lines.append("")
    lines.append("> Excel 报告中每个模型均有独立 Sheet，包含预测值 vs 实际值散点图与残差图。")
    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 对比分析（compare）
# ──────────────────────────────────────────────────────────

def analyze_compare(df: pd.DataFrame, value_col: str, group_col: str) -> AnalysisResult:
    """
    对比分析：比较不同分组之间目标值的差异。
    - 2组：独立样本 t 检验
    - >2组：单因素 ANOVA
    """
    from scipy import stats
    import warnings

    group_names = df[group_col].dropna().unique().tolist()

    stat_rows = []
    for name, grp in df.groupby(group_col):
        vals = grp[value_col].dropna()
        n = len(vals)
        # 单元素组 std 无意义，用 NaN 而不是触发 RuntimeWarning
        std_val = float(vals.std()) if n >= 2 else float("nan")
        stat_rows.append({
            "组别": name,
            "样本量": n,
            "均值": round(float(vals.mean()), 4) if n > 0 else float("nan"),
            "标准差": round(std_val, 4) if not (std_val != std_val) else "—",
            "中位数": round(float(vals.median()), 4) if n > 0 else float("nan"),
            "最小值": round(float(vals.min()), 4) if n > 0 else float("nan"),
            "最大值": round(float(vals.max()), 4) if n > 0 else float("nan"),
        })
    group_stats_df = pd.DataFrame(stat_rows)

    # 只保留样本量 >= 2 的组用于统计检验
    groups_for_test = [
        grp[value_col].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp[value_col].dropna()) >= 2
    ]

    insufficient_note = ""
    n_skipped = len(group_names) - len(groups_for_test)
    if n_skipped > 0:
        insufficient_note = f"\n> ⚠️ {n_skipped} 个分组样本量不足（< 2），已排除在统计检验之外。"

    stat, p_val, test_name = float("nan"), float("nan"), "无法检验"

    if len(groups_for_test) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if len(groups_for_test) == 2:
                stat, p_val = stats.ttest_ind(*groups_for_test, equal_var=False)
                test_name = "独立样本 t 检验"
            else:
                stat, p_val = stats.f_oneway(*groups_for_test)
                test_name = "单因素 ANOVA"
    else:
        test_name = "样本量不足，无法执行统计检验"

    import math
    if math.isnan(stat) or math.isnan(p_val):
        sig = "无法判断（数据不足）"
    else:
        sig = "**显著差异**（p < 0.05）" if p_val < 0.05 else "无显著差异（p ≥ 0.05）"

    n_groups = len(group_names)

    result = AnalysisResult(
        mode="compare",
        target_y=value_col,
        x_columns=[group_col],
        valid_row_count=len(df),
        raw_row_count=len(df),
        compare_group_col=group_col,
        compare_value_col=value_col,
        compare_group_stats_df=group_stats_df,
        compare_test_name=test_name,
        compare_stat=stat if not math.isnan(stat) else None,
        compare_p_value=p_val if not math.isnan(p_val) else None,
    )

    lines = [
        f"## 📊 对比分析：{value_col} 按 {group_col}",
        f"",
        f"### 分组统计（共 {n_groups} 组）",
        f"",
        "| 组别 | 样本量 | 均值 | 标准差 | 中位数 |",
        "|------|--------|------|--------|--------|",
    ]
    for _, r in group_stats_df.iterrows():
        lines.append(f"| {r['组别']} | {r['样本量']} | {r['均值']} | {r['标准差']} | {r['中位数']} |")

    lines += [
        f"",
        f"### 统计检验：{test_name}",
    ]
    if not math.isnan(stat) and not math.isnan(p_val):
        lines += [
            f"- 统计量：**{stat:.4f}**",
            f"- p 值：**{p_val:.4f}**",
            f"- 结论：各组 {value_col} {sig}",
        ]
    else:
        lines.append(f"- {sig}")
    if insufficient_note:
        lines.append(insufficient_note)

    result.summary_text = "\n".join(lines)
    return result


# ──────────────────────────────────────────────────────────
# 交叉分析（crosstab）
# ──────────────────────────────────────────────────────────

def analyze_crosstab(df: pd.DataFrame,
                     row_col: str,
                     col_col: str,
                     value_col: str = "",
                     agg: str = "count") -> AnalysisResult:
    """
    交叉分析（透视表）：
    - value_col 为空 / agg=="count"：计数交叉表 + 卡方检验
    - 否则：聚合（sum/mean）透视表 + 描述性统计
    """
    from scipy import stats
    import math

    is_count = (agg == "count" or not value_col)

    if is_count:
        pivot = pd.crosstab(df[row_col], df[col_col])
        agg_desc = "计数"
    else:
        agg_fn = "sum" if agg == "sum" else "mean"
        pivot = df.pivot_table(index=row_col, columns=col_col,
                               values=value_col, aggfunc=agg_fn)
        pivot = pivot.round(4)
        agg_desc = "求和" if agg_fn == "sum" else "求均值"

    n_rows, n_cols = pivot.shape

    # ── 统计检验 ────────────────────────────────────────────
    chi2_stat = chi2_p = cramers_v = None
    chi2_note = ""
    if is_count and n_rows >= 2 and n_cols >= 2:
        try:
            chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(pivot)
            # Cramér's V（效应量，0~1）
            n_total = int(pivot.values.sum())
            cramers_v = math.sqrt(chi2_stat / (n_total * (min(n_rows, n_cols) - 1))) if n_total > 0 else 0
            # 期望频数 < 5 的格子占比（卡方检验可靠性判断）
            low_exp_pct = (expected < 5).sum() / expected.size
            if low_exp_pct > 0.2:
                chi2_note = f"（⚠️ {low_exp_pct:.0%} 的格子期望频数 < 5，卡方结果仅供参考）"
        except Exception:
            pass

    # ── 行列百分比（仅计数模式）──────────────────────────────
    row_pct_df = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1) if is_count else None
    col_pct_df = pivot.div(pivot.sum(axis=0), axis=1).mul(100).round(1) if is_count else None

    # ── 找最高频/最大值组合 ──────────────────────────────────
    top_combos = []
    flat = pivot.stack().reset_index()
    flat.columns = [row_col, col_col, "值"]
    flat_sorted = flat.sort_values("值", ascending=False).head(5)
    for _, r in flat_sorted.iterrows():
        top_combos.append((r[row_col], r[col_col], r["值"]))

    # ── 构建结果 ────────────────────────────────────────────
    result = AnalysisResult(
        mode="crosstab",
        target_y=value_col or "计数",
        x_columns=[row_col, col_col],
        valid_row_count=len(df),
        raw_row_count=len(df),
        crosstab_row_col=row_col,
        crosstab_col_col=col_col,
        crosstab_value_col=value_col,
        crosstab_agg=agg,
        crosstab_df=pivot,
        crosstab_row_pct_df=row_pct_df,
    )

    # ── summary_text ────────────────────────────────────────
    lines = [
        f"## 🔀 交叉分析：{row_col} × {col_col}",
        f"",
        f"- 聚合方式：**{agg_desc}**",
        f"- 透视表规模：**{n_rows} 行 × {n_cols} 列**",
        f"- 总样本量：**{len(df)} 条**",
        f"",
    ]

    # 卡方检验结论
    if chi2_stat is not None:
        sig_text = "**存在显著关联**（p < 0.05）" if chi2_p < 0.05 else "无显著关联（p ≥ 0.05）"
        if cramers_v is not None:
            if cramers_v < 0.1:
                effect = "极弱"
            elif cramers_v < 0.3:
                effect = "弱"
            elif cramers_v < 0.5:
                effect = "中等"
            else:
                effect = "强"
            effect_desc = f"Cramér's V = **{cramers_v:.3f}**（{effect}关联）"
        else:
            effect_desc = ""

        lines += [
            f"### 📐 独立性检验（卡方检验）{chi2_note}",
            f"- χ² 统计量：**{chi2_stat:.4f}**，自由度：**{dof}**",
            f"- p 值：**{chi2_p:.4f}**",
            f"- 结论：{row_col} 与 {col_col} {sig_text}",
        ]
        if effect_desc:
            lines.append(f"- 关联强度：{effect_desc}")
        lines.append("")

    # Top 组合
    if top_combos:
        lines.append(f"### 🏆 Top 5 高频组合")
        lines.append(f"| {row_col} | {col_col} | {'频次' if is_count else agg_desc} |")
        lines.append("|------|------|------|")
        for r, c, v in top_combos:
            lines.append(f"| {r} | {c} | {v} |")
        lines.append("")

    # 透视表预览
    if not pivot.empty:
        preview = pivot.iloc[:6, :6]
        lines.append(f"### 📊 透视表预览（前 {min(6, n_rows)} 行 × {min(6, n_cols)} 列）")
        try:
            lines.append(preview.to_markdown())
        except ImportError:
            lines.append("```\n" + preview.to_string() + "\n```")

    # 行百分比预览（计数模式）
    if row_pct_df is not None and not row_pct_df.empty:
        lines.append(f"\n### 📈 行百分比（%）— {row_col} 内各 {col_col} 占比")
        preview_pct = row_pct_df.iloc[:6, :6]
        try:
            lines.append(preview_pct.to_markdown())
        except ImportError:
            lines.append("```\n" + preview_pct.to_string() + "\n```")

    lines.append("\n> 完整透视表已写入 Excel 报告，支持下载查看。")

    result.summary_text = "\n".join(lines)
    return result
