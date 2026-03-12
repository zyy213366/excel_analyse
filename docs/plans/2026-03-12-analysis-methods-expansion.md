# 分析方法扩展实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有三模式（y_vs_all / two_column / multi_x_vs_y）基础上，新增 7 种主流统计/机器学习分析方法，覆盖时间序列、PCA、ANOVA、逻辑回归、聚类、神经网络回归、岭/套索回归。

**Architecture:** 新增模式均在 `core/analysis_engine.py` 中实现为独立函数，结果统一扩展 `AnalysisResult` dataclass；`result_formatter.py` 新增对应格式化器；`api/routes.py` 路由扩展；`nlp_parser.py` 提示词更新。不破坏任何现有三个模式。

**Tech Stack:** scikit-learn（已有，MLP/Logistic/Ridge/Lasso/KMeans/PCA）、scipy（已有，ANOVA）、statsmodels（新增，ARIMA/ADF/时序分解）、numpy/pandas（已有）

---

## 当前分析模式（只读，勿改）

| 模式键 | 函数 | 核心算法 |
|--------|------|---------|
| `y_vs_all` | `analyze_y_vs_all` | RF重要性 + Pearson + Spearman + 单变量线性回归 |
| `two_column` | `analyze_two_column` | 双向线性回归 + Pearson + Spearman + 描述统计 |
| `multi_x_vs_y` | `analyze_multi_x_vs_y` | 多元线性回归 + RF重要性 + 多重共线性检测 |

## 新增分析模式一览

| 模式键 | 函数 | 核心算法 | 典型用户指令 |
|--------|------|---------|------------|
| `time_series` | `analyze_time_series` | 趋势检测 + 季节分解 + ADF平稳性 + ARIMA预测 | "分析销售额的时间趋势" |
| `pca` | `analyze_pca` | PCA主成分分析 + 方差贡献 + 载荷矩阵 | "对所有变量做降维分析" |
| `anova` | `analyze_anova` | 单因素ANOVA + Tukey HSD + 效应量(η²) | "比较各组别均值差异" |
| `logistic` | `analyze_logistic` | 逻辑回归 + 系数/Odds Ratio + ROC/AUC | "预测是否达标" |
| `cluster` | `analyze_cluster` | K-Means + 肘部法则 + 簇统计 | "对数据分组聚类" |
| `neural_reg` | `analyze_neural_reg` | MLP神经网络回归 + 与线性模型对比 | "用神经网络预测产量" |
| `ridge_lasso` | `analyze_ridge_lasso` | 岭/套索回归 + 交叉验证 + 系数路径 | "做岭回归/正则化回归" |

---

## Task 1：新增依赖 statsmodels

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

**Step 1：在 requirements.txt 末尾追加**

```
statsmodels>=0.14.0
```

**Step 2：更新 Dockerfile 构建时间戳（强制重建镜像）**

将 `Dockerfile` 末尾注释 `# build: 2026-03-12c` 改为 `# build: 2026-03-12d`

**Step 3：本地验证安装**

```bash
pip install statsmodels>=0.14.0
python -c "import statsmodels; print(statsmodels.__version__)"
# 预期输出：0.14.x
```

**Step 4：提交**

```bash
git add requirements.txt Dockerfile
git commit -m "chore: add statsmodels for time-series analysis"
```

---

## Task 2：扩展 AnalysisResult dataclass

**Files:**
- Modify: `core/analysis_engine.py`（只改 dataclass 部分，不改现有函数）

**Step 1：在 `AnalysisResult` dataclass 末尾追加新模式的字段**

在 `x_corr_matrix: Optional[pd.DataFrame] = None` 一行之后追加：

```python
    # ── 时序分析 (time_series) ──────────────────────────
    time_col: str = ""
    value_col: str = ""
    ts_trend_slope: float = 0.0           # 趋势斜率（线性）
    ts_trend_r2: float = 0.0              # 趋势线 R²
    ts_adf_stat: float = 0.0              # ADF 检验统计量
    ts_adf_pvalue: float = 1.0            # ADF p 值（<0.05 则平稳）
    ts_decompose_df: Optional[pd.DataFrame] = None   # trend/seasonal/residual
    ts_forecast_df: Optional[pd.DataFrame] = None    # 预测值（若 ARIMA 成功）
    ts_arima_order: tuple = ()             # (p,d,q)

    # ── PCA 分析 ────────────────────────────────────────
    pca_n_components: int = 0
    pca_explained_ratio: list[float] = field(default_factory=list)  # 每个主成分方差贡献率
    pca_cumulative_ratio: list[float] = field(default_factory=list)  # 累计方差贡献率
    pca_loadings_df: Optional[pd.DataFrame] = None  # 载荷矩阵（列=主成分，行=原始变量）
    pca_scores_df: Optional[pd.DataFrame] = None    # 主成分得分

    # ── ANOVA 分析 ──────────────────────────────────────
    anova_group_col: str = ""
    anova_f_stat: float = 0.0
    anova_p_value: float = 1.0
    anova_eta_squared: float = 0.0        # 效应量 η²
    anova_group_stats_df: Optional[pd.DataFrame] = None  # 各组 n/mean/std
    anova_tukey_df: Optional[pd.DataFrame] = None        # Tukey HSD 两两比较

    # ── 逻辑回归 (logistic) ──────────────────────────────
    logistic_coef_df: Optional[pd.DataFrame] = None  # Feature / Coef / OddsRatio / p_value
    logistic_accuracy: float = 0.0
    logistic_auc: float = 0.0
    logistic_classes: list = field(default_factory=list)

    # ── 聚类分析 (cluster) ───────────────────────────────
    cluster_n: int = 0
    cluster_inertia_list: list[float] = field(default_factory=list)  # 肘部法则 SSE
    cluster_labels: Optional[pd.Series] = None       # 每行的簇标签
    cluster_centers_df: Optional[pd.DataFrame] = None  # 簇中心
    cluster_stats_df: Optional[pd.DataFrame] = None    # 各簇样本数/均值

    # ── 神经网络回归 (neural_reg) ────────────────────────
    neural_r2_train: float = 0.0
    neural_r2_test: float = 0.0
    neural_mae_test: float = 0.0
    neural_vs_linear_r2: float = 0.0   # 同数据线性回归 R²（对比用）
    neural_hidden_layers: tuple = ()

    # ── 岭/套索回归 (ridge_lasso) ────────────────────────
    ridge_lasso_type: str = ""          # "ridge" / "lasso" / "both"
    ridge_best_alpha: float = 0.0
    lasso_best_alpha: float = 0.0
    ridge_coef_df: Optional[pd.DataFrame] = None  # Feature / Ridge_Coef / Lasso_Coef / OLS_Coef
    ridge_r2: float = 0.0
    lasso_r2: float = 0.0
    lasso_selected_features: list[str] = field(default_factory=list)  # Lasso 保留的非零特征
```

**Step 2：验证 dataclass 可正常实例化**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from core.analysis_engine import AnalysisResult
r = AnalysisResult(mode='time_series', target_y='y', x_columns=[], valid_row_count=10, raw_row_count=10)
print('AnalysisResult 扩展 OK, pca_n_components =', r.pca_n_components)
"
# 预期：AnalysisResult 扩展 OK, pca_n_components = 0
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: extend AnalysisResult with fields for 7 new analysis modes"
```

---

## Task 3：实现 analyze_time_series

**Files:**
- Modify: `core/analysis_engine.py`（在文件末尾追加函数）

**Step 1：在 `analyze_multi_x_vs_y` 函数之后追加**

```python
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
```

**Step 2：验证函数可被导入**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_time_series
df = pd.DataFrame({'t': range(50), 'v': np.random.randn(50).cumsum()})
r = analyze_time_series(df, 't', 'v')
print('mode:', r.mode, '| trend_slope:', round(r.ts_trend_slope, 4), '| adf_p:', round(r.ts_adf_pvalue, 4))
"
# 预期：mode: time_series | trend_slope: ...（不报错即可）
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_time_series (trend + ADF + ARIMA)"
```

---

## Task 4：实现 analyze_pca

**Files:**
- Modify: `core/analysis_engine.py`（在文件末尾追加）

**Step 1：追加函数**

```python
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
```

**Step 2：验证**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_pca
df = pd.DataFrame(np.random.randn(100, 5), columns=['A','B','C','D','E'])
r = analyze_pca(df, ['A','B','C','D','E'])
print('mode:', r.mode, '| n_components:', r.pca_n_components, '| cumulative:', [round(x,3) for x in r.pca_cumulative_ratio])
"
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_pca with auto component selection"
```

---

## Task 5：实现 analyze_anova

**Files:**
- Modify: `core/analysis_engine.py`

**Step 1：追加函数**

```python
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
```

**Step 2：验证**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_anova
df = pd.DataFrame({'group': ['A']*30+['B']*30+['C']*30, 'value': np.concatenate([np.random.randn(30)+0, np.random.randn(30)+2, np.random.randn(30)+4])})
r = analyze_anova(df, 'value', 'group')
print('mode:', r.mode, '| F:', round(r.anova_f_stat,3), '| p:', round(r.anova_p_value,4), '| eta2:', round(r.anova_eta_squared,3))
"
# 预期：F 较大，p 接近 0
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_anova with Tukey HSD post-hoc"
```

---

## Task 6：实现 analyze_logistic

**Files:**
- Modify: `core/analysis_engine.py`

**Step 1：追加函数**

```python
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
```

**Step 2：验证**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_logistic
np.random.seed(42)
df = pd.DataFrame({'x1': np.random.randn(100), 'x2': np.random.randn(100)})
df['y'] = (df['x1'] + df['x2'] + np.random.randn(100)*0.5 > 0).astype(int)
r = analyze_logistic(df, 'y', ['x1','x2'])
print('mode:', r.mode, '| accuracy:', round(r.logistic_accuracy,3), '| auc:', round(r.logistic_auc,3))
"
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_logistic with AUC-ROC"
```

---

## Task 7：实现 analyze_cluster

**Files:**
- Modify: `core/analysis_engine.py`

**Step 1：追加函数**

```python
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
```

**Step 2：验证**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_cluster
df = pd.DataFrame({'x': np.r_[np.random.randn(50)+5, np.random.randn(50)-5], 'y': np.r_[np.random.randn(50)+5, np.random.randn(50)-5]})
r = analyze_cluster(df, ['x','y'])
print('mode:', r.mode, '| k:', r.cluster_n, '| cluster sizes:', r.cluster_stats_df['样本量'].tolist())
"
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_cluster with elbow method auto-K"
```

---

## Task 8：实现 analyze_neural_reg

**Files:**
- Modify: `core/analysis_engine.py`

**Step 1：追加函数**

```python
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
```

**Step 2：验证**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_neural_reg
np.random.seed(0)
df = pd.DataFrame({'x1': np.random.randn(200), 'x2': np.random.randn(200)})
df['y'] = df['x1']**2 + df['x2'] + np.random.randn(200)*0.3
r = analyze_neural_reg(df, 'y', ['x1','x2'])
print('mode:', r.mode, '| mlp_r2:', round(r.neural_r2_test,3), '| linear_r2:', round(r.neural_vs_linear_r2,3))
"
# 预期：mlp_r2 应明显高于 linear_r2（因为数据含非线性 x1^2）
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_neural_reg with linear regression baseline"
```

---

## Task 9：实现 analyze_ridge_lasso

**Files:**
- Modify: `core/analysis_engine.py`

**Step 1：追加函数**

```python
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
```

**Step 2：验证**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_ridge_lasso
np.random.seed(1)
df = pd.DataFrame(np.random.randn(100, 5), columns=['x1','x2','x3','x4','x5'])
df['y'] = 2*df['x1'] - df['x3'] + np.random.randn(100)*0.5
r = analyze_ridge_lasso(df, 'y', ['x1','x2','x3','x4','x5'])
print('mode:', r.mode, '| ridge_r2:', round(r.ridge_r2,3), '| lasso_selected:', r.lasso_selected_features)
"
# 预期：lasso_selected 应包含 x1, x3（其他特征系数为0）
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_ridge_lasso with cross-validated alpha"
```

---

## Task 10：更新 result_formatter.py

**Files:**
- Modify: `core/result_formatter.py`

**Step 1：在 `detect_focus` 函数中追加新模式的 elif 分支**

在 `elif mode == "multi_x_vs_y":` 块之后追加：

```python
    elif mode == "time_series":
        return "default"

    elif mode == "pca":
        return "default"

    elif mode == "anova":
        if any(k in inst for k in ["tukey", "事后检验", "两两比较", "post-hoc"]):
            return "tukey_detail"
        return "default"

    elif mode in ("logistic", "cluster", "neural_reg", "ridge_lasso"):
        return "default"
```

**Step 2：在 `format_result` 函数中追加新模式的 formatters 映射**

在 `"multi_x_vs_y": { ... }` 后追加：

```python
        "time_series": {
            "default": _fmt_time_series,
        },
        "pca": {
            "default": _fmt_pca,
        },
        "anova": {
            "tukey_detail": _fmt_anova_tukey,
            "default":      _fmt_anova,
        },
        "logistic": {
            "default": _fmt_logistic,
        },
        "cluster": {
            "default": _fmt_cluster,
        },
        "neural_reg": {
            "default": _fmt_neural_reg,
        },
        "ridge_lasso": {
            "default": _fmt_ridge_lasso,
        },
```

**Step 3：在文件末尾追加 7 个格式化函数**

（详见下方代码块，每个函数约 20-40 行，基于 analysis 字段输出 Markdown 表格）

```python
# ──────────────────────────────────────────────────────────
# 新模式格式化器
# ──────────────────────────────────────────────────────────

def _fmt_time_series(a: "AnalysisResult") -> str:
    """时间序列分析结果格式化"""
    return a.summary_text  # summary_text 已在 analyze_time_series 中构建完整


def _fmt_pca(a: "AnalysisResult") -> str:
    """PCA 分析结果格式化"""
    return a.summary_text


def _fmt_anova(a: "AnalysisResult") -> str:
    """ANOVA 完整结果（含分组统计）"""
    if a.anova_group_stats_df is None:
        return a.summary_text
    lines = [a.summary_text, "", "### 各组描述统计"]
    if a.anova_group_stats_df is not None:
        cols = a.anova_group_stats_df.columns.tolist()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "------|" * len(cols))
        for _, row in a.anova_group_stats_df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt_anova_tukey(a: "AnalysisResult") -> str:
    """ANOVA + Tukey HSD 事后检验细节"""
    base = _fmt_anova(a)
    if a.anova_tukey_df is None:
        return base + "\n\n> ⚠️ Tukey HSD 不可用（statsmodels 未安装）"
    lines = [base, "", "### Tukey HSD 两两比较（p<0.05 = 显著差异）"]
    cols = a.anova_tukey_df.columns.tolist()
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "------|" * len(cols))
    for _, row in a.anova_tukey_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt_logistic(a: "AnalysisResult") -> str:
    """逻辑回归结果格式化"""
    if a.logistic_coef_df is None:
        return a.summary_text
    lines = [
        a.summary_text, "",
        "### 回归系数与 Odds Ratio",
        "| 特征 | 系数 | Odds Ratio | 影响方向 |",
        "|------|------|-----------|---------|",
    ]
    for _, row in a.logistic_coef_df.iterrows():
        d = "↑ 增加正类概率" if row["Coefficient"] > 0 else "↓ 降低正类概率"
        lines.append(f"| `{row['Feature']}` | {row['Coefficient']:+.4f} | {row['OddsRatio']:.4f} | {d} |")
    return "\n".join(lines)


def _fmt_cluster(a: "AnalysisResult") -> str:
    """聚类分析结果格式化"""
    if a.cluster_stats_df is None:
        return a.summary_text
    lines = [a.summary_text, "", "### 各簇统计（均值）"]
    cols = a.cluster_stats_df.columns.tolist()
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "------|" * len(cols))
    for _, row in a.cluster_stats_df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt_neural_reg(a: "AnalysisResult") -> str:
    """神经网络回归结果格式化"""
    return a.summary_text


def _fmt_ridge_lasso(a: "AnalysisResult") -> str:
    """岭/套索回归系数对比表格"""
    if a.ridge_coef_df is None:
        return a.summary_text
    lines = [
        a.summary_text, "",
        "### 系数对比（OLS vs Ridge vs Lasso）",
        "| 特征 | OLS 系数 | Ridge 系数 | Lasso 系数 |",
        "|------|---------|-----------|----------|",
    ]
    for _, row in a.ridge_coef_df.iterrows():
        lasso_val = f"{row['Lasso_Coef']:+.4f}" if abs(row["Lasso_Coef"]) > 1e-6 else "**0（压缩）**"
        lines.append(f"| `{row['Feature']}` | {row['OLS_Coef']:+.4f} | {row['Ridge_Coef']:+.4f} | {lasso_val} |")
    return "\n".join(lines)
```

**Step 4：验证格式化器可导入**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from core.result_formatter import format_result, detect_focus
print('result_formatter import OK')
print('time_series focus:', detect_focus('分析时间趋势', 'time_series'))
print('pca focus:', detect_focus('降维分析', 'pca'))
"
# 预期：两行 focus 输出均为 default，不报错
```

**Step 5：提交**

```bash
git add core/result_formatter.py
git commit -m "feat: add result formatters for 7 new analysis modes"
```

---

## Task 11：更新 NLP Parser 提示词

**Files:**
- Modify: `prompts/intent_parser.txt`

**Step 1：读取当前提示词，在分析模式列表部分追加新模式**

在现有 3 种模式描述之后追加：

```
4. time_series：用于时间序列分析，当用户提到时间趋势、季节性、预测、走势、周期时使用。需要一个时间列（target_y）和一个数值列（x_columns 第一个元素）。
5. pca：用于主成分分析/降维，当用户提到 PCA、降维、主成分、factor analysis 时使用。x_columns 为参与分析的列，target_y 可为空。
6. anova：用于组间差异分析，当用户提到方差分析、ANOVA、组间差异、各组比较时使用。target_y 为数值列，x_columns 第一个元素为分组变量列。
7. logistic：用于分类预测，当用户提到逻辑回归、二元分类、0/1预测、是否/达标 时使用。target_y 为二元目标列。
8. cluster：用于聚类/分组发现，当用户提到 K-Means、聚类、分群、客户分群 时使用。x_columns 为参与聚类的列，target_y 可为空。
9. neural_reg：用于神经网络回归，当用户提到神经网络、深度学习、MLP、非线性回归 时使用。
10. ridge_lasso：用于正则化回归，当用户提到岭回归、套索、Lasso、Ridge、正则化 时使用。
```

**Step 2：验证 NLP parser 可被导入（不实际调用 API）**

```bash
python -c "
import sys; sys.path.insert(0, '.')
# 只测试提示词文件是否存在且可读
from pathlib import Path
txt = Path('prompts/intent_parser.txt').read_text(encoding='utf-8')
assert 'time_series' in txt or 'pca' in txt, '提示词未更新'
print('提示词包含新模式描述，OK')
"
```

**Step 3：更新 routes.py 中的关键词覆盖逻辑**

在 `api/routes.py` 的关键词覆盖段（约第 210-214 行）中，在 `_multi_kw` 列表后追加时序、PCA 等关键词的强制切换逻辑：

```python
_ts_kw = ["时间趋势", "时序", "走势", "预测未来", "季节", "arima", "trend"]
if mode == "y_vs_all" and any(k in inst_lower for k in _ts_kw):
    mode = "time_series"

_pca_kw = ["pca", "降维", "主成分", "factor"]
if mode == "y_vs_all" and any(k in inst_lower for k in _pca_kw):
    mode = "pca"
    x_cols = []  # 自动取所有数值列

_anova_kw = ["方差分析", "anova", "组间", "各组差异", "组别比较"]
if mode == "y_vs_all" and any(k in inst_lower for k in _anova_kw):
    mode = "anova"

_cluster_kw = ["聚类", "k-means", "kmeans", "分群", "分组发现"]
if mode == "y_vs_all" and any(k in inst_lower for k in _cluster_kw):
    mode = "cluster"
    x_cols = []

_neural_kw = ["神经网络", "mlp", "深度学习", "neural", "非线性回归"]
if mode == "y_vs_all" and any(k in inst_lower for k in _neural_kw):
    mode = "neural_reg"

_ridge_kw = ["岭回归", "套索", "lasso", "ridge", "正则化"]
if mode == "y_vs_all" and any(k in inst_lower for k in _ridge_kw):
    mode = "ridge_lasso"
```

**Step 4：提交**

```bash
git add prompts/intent_parser.txt api/routes.py
git commit -m "feat: extend NLP parser prompt and keyword routing for 7 new modes"
```

---

## Task 12：更新 api/routes.py 路由分发

**Files:**
- Modify: `api/routes.py`

**Step 1：在 `api_analyze` 函数中的"执行分析"部分（if/elif 链）扩展**

将现有的 `if mode == "y_vs_all": ... elif mode == "two_column": ... else:` 扩展为：

```python
if mode == "y_vs_all":
    analysis = analyze_y_vs_all(clean_df, target_y)
elif mode == "two_column":
    analysis = analyze_two_column(clean_df, x_cols[0], target_y)
elif mode == "multi_x_vs_y":
    valid_x = [c for c in x_cols if c in clean_df.columns]
    if not valid_x:
        return JSONResponse({"success": False, "error": "所有自变量列在清洗后均不存在"})
    analysis = analyze_multi_x_vs_y(clean_df, target_y, valid_x)
elif mode == "time_series":
    time_c = x_cols[0] if x_cols else None
    if not time_c:
        return JSONResponse({"success": False, "error": "时序分析需要指定时间列（x_columns）"})
    from core.analysis_engine import analyze_time_series
    analysis = analyze_time_series(clean_df, time_c, target_y)
elif mode == "pca":
    cols_for_pca = x_cols if x_cols else [c for c in numeric_cols if c != target_y]
    if len(cols_for_pca) < 2:
        return JSONResponse({"success": False, "error": "PCA 至少需要 2 个变量"})
    from core.analysis_engine import analyze_pca
    analysis = analyze_pca(clean_df, cols_for_pca)
elif mode == "anova":
    group_c = x_cols[0] if x_cols else None
    if not group_c:
        return JSONResponse({"success": False, "error": "ANOVA 需要指定分组列（x_columns）"})
    from core.analysis_engine import analyze_anova
    analysis = analyze_anova(clean_df, target_y, group_c)
elif mode == "logistic":
    valid_x = [c for c in x_cols if c in clean_df.columns]
    if not valid_x:
        valid_x = [c for c in numeric_cols if c != target_y]
    from core.analysis_engine import analyze_logistic
    analysis = analyze_logistic(clean_df, target_y, valid_x)
elif mode == "cluster":
    cols_for_cluster = x_cols if x_cols else [c for c in numeric_cols if c != target_y]
    if len(cols_for_cluster) < 2:
        return JSONResponse({"success": False, "error": "聚类至少需要 2 个变量"})
    from core.analysis_engine import analyze_cluster
    analysis = analyze_cluster(clean_df, cols_for_cluster)
elif mode == "neural_reg":
    valid_x = [c for c in x_cols if c in clean_df.columns]
    if not valid_x:
        valid_x = [c for c in numeric_cols if c != target_y]
    from core.analysis_engine import analyze_neural_reg
    analysis = analyze_neural_reg(clean_df, target_y, valid_x)
elif mode == "ridge_lasso":
    valid_x = [c for c in x_cols if c in clean_df.columns]
    if not valid_x:
        valid_x = [c for c in numeric_cols if c != target_y]
    from core.analysis_engine import analyze_ridge_lasso
    analysis = analyze_ridge_lasso(clean_df, target_y, valid_x)
else:
    return JSONResponse({"success": False, "error": f"未知分析模式：{mode}"})
```

**Step 2：扩展 `_build_table_data` 函数，处理新模式**

在现有三模式的 `elif` 链末尾追加：

```python
elif analysis.mode == "pca" and analysis.pca_loadings_df is not None:
    for i, (pc, ratio) in enumerate(zip(
        [f"PC{j+1}" for j in range(analysis.pca_n_components)],
        analysis.pca_explained_ratio
    )):
        rows.append({"主成分": pc, "方差贡献率": f"{ratio:.1%}", "累计贡献率": f"{analysis.pca_cumulative_ratio[i]:.1%}"})
elif analysis.mode == "anova" and analysis.anova_group_stats_df is not None:
    for _, r in analysis.anova_group_stats_df.iterrows():
        rows.append({"组别": str(r["组别"]), "样本量": r["样本量"], "均值": r["均值"]})
elif analysis.mode == "logistic" and analysis.logistic_coef_df is not None:
    for _, r in analysis.logistic_coef_df.iterrows():
        rows.append({"特征": r["Feature"], "系数": f"{r['Coefficient']:+.4f}", "Odds Ratio": f"{r['OddsRatio']:.4f}"})
elif analysis.mode == "cluster" and analysis.cluster_stats_df is not None:
    for _, r in analysis.cluster_stats_df.iterrows():
        rows.append(dict(r))
elif analysis.mode in ("neural_reg", "ridge_lasso", "time_series"):
    pass  # summary_text 已包含核心数据
```

**Step 3：更新 `AnalyzeRequest` 的 `manual_mode` 字段注释（不改验证逻辑）**

将 `manual_mode: str = "y_vs_all"` 改为（字段文档说明）：

```python
manual_mode: str = "y_vs_all"  # y_vs_all/two_column/multi_x_vs_y/time_series/pca/anova/logistic/cluster/neural_reg/ridge_lasso
```

**Step 4：验证服务启动不报错**

```bash
python app.py &
sleep 3
curl -s http://localhost:7860/api/history | head -c 100
# 预期：返回 [] 或历史列表
kill %1
```

**Step 5：提交**

```bash
git add api/routes.py
git commit -m "feat: route 7 new analysis modes in api/routes.py"
```

---

## Task 13：更新前端 analyze.html（手动模式下拉框）

**Files:**
- Modify: `templates/analyze.html`

**Step 1：在手动分析模式的 `<select>` 或 Alpine.js 数据中添加新选项**

找到现有的三个模式选项（y_vs_all / two_column / multi_x_vs_y），在其后追加：

```html
<option value="time_series">📈 时间序列分析（趋势+ADF+ARIMA）</option>
<option value="pca">🔬 PCA 主成分降维</option>
<option value="anova">📊 ANOVA 方差分析（组间差异）</option>
<option value="logistic">🎯 逻辑回归（二元分类）</option>
<option value="cluster">🔵 K-Means 聚类分析</option>
<option value="neural_reg">🧠 神经网络回归（MLP）</option>
<option value="ridge_lasso">🔧 岭回归 / 套索回归</option>
```

**Step 2：验证页面不报 JS 错误**

启动服务后访问 `http://localhost:7860/analyze`，打开浏览器控制台确认无报错，下拉框中出现新选项。

**Step 3：提交**

```bash
git add templates/analyze.html
git commit -m "feat: add 7 new analysis modes to manual mode dropdown"
```

---

## Task 14：端到端集成测试

**Step 1：准备测试数据**

创建 `tests/test_data_generator.py`（或直接用 Python 命令行）：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100
df = pd.DataFrame({
    "时间序号": range(n),
    "销售额":   np.cumsum(np.random.randn(n)) + 100,
    "温度":     np.random.randn(n) * 10 + 25,
    "湿度":     np.random.randn(n) * 5 + 60,
    "是否达标": (np.random.randn(n) > 0).astype(int),
    "分类":     np.random.choice(["A","B","C"], n),
})
df.to_excel("tests/integration_test.xlsx", index=False)
print("测试数据已生成：tests/integration_test.xlsx")
```

**Step 2：运行集成测试**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from utils.data_loader import load_excel, get_numeric_columns, preprocess_for_analysis
from core.analysis_engine import (
    analyze_time_series, analyze_pca, analyze_anova,
    analyze_logistic, analyze_cluster, analyze_neural_reg, analyze_ridge_lasso
)

df, _ = load_excel('tests/integration_test.xlsx')
numeric_cols = get_numeric_columns(df)
print('数值列:', numeric_cols)

# 时序
clean, _, _ = preprocess_for_analysis(df, ['时间序号', '销售额'])
r = analyze_time_series(clean, '时间序号', '销售额')
print('time_series OK | trend_slope:', round(r.ts_trend_slope, 3))

# PCA
clean, _, _ = preprocess_for_analysis(df, ['销售额','温度','湿度'])
r = analyze_pca(clean, ['销售额','温度','湿度'])
print('pca OK | n_components:', r.pca_n_components)

# ANOVA
clean, _, _ = preprocess_for_analysis(df, ['销售额'])
clean['分类'] = df['分类'].values[:len(clean)]
r = analyze_anova(clean, '销售额', '分类')
print('anova OK | F:', round(r.anova_f_stat, 3), '| p:', round(r.anova_p_value, 3))

# 逻辑回归
clean, _, _ = preprocess_for_analysis(df, ['温度', '湿度', '是否达标'])
r = analyze_logistic(clean, '是否达标', ['温度', '湿度'])
print('logistic OK | accuracy:', round(r.logistic_accuracy, 3))

# 聚类
clean, _, _ = preprocess_for_analysis(df, ['温度','湿度'])
r = analyze_cluster(clean, ['温度','湿度'])
print('cluster OK | k:', r.cluster_n)

# 神经网络
clean, _, _ = preprocess_for_analysis(df, ['温度','湿度','销售额'])
r = analyze_neural_reg(clean, '销售额', ['温度','湿度'])
print('neural_reg OK | r2_test:', round(r.neural_r2_test, 3))

# 岭/套索
r = analyze_ridge_lasso(clean, '销售额', ['温度','湿度'])
print('ridge_lasso OK | ridge_r2:', round(r.ridge_r2, 3), '| lasso_selected:', r.lasso_selected_features)

print()
print('=== 所有 7 种新模式集成测试通过 ===')
"
```

**预期输出：** 每行均有 `OK`，最后一行打印 `=== 所有 7 种新模式集成测试通过 ===`

**Step 3：提交**

```bash
git add tests/
git commit -m "test: add integration test data and validation for 7 new analysis modes"
```

---

## Task 15：推送到 GitHub/ModelScope

**Step 1：确认 .env 不在追踪列表**

```bash
git status
# 确认没有 .env、uploads/、*.db 出现
```

**Step 2：推送**

```bash
git push origin master
```

**Step 3：观察 GitHub Actions**

进入 GitHub 仓库 → Actions，确认 `Sync to ModelScope Studio` 工作流成功。

**Step 4：验证 ModelScope**

访问创空间，上传测试文件，测试以下指令：
- "分析销售额的时间趋势" → 应触发 `time_series` 模式
- "对所有变量做 PCA 降维" → 应触发 `pca` 模式
- "用神经网络预测销售额" → 应触发 `neural_reg` 模式
- "做岭回归分析" → 应触发 `ridge_lasso` 模式

---

---

## Task 16：新增 model_comparison 模式核心数据结构

**背景：** 在发现 Top 3-5 影响因素后，自动对比多个回归/分类模型，找出最优模型，并在 Excel 报告中为每个模型单独生成一个 sheet，绘制预测值 vs 实际值散点图。

**模型赛马逻辑：**
1. 使用 5 折交叉验证评估每个模型
2. 自动识别变量类型（连续 / 分类），决定是否编码
3. 按 R²（越高越好）选出冠军模型
4. 每个模型独立一张 Excel Sheet，展示预测 vs 实际散点图 + 残差折线图

**参赛模型（回归任务）：**

| 模型名 | 类 | 特点 |
|--------|-----|------|
| 线性回归 | `LinearRegression` | 基准，可解释性强 |
| 岭回归 | `RidgeCV` | 解决共线性 |
| 随机森林 | `RandomForestRegressor` | 非线性，稳健 |
| 梯度提升 | `GradientBoostingRegressor` | 高精度 |
| 神经网络 | `MLPRegressor` | 深度非线性 |

**Files:**
- Modify: `core/analysis_engine.py`（扩展 AnalysisResult + 新增函数）

**Step 1：在 AnalysisResult dataclass 末尾追加 model_comparison 字段**

```python
    # ── 模型对比 (model_comparison) ─────────────────────────
    mc_x_columns: list[str] = field(default_factory=list)   # 参与建模的 X 列
    mc_comparison_df: Optional[pd.DataFrame] = None          # 模型对比汇总表
    mc_best_model_name: str = ""                              # 最优模型名称
    mc_best_model_r2: float = 0.0                            # 最优模型 R²
    mc_predictions: dict = field(default_factory=dict)        # {模型名: {"actual": [...], "predicted": [...], "residual": [...]}}
    mc_feature_types: dict = field(default_factory=dict)      # {列名: "continuous" / "categorical"}
    mc_cv_scores: dict = field(default_factory=dict)          # {模型名: [cv_r2_fold1, fold2, ...]}
```

**Step 2：验证 dataclass 扩展后可正常实例化**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from core.analysis_engine import AnalysisResult
r = AnalysisResult(mode='model_comparison', target_y='y', x_columns=[], valid_row_count=10, raw_row_count=10)
print('mc_best_model_name:', repr(r.mc_best_model_name), '| mc_predictions:', r.mc_predictions)
"
# 预期：mc_best_model_name: '' | mc_predictions: {}
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: extend AnalysisResult with model_comparison fields"
```

---

## Task 17：实现 analyze_model_comparison 函数

**Files:**
- Modify: `core/analysis_engine.py`（在文件末尾追加）

**Step 1：追加辅助函数 `_detect_feature_types` 和主函数 `analyze_model_comparison`**

```python
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
    多模型赛马：对 top_n 个 X 列自动训练 5 个回归模型，5 折 CV 评估，
    找出最优模型，并为每个模型保存预测值 vs 实际值数据（供报告绘图）。

    x_cols 建议传入 y_vs_all / multi_x_vs_y 识别出的 Top 3-5 影响因素。
    若 x_cols 为空，则自动取 target_y 以外的所有数值列（最多 top_n 个）。
    """
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import warnings

    # ── 变量类型检测 ──
    feature_types = _detect_feature_types(df, x_cols)

    # ── 预处理：分类列标签编码，连续列保留 ──
    X_raw = df[x_cols].copy()
    for col, ftype in feature_types.items():
        if ftype == "categorical":
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    X = X_raw.values.astype(float)
    y = df[target_y].values.astype(float)
    n = len(y)

    # ── 标准化（对线性/岭/神经网络有效，RF/GBR 无影响） ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 候选模型定义 ──
    models = {
        "线性回归":   LinearRegression(),
        "岭回归":     RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        "随机森林":   RandomForestRegressor(n_estimators=100, random_state=RF_RANDOM_STATE, n_jobs=1),
        "梯度提升":   GradientBoostingRegressor(n_estimators=100, random_state=RF_RANDOM_STATE),
        "神经网络MLP": MLPRegressor(
            hidden_layer_sizes=(min(64, len(x_cols)*8), min(32, len(x_cols)*4)),
            max_iter=500, random_state=RF_RANDOM_STATE, early_stopping=True
        ),
    }

    kf = KFold(n_splits=min(5, n // 5), shuffle=True, random_state=42)

    comparison_rows = []
    predictions_store = {}
    cv_scores_store = {}

    for model_name, model in models.items():
        # 线性/岭/神经网络用标准化数据；RF/GBR 用原始数据
        use_scaled = model_name in ("线性回归", "岭回归", "神经网络MLP")
        X_use = X_scaled if use_scaled else X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_r2 = cross_val_score(model, X_use, y, cv=kf, scoring="r2")
            model.fit(X_use, y)

        y_pred = model.predict(X_use)
        r2 = float(r2_score(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        cv_mean = float(np.mean(cv_r2))
        cv_std = float(np.std(cv_r2))

        comparison_rows.append({
            "模型":         model_name,
            "训练集R²":    round(r2, 4),
            "CV均值R²":    round(cv_mean, 4),
            "CV标准差":    round(cv_std, 4),
            "RMSE":         round(rmse, 4),
            "MAE":          round(mae, 4),
        })

        residuals = y - y_pred
        predictions_store[model_name] = {
            "actual":    y.tolist(),
            "predicted": y_pred.tolist(),
            "residual":  residuals.tolist(),
        }
        cv_scores_store[model_name] = cv_r2.tolist()

    comparison_df = pd.DataFrame(comparison_rows).sort_values("CV均值R²", ascending=False)
    best_row = comparison_df.iloc[0]
    best_name = str(best_row["模型"])
    best_r2 = float(best_row["CV均值R²"])

    result = AnalysisResult(
        mode="model_comparison",
        target_y=target_y,
        x_columns=x_cols,
        valid_row_count=n,
        raw_row_count=n,
        mc_x_columns=x_cols,
        mc_comparison_df=comparison_df,
        mc_best_model_name=best_name,
        mc_best_model_r2=best_r2,
        mc_predictions=predictions_store,
        mc_feature_types=feature_types,
        mc_cv_scores=cv_scores_store,
    )

    # summary_text
    lines = [
        f"## 🏆 多模型对比分析（AutoML-lite）",
        f"**目标变量**：`{target_y}`  **输入特征**：{', '.join(f'`{c}`' for c in x_cols)}",
        f"**有效样本**：{n} 行  **评估方式**：{min(5, n//5)} 折交叉验证",
        f"",
        f"### 🥇 最优模型：{best_name}",
        f"- 交叉验证 R² = **{best_r2:.4f}**（越高越好）",
        f"",
        f"### 📊 模型对比排名",
        f"| 排名 | 模型 | CV R² | CV σ | RMSE | MAE |",
        f"|------|------|-------|------|------|-----|",
    ]
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        lines.append(
            f"| {medals[i]} | **{row['模型']}** | {row['CV均值R²']:.4f} | "
            f"±{row['CV标准差']:.4f} | {row['RMSE']:.4f} | {row['MAE']:.4f} |"
        )
    lines += [
        f"",
        f"> Excel 报告中每个模型均有独立 Sheet，包含预测值 vs 实际值散点图与残差图。",
    ]
    result.summary_text = "\n".join(lines)
    return result
```

**Step 2：验证函数**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_model_comparison
np.random.seed(42)
df = pd.DataFrame({'x1': np.random.randn(100), 'x2': np.random.randn(100)})
df['y'] = 2*df['x1'] - df['x2']**2 + np.random.randn(100)*0.3
r = analyze_model_comparison(df, 'y', ['x1','x2'])
print('mode:', r.mode)
print('best:', r.mc_best_model_name, '| CV R²:', round(r.mc_best_model_r2, 3))
print('models compared:', list(r.mc_predictions.keys()))
"
# 预期：5 个模型都在 mc_predictions 中，随机森林/梯度提升因非线性应排名靠前
```

**Step 3：提交**

```bash
git add core/analysis_engine.py
git commit -m "feat: implement analyze_model_comparison with 5-model cross-validated comparison"
```

---

## Task 18：为 model_comparison 生成多 Sheet Excel 报告

**Files:**
- Modify: `core/report_builder.py`

**Step 1：在 `build_report` 函数的 `if/elif` 链中追加 model_comparison 分支**

在 `elif result.mode == "multi_x_vs_y":` 之后追加：

```python
    elif result.mode == "model_comparison":
        _build_model_comparison(writer, workbook, result, fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct)
```

**Step 2：在文件末尾追加 `_build_model_comparison` 函数**

```python
# ──────────────────────────────────────────────────────────
# 模式11：模型对比报告（一模型一 Sheet）
# ──────────────────────────────────────────────────────────

def _build_model_comparison(writer, workbook, result: AnalysisResult,
                             fmt_title, fmt_header, fmt_center, fmt_num4, fmt_pct):
    """
    为模型对比结果生成多 Sheet Excel 报告：
    Sheet 1：模型对比汇总（排名表 + R² 柱状图）
    Sheet 2~N：每个模型独立 Sheet（预测值 vs 实际值散点图 + 残差折线图）
    """
    import numpy as np

    # ── 通用颜色定义 ──
    MODEL_COLORS = {
        "线性回归":    "#4472C4",
        "岭回归":      "#70AD47",
        "随机森林":    "#ED7D31",
        "梯度提升":    "#C00000",
        "神经网络MLP": "#7030A0",
    }
    fmt_best = workbook.add_format({"bold": True, "bg_color": "#FFD700", "align": "center", "border": 1})
    fmt_num2 = workbook.add_format({"num_format": "0.0000", "align": "center"})

    comp_df = result.mc_comparison_df
    target_y = result.target_y

    # ════════════════════════════════════════════════════════
    # Sheet 1：模型对比汇总
    # ════════════════════════════════════════════════════════
    sht = workbook.add_worksheet("📊 模型对比汇总")
    sht.set_column("A:A", 16)
    sht.set_column("B:G", 14)

    sht.write("A1", f"🏆 多模型对比分析 — {target_y}", fmt_title)
    sht.write("A2", f"特征：{', '.join(result.mc_x_columns)}  |  样本量：{result.valid_row_count} 行  |  5折交叉验证")

    # 表头
    headers = ["模型", "CV均值R²", "CV标准差", "训练集R²", "RMSE", "MAE"]
    for j, h in enumerate(headers):
        sht.write(3, j, h, fmt_header)

    # 数据行（第一行高亮为最优模型）
    for i, (_, row) in enumerate(comp_df.iterrows()):
        fmt_use = fmt_best if i == 0 else fmt_center
        fmt_use_n = fmt_best if i == 0 else fmt_num2
        sht.write(4 + i, 0, row["模型"], fmt_use)
        sht.write(4 + i, 1, row["CV均值R²"], fmt_use_n)
        sht.write(4 + i, 2, row["CV标准差"], fmt_use_n)
        sht.write(4 + i, 3, row["训练集R²"], fmt_use_n)
        sht.write(4 + i, 4, row["RMSE"], fmt_use_n)
        sht.write(4 + i, 5, row["MAE"], fmt_use_n)

    # 最优模型说明
    sht.write(4 + len(comp_df) + 1, 0,
              f"✅ 最优模型：{result.mc_best_model_name}（CV R² = {result.mc_best_model_r2:.4f}）",
              workbook.add_format({"bold": True, "font_color": "#C00000"}))

    # R² 对比柱状图（使用汇总表数据）
    # 把 comp_df 写到隐藏 sheet 供图表引用
    comp_df.to_excel(writer, sheet_name="_mc_data", index=False)
    writer.sheets["_mc_data"].hide()

    n_models = len(comp_df)
    chart_r2 = workbook.add_chart({"type": "bar"})
    chart_r2.add_series({
        "name":       "CV均值R²",
        "categories": ["_mc_data", 1, 0, n_models, 0],
        "values":     ["_mc_data", 1, 1, n_models, 1],
        "data_labels": {"value": True, "num_format": "0.000"},
        "fill":       {"color": "#4472C4"},
    })
    chart_r2.set_title({"name": "各模型交叉验证 R²（越高越好）"})
    chart_r2.set_x_axis({"name": "R²"})
    chart_r2.set_y_axis({"name": "模型", "reverse": True})
    chart_r2.set_legend({"none": True})
    chart_r2.set_size({"width": 500, "height": 300})
    sht.insert_chart("H4", chart_r2)

    # ════════════════════════════════════════════════════════
    # Sheet 2~N：每个模型的预测 vs 实际详情
    # ════════════════════════════════════════════════════════
    for model_name, preds in result.mc_predictions.items():
        actual    = preds["actual"]
        predicted = preds["predicted"]
        residual  = preds["residual"]
        n = len(actual)
        color = MODEL_COLORS.get(model_name, "#4472C4")

        # 找到该模型的指标
        model_row = comp_df[comp_df["模型"] == model_name].iloc[0]
        cv_r2 = model_row["CV均值R²"]
        rmse  = model_row["RMSE"]
        mae   = model_row["MAE"]
        is_best = model_name == result.mc_best_model_name

        # Sheet 名（Excel sheet 名不超过 31 字符）
        sheet_name = f"{'🥇' if is_best else '📈'}{model_name}"[:31]
        data_sheet = f"_data_{model_name}"[:31]

        # 写数据到隐藏 sheet
        pred_df = pd.DataFrame({
            "实际值": actual,
            "预测值": predicted,
            "残差":   residual,
            "绝对误差": [abs(r) for r in residual],
        })
        pred_df.to_excel(writer, sheet_name=data_sheet, index=True)
        writer.sheets[data_sheet].hide()

        # 模型结果 sheet
        msht = workbook.add_worksheet(sheet_name)
        msht.set_column("A:A", 18)
        msht.set_column("B:E", 14)

        title_fmt = workbook.add_format({
            "bold": True, "font_size": 14,
            "font_color": "#C00000" if is_best else "#2F5496",
        })
        msht.write("A1", f"{'🥇 最优模型 ' if is_best else ''}『{model_name}』预测分析 — {target_y}", title_fmt)
        msht.write("A2", f"CV R² = {cv_r2:.4f}  |  RMSE = {rmse:.4f}  |  MAE = {mae:.4f}  |  样本量 = {n}")

        # 小型指标表
        msht.write(3, 0, "指标", fmt_header)
        msht.write(3, 1, "值", fmt_header)
        for i, (k, v) in enumerate([
            ("CV均值R²", f"{cv_r2:.4f}"),
            ("RMSE（均方根误差）", f"{rmse:.4f}"),
            ("MAE（平均绝对误差）", f"{mae:.4f}"),
            ("样本量", str(n)),
        ]):
            msht.write(4 + i, 0, k, fmt_center)
            msht.write(4 + i, 1, v, fmt_center)

        # ── 散点图：预测值 vs 实际值 ──
        # 每个点写入 sheet（XlsxWriter 散点图需要连续单元格数据）
        # 数据已在 data_sheet 的 B列(实际值) C列(预测值)，行从 index=1 开始
        # data_sheet 列布局：index(A), 实际值(B), 预测值(C), 残差(D), 绝对误差(E)
        # XlsxWriter 使用 1-based 列索引

        chart_scatter = workbook.add_chart({"type": "scatter"})
        chart_scatter.add_series({
            "name":       "预测 vs 实际",
            "categories": [data_sheet, 1, 1, n, 1],   # 实际值（X轴）
            "values":     [data_sheet, 1, 2, n, 2],   # 预测值（Y轴）
            "marker": {
                "type": "circle",
                "size": 5,
                "fill":   {"color": color},
                "border": {"color": color},
            },
            "trendline": {
                "type":           "linear",
                "name":           "拟合线",
                "line":           {"color": "#C00000", "width": 1.5, "dash_type": "dash"},
                "display_equation": False,
                "display_r_squared": False,
            },
        })
        chart_scatter.set_title({"name": f"预测值 vs 实际值（{model_name}）"})
        chart_scatter.set_x_axis({"name": f"实际值"})
        chart_scatter.set_y_axis({"name": "预测值"})
        chart_scatter.set_legend({"none": True})
        chart_scatter.set_size({"width": 450, "height": 320})
        msht.insert_chart("G2", chart_scatter)

        # ── 残差折线图 ──
        chart_residual = workbook.add_chart({"type": "line"})
        chart_residual.add_series({
            "name":   "残差",
            "values": [data_sheet, 1, 3, n, 3],       # 残差列（D列，index=3）
            "line":   {"color": color, "width": 1},
            "marker": {"type": "none"},
        })
        chart_residual.set_title({"name": f"残差分布（{model_name}）"})
        chart_residual.set_x_axis({"name": "样本序号"})
        chart_residual.set_y_axis({"name": "残差（实际 - 预测）"})
        chart_residual.set_legend({"none": True})
        chart_residual.set_size({"width": 450, "height": 200})
        msht.insert_chart("G22", chart_residual)
```

**Step 3：验证报告生成不报错**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_model_comparison
from core.report_builder import build_report

np.random.seed(42)
df = pd.DataFrame({'x1': np.random.randn(80), 'x2': np.random.randn(80)})
df['y'] = 2*df['x1'] - df['x2']**2 + np.random.randn(80)*0.3
r = analyze_model_comparison(df, 'y', ['x1', 'x2'])
out = build_report(r, 'tests/test_model_comparison.xlsx')
print('报告生成成功：', out)
import os; os.remove(str(out))
"
# 预期：reports/test_model_comparison.xlsx 生成后被删除，无报错
```

**Step 4：提交**

```bash
git add core/report_builder.py
git commit -m "feat: multi-sheet Excel report for model_comparison with scatter+residual charts"
```

---

## Task 19：result_formatter.py 追加 model_comparison 格式化器

**Files:**
- Modify: `core/result_formatter.py`

**Step 1：在 `detect_focus` 的新模式 elif 块中追加**

```python
    elif mode == "model_comparison":
        if any(k in inst for k in ["最优", "冠军", "最好", "best"]):
            return "best_only"
        return "default"
```

**Step 2：在 `format_result` 的 formatters 字典中追加**

```python
        "model_comparison": {
            "best_only": _fmt_mc_best_only,
            "default":   lambda a: a.summary_text,
        },
```

**Step 3：在文件末尾追加格式化函数**

```python
def _fmt_mc_best_only(a: "AnalysisResult") -> str:
    """只展示最优模型的结论"""
    if not a.mc_best_model_name:
        return a.summary_text
    row = a.mc_comparison_df[a.mc_comparison_df["模型"] == a.mc_best_model_name].iloc[0]
    lines = [
        f"## 🥇 最优模型：{a.mc_best_model_name}",
        f"**目标变量**：`{a.target_y}`  **特征**：{', '.join(f'`{c}`' for c in a.mc_x_columns)}",
        f"",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| CV均值R² | **{row['CV均值R²']:.4f}** |",
        f"| CV标准差 | ±{row['CV标准差']:.4f} |",
        f"| RMSE | {row['RMSE']:.4f} |",
        f"| MAE | {row['MAE']:.4f} |",
        f"",
        f"> Excel 报告中有各模型的预测 vs 实际值详情 Sheet。",
    ]
    return "\n".join(lines)
```

**Step 4：提交**

```bash
git add core/result_formatter.py
git commit -m "feat: add model_comparison result formatter"
```

---

## Task 20：更新 api/routes.py 路由分发

**Files:**
- Modify: `api/routes.py`

**Step 1：在顶部 import 区追加新函数**

在 `from core.analysis_engine import analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y` 一行改为：

```python
from core.analysis_engine import (
    analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y,
    analyze_model_comparison,
)
```

**Step 2：在执行分析的 if/elif 链末尾追加 model_comparison 分支**

（在 `elif mode == "ridge_lasso":` 块之后，`else:` 之前）

```python
elif mode == "model_comparison":
    # x_cols 优先来自 NLP 解析的 Top 因素列表
    # 若为空，取 multi_x_vs_y 最近一次结果的 Top 5（或全部数值列 Top 5）
    mc_x = [c for c in x_cols if c in numeric_cols and c != target_y]
    if not mc_x:
        mc_x = [c for c in numeric_cols if c != target_y][:5]
    if len(mc_x) < 1:
        return JSONResponse({"success": False, "error": "model_comparison 模式需要至少 1 个特征列"})
    analysis = analyze_model_comparison(clean_df, target_y, mc_x)
```

**Step 3：在 `_build_table_data` 末尾追加 model_comparison 表格**

```python
elif analysis.mode == "model_comparison" and analysis.mc_comparison_df is not None:
    for i, (_, row) in enumerate(analysis.mc_comparison_df.iterrows()):
        rows.append({
            "排名": i + 1,
            "模型": row["模型"],
            "CV R²": f"{row['CV均值R²']:.4f}",
            "RMSE": f"{row['RMSE']:.4f}",
            "MAE": f"{row['MAE']:.4f}",
        })
```

**Step 4：在 NLP 关键词覆盖段追加 model_comparison 关键词**

```python
_mc_trigger_kw = ["多模型", "模型对比", "最优模型", "auto", "哪个模型最好", "建立多个模型", "比较模型"]
if mode in ("y_vs_all", "multi_x_vs_y") and any(k in inst_lower for k in _mc_trigger_kw):
    mode = "model_comparison"
    # x_cols 保留 NLP 解析的特征列（若有）
```

**Step 5：验证服务启动并返回正常**

```bash
python app.py &
sleep 3
curl -s http://localhost:7860/api/history
kill %1
```

**Step 6：提交**

```bash
git add api/routes.py
git commit -m "feat: route model_comparison mode in api/routes.py"
```

---

## Task 21：更新前端（手动模式下拉框 + Y vs All 联动按钮）

**Files:**
- Modify: `templates/analyze.html`

**Step 1：在手动模式下拉框中追加 model_comparison 选项**

在 Task 13 追加的末尾之后再追加：

```html
<option value="model_comparison">🏆 多模型对比（AutoML-lite）</option>
```

**Step 2：在 y_vs_all 结果展示区追加"一键跑模型对比"按钮**

找到分析结果展示区域（`summary_text` 展示部分），在其下方追加：

```html
<!-- 仅在 y_vs_all 结果出来后显示 -->
<template x-if="result && currentMode === 'y_vs_all'">
  <div class="mt-3">
    <button
      class="btn btn-sm btn-outline-warning"
      @click="runModelComparison()"
      :disabled="loading"
    >
      🏆 用 Top 5 因素跑多模型对比
    </button>
  </div>
</template>
```

**Step 3：在 Alpine.js 脚本中追加 `runModelComparison` 方法**

```javascript
async runModelComparison() {
  // 读取当前 y_vs_all 结果中的 top 5 特征
  const topFeatures = (this.tableData || [])
    .slice(0, 5)
    .map(r => r['特征'])
    .filter(Boolean);
  if (!topFeatures.length) {
    alert('请先完成 y_vs_all 分析以获取 Top 特征');
    return;
  }
  // 以 model_comparison 模式重新提交
  this.manualMode = 'model_comparison';
  this.manualXCols = topFeatures;
  await this.runAnalysis();
},
```

**Step 4：验证页面按钮可正常渲染（启动服务，访问 /analyze，完成 y_vs_all 分析后按钮出现）**

**Step 5：提交**

```bash
git add templates/analyze.html
git commit -m "feat: add model_comparison to dropdown + one-click button after y_vs_all"
```

---

## Task 22：端到端集成测试（model_comparison）

**Step 1：运行集成测试**

```bash
python -c "
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.analysis_engine import analyze_model_comparison
from core.report_builder import build_report
from core.result_formatter import format_result

np.random.seed(42)
n = 120
df = pd.DataFrame({
    'x1': np.random.randn(n),
    'x2': np.random.randn(n),
    'x3': np.random.choice(['A','B','C'], n),  # 分类变量
})
df['y'] = 2*df['x1'] - df['x2']**2 + np.where(df['x3']=='A', 1, -1) + np.random.randn(n)*0.3

r = analyze_model_comparison(df, 'y', ['x1','x2'])

# 检查结果完整性
assert r.mode == 'model_comparison'
assert len(r.mc_predictions) == 5, f'期望5个模型, 得到 {len(r.mc_predictions)}'
assert r.mc_best_model_name in r.mc_predictions
assert r.mc_comparison_df is not None and len(r.mc_comparison_df) == 5

# 测试报告生成
import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
    tmp = f.name
build_report(r, tmp)
assert os.path.getsize(tmp) > 5000, '报告文件异常小'
os.remove(tmp)

# 测试格式化器
text = format_result(r, '哪个模型最好')
assert r.mc_best_model_name in text

print('=== model_comparison 集成测试全部通过 ===')
print(f'最优模型: {r.mc_best_model_name} (CV R²={r.mc_best_model_r2:.3f})')
print(r.mc_comparison_df[['模型','CV均值R²','RMSE']].to_string(index=False))
"
```

**预期输出：**
```
=== model_comparison 集成测试全部通过 ===
最优模型: 梯度提升 (CV R²=0.xxx)
       模型  CV均值R²    RMSE
     梯度提升   0.xxx   0.xxx
     随机森林   0.xxx   0.xxx
     ...
```

**Step 2：提交**

```bash
git add tests/
git commit -m "test: add integration test for model_comparison mode"
```

---

## 依赖变更汇总

| 包 | 现状 | 变更 |
|----|------|------|
| `statsmodels>=0.14.0` | 未安装 | **新增**（Task 1） |
| `scikit-learn>=1.3.0` | 已有 | 无变更（MLPRegressor/LogisticRegression/KMeans/PCA/Ridge/Lasso/GradientBoosting 均在其中） |
| `scipy>=1.10.0` | 已有 | 无变更（ANOVA 使用 `scipy.stats.f_oneway`） |

## 新增分析模式汇总

| 模式 | 函数 | 新 Requirements | 最小数据量 |
|------|------|----------------|-----------|
| `time_series` | `analyze_time_series` | statsmodels | 10 行 |
| `pca` | `analyze_pca` | —（sklearn 已有） | 2 行 2 列 |
| `anova` | `analyze_anova` | statsmodels（Tukey） | 3 组 × 3 行 |
| `logistic` | `analyze_logistic` | —（sklearn 已有） | 20 行 |
| `cluster` | `analyze_cluster` | —（sklearn 已有） | 20 行 |
| `neural_reg` | `analyze_neural_reg` | —（sklearn 已有） | 30 行 |
| `ridge_lasso` | `analyze_ridge_lasso` | —（sklearn 已有） | 10 行 |
| `model_comparison` | `analyze_model_comparison` | —（sklearn 已有） | 30 行 |
