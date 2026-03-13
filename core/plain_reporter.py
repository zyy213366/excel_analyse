"""
plain_reporter.py — 用 DeepSeek 把专业分析结果翻译成通俗业务报告。
每种分析模式都有定制化的提示词，确保解释准确且有针对性。
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.analysis_engine import AnalysisResult


# ── 各模式的补充说明（帮助 DS 理解上下文）─────────────────────────────────────

_MODE_CONTEXT = {
    "y_vs_all": (
        "这是「影响因素分析」，使用随机森林算法计算每个自变量对目标变量的重要性得分（0~1），"
        "得分越高说明该变量对目标的影响越大。"
    ),
    "model_comparison": (
        "这是「多模型对比」，同时训练多个机器学习模型（线性回归、随机森林等）并比较效果。"
        "R²（决定系数）越接近1越好，MAE/RMSE（误差）越小越好。"
    ),
    "multi_x_vs_y": (
        "这是「多元线性回归」，分析多个自变量对目标变量的线性影响。"
        "系数代表该变量每增加1个单位，目标变量的变化量；p值<0.05表示该影响在统计上显著。"
    ),
    "neural_reg": (
        "这是「神经网络回归」，使用深度学习模型建立输入和输出之间的复杂非线性关系。"
        "R²越接近1越好；训练集和测试集R²差距很大说明过拟合（模型只记住了训练数据）。"
    ),
    "ridge_lasso": (
        "这是「岭回归/套索回归」，处理自变量之间存在多重共线性（互相关联）时的特殊回归方法。"
        "套索回归会把不重要的变量系数压缩为0，相当于自动筛选变量。"
    ),
    "logistic": (
        "这是「逻辑回归分类」，预测某个事件是否发生（二分类）。"
        "准确率代表预测正确的比例；正系数表示该变量增大时目标事件更可能发生。"
    ),
    "cluster": (
        "这是「K-Means聚类」，将数据自动分成若干组，同一组内的记录相互相似。"
        "每个簇中心代表该组的「典型画像」。"
    ),
    "pca": (
        "这是「主成分分析（PCA）」，把很多高度相关的变量合并成少数几个综合指标（主成分），"
        "在不太损失信息的前提下简化数据结构。解释方差比例越高，说明这些主成分保留的信息越多。"
    ),
    "anova": (
        "这是「方差分析（ANOVA）」，检验不同分组之间某个指标的均值是否存在显著差异。"
        "p值<0.05表示各组之间存在显著差异，F值越大差异越明显。"
    ),
    "time_series": (
        "这是「时间序列分析」，分析某指标随时间的变化趋势、周期性规律。"
    ),
    "two_column": (
        "这是「双变量相关性分析」，检验两个变量之间是否存在线性关联。"
        "相关系数r：±1表示完全线性关系，0表示无线性关系；|r|>0.5通常认为中等相关。"
    ),
    "compare": (
        "这是「分组对比分析」，比较不同分组在某指标上的差异和分布。"
        "p值<0.05表示各组之间存在显著差异；t/F统计量越大差异越明显。"
    ),
    "crosstab": (
        "这是「交叉分析（列联表）」，检验两个分类变量之间是否存在关联。"
        "卡方检验p值<0.05表示两个变量不独立（存在显著关联）；"
        "Cramér's V是关联强度：0~0.1极弱，0.1~0.3弱，0.3~0.5中等，>0.5强。"
    ),
}

_MODE_NAMES = {
    "y_vs_all": "影响因素分析",
    "model_comparison": "多模型对比",
    "multi_x_vs_y": "多元线性回归",
    "neural_reg": "神经网络回归",
    "ridge_lasso": "岭回归/套索回归",
    "logistic": "逻辑回归（分类预测）",
    "cluster": "聚类分析",
    "pca": "主成分分析（PCA）",
    "anova": "方差分析",
    "time_series": "时间序列分析",
    "two_column": "相关性分析",
    "compare": "分组对比分析",
    "crosstab": "交叉分析（列联表）",
}


def _extract_key_numbers(analysis: "AnalysisResult") -> str:
    """从 AnalysisResult 提取关键数字，压缩为紧凑文本供 DS 解读。"""
    lines = []
    mode = analysis.mode

    if mode == "y_vs_all" and analysis.feature_importance_df is not None:
        df = analysis.feature_importance_df
        top = df.head(8)
        lines.append("【特征重要性 Top8】")
        for _, row in top.iterrows():
            feat = row.get("特征") or row.get("feature") or row.iloc[0]
            imp = row.get("重要性") or row.get("importance") or row.iloc[1]
            lines.append(f"  {feat}: {float(imp):.4f}")

    elif mode == "model_comparison" and analysis.mc_comparison_df is not None:
        df = analysis.mc_comparison_df
        lines.append("【模型对比结果】")
        for _, row in df.iterrows():
            lines.append("  " + " | ".join(f"{k}={v}" for k, v in row.items()))

    elif mode == "multi_x_vs_y" and analysis.feature_importance_df is not None:
        df = analysis.feature_importance_df
        lines.append(f"【多元回归 R²={getattr(analysis, 'r2', 'N/A')}】")
        lines.append("【回归系数与显著性】")
        for _, row in df.iterrows():
            feat = row.iloc[0]
            coef = row.get("系数") or row.get("coef") or (row.iloc[1] if len(row) > 1 else "")
            pval = row.get("p值") or row.get("p_value") or (row.iloc[-1] if len(row) > 2 else "")
            lines.append(f"  {feat}: 系数={coef}, p值={pval}")

    elif mode == "neural_reg":
        lines.append(f"训练集R²={getattr(analysis, 'neural_r2_train', 'N/A'):.4f}")
        lines.append(f"测试集R²={getattr(analysis, 'neural_r2_test', 'N/A'):.4f}")

    elif mode == "ridge_lasso" and analysis.feature_importance_df is not None:
        df = analysis.feature_importance_df
        lines.append("【回归系数】")
        for _, row in df.iterrows():
            lines.append("  " + " | ".join(f"{k}={v}" for k, v in row.items()))

    elif mode == "logistic":
        lines.append(f"准确率={getattr(analysis, 'logistic_accuracy', 'N/A'):.2%}")
        if analysis.logistic_coef_df is not None:
            lines.append("【各变量系数】")
            for _, row in analysis.logistic_coef_df.head(10).iterrows():
                lines.append("  " + " | ".join(f"{k}={v}" for k, v in row.items()))

    elif mode == "cluster":
        lines.append(f"聚类数={getattr(analysis, 'cluster_n', 'N/A')}")
        if analysis.cluster_centers_df is not None:
            lines.append("【各簇中心特征】")
            lines.append(analysis.cluster_centers_df.to_string(max_rows=5))

    elif mode == "pca":
        ratios = getattr(analysis, "pca_explained_ratio", [])
        if ratios:
            lines.append(f"前{len(ratios)}个主成分累计解释方差：{sum(ratios):.1%}")
            for i, r in enumerate(ratios[:5]):
                lines.append(f"  PC{i+1}: {r:.1%}")

    elif mode == "anova":
        lines.append(f"F统计量={getattr(analysis, 'anova_f_stat', 'N/A'):.4f}")
        lines.append(f"p值={getattr(analysis, 'anova_p_value', 'N/A'):.4f}")
        lines.append(f"分组列={getattr(analysis, 'anova_group_col', '')}")
        if analysis.anova_group_stats is not None:
            lines.append("【各组统计】")
            lines.append(analysis.anova_group_stats.to_string())

    elif mode == "two_column":
        lines.append(f"Pearson相关系数={getattr(analysis, 'pearson_r', 'N/A'):.4f}")
        lines.append(f"p值={getattr(analysis, 'pearson_p', 'N/A'):.4f}")

    elif mode == "time_series":
        lines.append(f"时间列={getattr(analysis, 'time_col', '')}")
        lines.append(f"目标变量={analysis.target_y}")

    elif mode == "crosstab":
        row_col = getattr(analysis, "crosstab_row_col", "")
        col_col = getattr(analysis, "crosstab_col_col", "")
        lines.append(f"行变量={row_col}，列变量={col_col}")
        if analysis.summary_text:
            # 从 summary_text 里提取卡方结果段落
            for line in analysis.summary_text.splitlines():
                if any(k in line for k in ["χ²", "p 值", "Cramér", "结论", "Top"]):
                    lines.append(line.strip("- #*").strip())

    elif mode == "compare":
        grp_col = getattr(analysis, "compare_group_col", "")
        val_col = getattr(analysis, "compare_value_col", "")
        stat = getattr(analysis, "compare_stat", None)
        pval = getattr(analysis, "compare_p_value", None)
        lines.append(f"分组列={grp_col}，目标值={val_col}")
        if stat is not None:
            lines.append(f"统计量={stat:.4f}")
        if pval is not None:
            lines.append(f"p值={pval:.4f}")
        if analysis.compare_group_stats_df is not None:
            lines.append("【各组统计】")
            lines.append(analysis.compare_group_stats_df.to_string(index=False))

    # 通用：附上原始 summary_text 的前600字符
    if analysis.summary_text:
        lines.append("\n【原始分析摘要（节选）】")
        lines.append(analysis.summary_text[:600])

    return "\n".join(lines)


def generate_plain_report(analysis: "AnalysisResult") -> str:
    """
    调用 DeepSeek，把技术性分析结果翻译成普通用户能读懂的业务报告。
    返回 markdown 格式的通俗报告字符串。
    """
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    if not DEEPSEEK_API_KEY:
        return ""

    mode = analysis.mode
    mode_name = _MODE_NAMES.get(mode, mode)
    mode_ctx = _MODE_CONTEXT.get(mode, "")
    key_numbers = _extract_key_numbers(analysis)
    target_y = analysis.target_y or "目标变量"

    prompt = f"""你是一位擅长把数据分析结果用大白话讲清楚的顾问，面向完全不懂统计学的业务人员写报告。

## 本次分析信息
- **分析类型**：{mode_name}
- **分析目标（Y）**：{target_y}
- **方法说明**：{mode_ctx}

## 关键分析数据
{key_numbers}

## 你的任务
请写一份【通俗业务解读报告】，严格按以下结构输出：

### 📊 分析结论（一句话版本）
用一句话说明最核心的发现是什么。

### 🔍 详细解读
逐一解释重要数字的含义，格式：
- **[指标名]**：[数值] → [大白话解释是什么意思，对业务有什么含义]

例如："**温度重要性得分**：0.38 → 温度是影响目标最大的因素，大约贡献了38%的影响力"
例如："**p值**：0.003 → 远小于0.05，说明这个关系不是偶然的，统计上非常可靠"
例如："**R²**：0.72 → 模型能解释72%的数据变化，剩下28%可能来自未收集的因素"

### 💡 业务建议
基于分析结果，给出2-3条具体的、可操作的建议（不要泛泛而谈）。

### ⚠️ 注意事项 / 局限性
简单提醒1-2条这个分析的局限或使用时要注意的事项。

---
要求：
1. 完全避免统计术语，如必须用请括号解释
2. 数字要结合{target_y}的实际含义来解释
3. 每条解读要具体，不能只说"越高越好"
4. 总长度控制在400字以内，简洁有力"""

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"（通俗报告生成失败：{e}）"
