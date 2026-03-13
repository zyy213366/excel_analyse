"""
Skill Executor — 顺序执行 plan，df 沿 pipeline 传递。
支持 run_analysis 特殊 skill（路由到分析引擎）。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd
from core.skills import SKILL_REGISTRY, SkillResult
from core.ai_planner import PlanStep


@dataclass
class ExecutionResult:
    df: pd.DataFrame
    success: bool
    steps_summary: list = field(default_factory=list)
    chart_option: Optional[dict] = None
    error: Optional[str] = None
    analysis_result: Optional[Any] = None   # AnalysisResult，由 run_analysis skill 填充
    result_sheets: Optional[dict] = None    # 多表结果，由 split_columns skill 填充

    @property
    def full_summary(self) -> str:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.steps_summary))


def _run_analysis_skill(df: pd.DataFrame, mode: str,
                        target_y: str = "", x_cols: list = None) -> SkillResult:
    """
    将分析模式路由到 core.analysis_engine 对应函数。
    返回的 SkillResult.df 是摘要表，analysis_result 通过 error 字段的特殊值传递
    （真正的 AnalysisResult 对象通过 execute_plan 的 analysis_result 字段返回）。
    """
    from core.analysis_engine import (
        analyze_y_vs_all, analyze_two_column, analyze_multi_x_vs_y,
        analyze_model_comparison, analyze_compare, analyze_crosstab,
    )
    from utils.data_loader import get_numeric_columns, preprocess_for_analysis

    if x_cols is None:
        x_cols = []

    numeric_cols = get_numeric_columns(df)

    # 校验 target_y
    if target_y and target_y not in df.columns:
        # 尝试大小写不敏感匹配
        tl = target_y.lower()
        match = next((c for c in df.columns if c.lower() == tl), None)
        if match:
            target_y = match
        else:
            return SkillResult(df, "", error=f"目标列 {target_y!r} 不存在，可用列：{list(df.columns)[:8]}")

    # 如果没有指定 target_y，取第一个数值列
    if not target_y and numeric_cols:
        target_y = numeric_cols[0]

    # 确定 cols_needed 并预处理
    # compare/anova/crosstab 的分组列可能是字符串，不能扔进数值预处理
    _categorical_group_modes = {"compare", "anova", "crosstab"}

    if mode in {"y_vs_all", "model_comparison", "neural_reg", "ridge_lasso", "multi_x_vs_y", "logistic"}:
        cols_needed = numeric_cols  # 全部数值列
    elif mode in {"pca", "cluster"}:
        cols_needed = x_cols if x_cols else numeric_cols
    elif mode == "two_column":
        cols_needed = [target_y, x_cols[0]] if x_cols else [target_y]
    elif mode in _categorical_group_modes:
        # 只对数值目标列做预处理，分组列单独保留
        cols_needed = [target_y] if target_y else []
    else:
        cols_needed = ([target_y] if target_y else []) + x_cols

    try:
        if mode in _categorical_group_modes and cols_needed:
            # 仅对数值目标列做清洗；分组列保持原始字符串不做 to_numeric
            _num_clean, raw_cnt, valid_cnt = preprocess_for_analysis(df, cols_needed)
            # preprocess_for_analysis 的 dropna 保留了原始 index，可安全 loc 回来
            group_col_for_mode = x_cols[0] if x_cols else ""
            extra_cols = [c for c in x_cols if c in df.columns and c != target_y]
            # 把干净的数值列 + 原始分组列组合成 clean_df
            clean_df = _num_clean.join(df.loc[_num_clean.index, extra_cols])
        else:
            clean_df, raw_cnt, valid_cnt = preprocess_for_analysis(df, cols_needed)
    except Exception as e:
        return SkillResult(df, "", error=f"数据预处理失败：{e}")

    if valid_cnt < 10:
        return SkillResult(df, "", error=f"有效数据不足（{valid_cnt} 行），无法分析")

    # 确定实际 x_cols（预处理后）
    clean_numeric = get_numeric_columns(clean_df)
    if x_cols:
        valid_x = [c for c in x_cols if c in clean_df.columns and c != target_y]
    else:
        valid_x = [c for c in clean_numeric if c != target_y]

    try:
        if mode == "y_vs_all":
            result = analyze_y_vs_all(clean_df, target_y)
        elif mode == "model_comparison":
            result = analyze_model_comparison(clean_df, target_y, valid_x)
        elif mode == "multi_x_vs_y":
            result = analyze_multi_x_vs_y(clean_df, target_y, valid_x)
        elif mode == "two_column":
            col2 = x_cols[0] if x_cols else (valid_x[0] if valid_x else "")
            if not col2:
                return SkillResult(df, "", error="two_column 需要指定第二列")
            result = analyze_two_column(clean_df, col2, target_y)
        elif mode == "compare":
            group_c = x_cols[0] if x_cols else ""
            if not group_c:
                return SkillResult(df, "", error="compare 需要指定分组列")
            result = analyze_compare(clean_df, target_y, group_c)
        elif mode == "crosstab":
            row_c = x_cols[0] if x_cols else ""
            col_c = x_cols[1] if len(x_cols) >= 2 else target_y
            result = analyze_crosstab(clean_df, row_c, col_c)
        elif mode in {"neural_reg", "ridge_lasso", "logistic", "pca", "cluster",
                      "anova", "time_series"}:
            # 动态导入
            import importlib
            engine = importlib.import_module("core.analysis_engine")
            fn_map = {
                "neural_reg": "analyze_neural_reg",
                "ridge_lasso": "analyze_ridge_lasso",
                "logistic": "analyze_logistic",
                "pca": "analyze_pca",
                "cluster": "analyze_cluster",
                "anova": "analyze_anova",
                "time_series": "analyze_time_series",
            }
            fn = getattr(engine, fn_map[mode])
            if mode in {"pca", "cluster"}:
                result = fn(clean_df, valid_x if valid_x else clean_numeric)
            elif mode == "anova":
                gc = x_cols[0] if x_cols else ""
                if not gc:
                    return SkillResult(df, "", error="anova 需要指定分组列")
                result = fn(clean_df, target_y, gc)
            elif mode == "time_series":
                tc = x_cols[0] if x_cols else ""
                if not tc:
                    return SkillResult(df, "", error="time_series 需要指定时间列")
                result = fn(clean_df, tc, target_y)
            else:
                result = fn(clean_df, target_y, valid_x)
        else:
            return SkillResult(df, "", error=f"未知分析模式：{mode!r}")
    except Exception as e:
        import traceback
        return SkillResult(df, "", error=f"分析执行失败：{e}\n{traceback.format_exc()[:400]}")

    # 把 AnalysisResult 藏在 SkillResult 里通过特殊属性传递
    sr = SkillResult(df=clean_df, summary=result.summary_text or "分析完成")
    sr._analysis_result = result          # executor 会提取这个
    return sr


def execute_plan(df: pd.DataFrame, plan: list) -> ExecutionResult:
    """
    顺序执行 plan 中的每个 step。
    - df 沿 pipeline 传递（每步输出是下一步输入）
    - run_analysis skill 特殊处理：执行分析引擎并填充 analysis_result
    - 遇到 error 立即停止，返回失败结果
    - chart_option 由最后一个 build_chart / run_analysis step 决定
    """
    current_df = df.copy()
    summaries = []
    chart_option = None
    analysis_result = None
    result_sheets = None

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

        # run_analysis 特殊处理
        if step.skill == "run_analysis":
            try:
                sr = _run_analysis_skill(current_df, **step.params)
            except TypeError as e:
                return ExecutionResult(df=current_df, success=False,
                                       steps_summary=summaries,
                                       error=f"run_analysis 参数错误：{e}")
            if sr.error:
                return ExecutionResult(df=current_df, success=False,
                                       steps_summary=summaries, error=sr.error)
            current_df = sr.df
            summaries.append(sr.summary)
            if hasattr(sr, "_analysis_result"):
                analysis_result = sr._analysis_result
            continue

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
        if hasattr(skill_result, "_result_sheets"):
            result_sheets = skill_result._result_sheets

    return ExecutionResult(
        df=current_df,
        success=True,
        steps_summary=summaries,
        chart_option=chart_option,
        analysis_result=analysis_result,
        result_sheets=result_sheets,
    )
