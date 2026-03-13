"""
Skill Executor — 顺序执行 plan，df 沿 pipeline 传递。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
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

    @property
    def full_summary(self) -> str:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.steps_summary))


def execute_plan(df: pd.DataFrame, plan: list) -> ExecutionResult:
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
