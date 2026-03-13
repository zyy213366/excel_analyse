"""Skill Executor 测试"""
import pytest
import pandas as pd
from core.ai_planner import PlanStep
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
        assert result.df["销售额"].iloc[0] == 1500

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

    def test_unknown_skill_returns_error(self, df):
        plan = [PlanStep("fly_to_moon", {}, "飞向月球")]
        result = execute_plan(df, plan)
        assert not result.success
        assert "fly_to_moon" in result.error

    def test_df_passes_through_pipeline(self, df):
        """验证 df 确实沿 pipeline 流动——前一步删列，后一步看不到该列"""
        plan = [
            PlanStep("delete_columns", {"columns": ["年龄"]}, "删除年龄"),
            PlanStep("sort", {"column": "销售额", "ascending": True}, "排序"),
        ]
        result = execute_plan(df, plan)
        assert result.success
        assert "年龄" not in result.df.columns


class TestExecutionResult:
    def test_full_summary_joins_steps(self, df):
        plan = [PlanStep("filter_rows", {"condition": "销售额 > 1000"}, "筛选")]
        result = execute_plan(df, plan)
        assert "1." in result.full_summary

    def test_full_summary_empty_for_no_steps(self, df):
        result = execute_plan(df, [])
        assert result.full_summary == ""
