"""AI Planner 测试（用 mock 替换真实 LLM 调用）"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from core.ai_planner import AIPLanner, PlanStep, plan_to_steps


@pytest.fixture
def schema_df():
    return pd.DataFrame({
        "姓名": ["张三", "李四"],
        "年龄": [25, 30],
        "部门": ["销售", "研发"],
        "销售额": [1200, 800],
    })


class TestPlanStep:
    def test_valid_step(self):
        step = PlanStep(skill="filter_rows", params={"condition": "销售额 > 1000"}, reason="筛选高销售")
        assert step.skill == "filter_rows"
        assert step.params["condition"] == "销售额 > 1000"

    def test_from_dict(self):
        d = {"skill": "delete_columns", "params": {"columns": ["年龄"]}, "reason": "删列"}
        step = PlanStep(**d)
        assert step.skill == "delete_columns"


class TestPlanToSteps:
    def test_parses_valid_json(self):
        raw = '{"plan": [{"skill": "filter_rows", "params": {"condition": "销售额 > 1000"}, "reason": "筛选"}]}'
        steps = plan_to_steps(raw)
        assert len(steps) == 1
        assert steps[0].skill == "filter_rows"

    def test_handles_markdown_code_block(self):
        raw = '```json\n{"plan": [{"skill": "sort", "params": {"column": "销售额", "ascending": false}, "reason": "排序"}]}\n```'
        steps = plan_to_steps(raw)
        assert len(steps) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="无法解析"):
            plan_to_steps("这不是JSON")

    def test_unknown_skill_raises(self):
        raw = '{"plan": [{"skill": "fly_to_moon", "params": {}, "reason": ""}]}'
        with pytest.raises(ValueError, match="未知 skill"):
            plan_to_steps(raw)


class TestAIPlanner:
    def test_build_prompt_contains_schema(self, schema_df):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("core.ai_planner.anthropic.Anthropic"):
                planner = AIPLanner()
                prompt = planner._build_prompt(schema_df, "删除年龄列")
                assert "年龄" in prompt
                assert "delete_columns" in prompt
                assert "删除年龄列" in prompt

    def test_build_prompt_contains_skill_list(self, schema_df):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("core.ai_planner.anthropic.Anthropic"):
                planner = AIPLanner()
                prompt = planner._build_prompt(schema_df, "随便")
                assert "filter_rows" in prompt
                assert "aggregate" in prompt

    @patch("core.ai_planner.anthropic.Anthropic")
    def test_plan_calls_claude_api(self, mock_anthropic_cls, schema_df):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text='{"plan": [{"skill": "filter_rows", "params": {"condition": "销售额 > 1000"}, "reason": "筛选"}]}')]
            mock_client.messages.create.return_value = mock_msg

            planner = AIPLanner()
            steps = planner.plan(schema_df, "筛选销售额大于1000的行")
            assert len(steps) == 1
            assert steps[0].skill == "filter_rows"
            mock_client.messages.create.assert_called_once()

    @patch("core.ai_planner.anthropic.Anthropic")
    def test_plan_multi_step(self, mock_anthropic_cls, schema_df):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            mock_client = MagicMock()
            mock_anthropic_cls.return_value = mock_client
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text='''{"plan": [
                {"skill": "delete_columns", "params": {"columns": ["年龄"]}, "reason": "删除年龄"},
                {"skill": "aggregate", "params": {"group_by": ["部门"], "agg": {"销售额": "sum"}}, "reason": "分组求和"},
                {"skill": "build_chart", "params": {"chart_type": "bar", "x": "部门", "y": ["销售额_sum"]}, "reason": "画图"}
            ]}''')]
            mock_client.messages.create.return_value = mock_msg

            planner = AIPLanner()
            steps = planner.plan(schema_df, "删除年龄列，按部门统计销售额，画柱状图")
            assert len(steps) == 3
            assert steps[2].skill == "build_chart"

    def test_missing_api_key_raises(self):
        import os
        # 确保没有 key
        env_backup = {}
        for k in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
            if k in os.environ:
                env_backup[k] = os.environ.pop(k)
        try:
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                AIPLanner()
        finally:
            os.environ.update(env_backup)
