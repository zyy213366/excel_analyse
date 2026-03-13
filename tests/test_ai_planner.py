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
    @patch("core.ai_planner.OpenAI")
    def test_build_prompt_contains_schema(self, mock_openai_cls, schema_df):
        mock_openai_cls.return_value = MagicMock()
        planner = AIPLanner()
        prompt = planner._build_prompt(schema_df, "删除年龄列")
        assert "年龄" in prompt
        assert "delete_columns" in prompt
        assert "删除年龄列" in prompt

    @patch("core.ai_planner.OpenAI")
    def test_build_prompt_contains_skill_list(self, mock_openai_cls, schema_df):
        mock_openai_cls.return_value = MagicMock()
        planner = AIPLanner()
        prompt = planner._build_prompt(schema_df, "随便")
        assert "filter_rows" in prompt
        assert "aggregate" in prompt

    @patch("core.ai_planner.OpenAI")
    def test_plan_calls_deepseek_api(self, mock_openai_cls, schema_df):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(
            content='{"plan": [{"skill": "filter_rows", "params": {"condition": "销售额 > 1000"}, "reason": "筛选"}]}'
        ))]
        mock_client.chat.completions.create.return_value = mock_resp

        planner = AIPLanner()
        steps = planner.plan(schema_df, "筛选销售额大于1000的行")
        assert len(steps) == 1
        assert steps[0].skill == "filter_rows"
        mock_client.chat.completions.create.assert_called_once()

    @patch("core.ai_planner.OpenAI")
    def test_plan_multi_step(self, mock_openai_cls, schema_df):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='''{"plan": [
            {"skill": "delete_columns", "params": {"columns": ["年龄"]}, "reason": "删除年龄"},
            {"skill": "aggregate", "params": {"group_by": ["部门"], "agg": {"销售额": "sum"}}, "reason": "分组求和"},
            {"skill": "build_chart", "params": {"chart_type": "bar", "x": "部门", "y": ["销售额_sum"]}, "reason": "画图"}
        ]}'''))]
        mock_client.chat.completions.create.return_value = mock_resp

        planner = AIPLanner()
        steps = planner.plan(schema_df, "删除年龄列，按部门统计销售额，画柱状图")
        assert len(steps) == 3
        assert steps[2].skill == "build_chart"

    @patch("core.ai_planner.DEEPSEEK_API_KEY", "")
    def test_missing_api_key_raises(self):
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            AIPLanner()
