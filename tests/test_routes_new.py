"""新 /api/chat 端点集成测试（用 mock 替换 LLM）"""
import pytest
import io
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

client = TestClient(app)


def _upload_test_file():
    """上传测试 Excel，返回 file_id"""
    df = pd.DataFrame({
        "部门": ["销售", "研发", "销售"],
        "销售额": [1200, 800, 1500],
        "年龄": [25, 30, None],
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    resp = client.post(
        "/api/upload",
        files={"file": ("test.xlsx", buf,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["file_id"]


@pytest.fixture(scope="module")
def file_id():
    return _upload_test_file()


# ── 基本功能 ──────────────────────────────────────────────────────────────────

@patch("api.routes.AIPLanner")
def test_chat_filter_rows(mock_cls, file_id):
    from core.ai_planner import PlanStep
    mock_planner = MagicMock()
    mock_cls.return_value = mock_planner
    mock_planner.plan.return_value = [
        PlanStep("filter_rows", {"condition": "销售额 > 1000"}, "筛选")
    ]
    resp = client.post("/api/chat", json={"file_id": file_id, "instruction": "筛选销售额大于1000的行"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["table_data"]) == 2


@patch("api.routes.AIPLanner")
def test_chat_delete_column(mock_cls, file_id):
    from core.ai_planner import PlanStep
    mock_planner = MagicMock()
    mock_cls.return_value = mock_planner
    mock_planner.plan.return_value = [
        PlanStep("delete_columns", {"columns": ["年龄"]}, "删除年龄")
    ]
    resp = client.post("/api/chat", json={"file_id": file_id, "instruction": "删除年龄列"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    if data["table_data"]:
        assert "年龄" not in data["table_data"][0]


@patch("api.routes.AIPLanner")
def test_chat_returns_chart_option(mock_cls, file_id):
    from core.ai_planner import PlanStep
    mock_planner = MagicMock()
    mock_cls.return_value = mock_planner
    mock_planner.plan.return_value = [
        PlanStep("build_chart", {"chart_type": "bar", "x": "部门", "y": ["销售额"]}, "画图")
    ]
    resp = client.post("/api/chat", json={"file_id": file_id, "instruction": "画柱状图"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["chart_option"] is not None


@patch("api.routes.AIPLanner")
def test_chat_skill_error_returns_friendly_message(mock_cls, file_id):
    from core.ai_planner import PlanStep
    mock_planner = MagicMock()
    mock_cls.return_value = mock_planner
    mock_planner.plan.return_value = [
        PlanStep("delete_columns", {"columns": ["不存在列"]}, "删列")
    ]
    resp = client.post("/api/chat", json={"file_id": file_id, "instruction": "删除不存在列"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "不存在列" in data["error"]


@patch("api.routes.AIPLanner")
def test_chat_returns_steps_list(mock_cls, file_id):
    from core.ai_planner import PlanStep
    mock_planner = MagicMock()
    mock_cls.return_value = mock_planner
    mock_planner.plan.return_value = [
        PlanStep("fill_missing", {"method": "mean"}, "填充"),
        PlanStep("deduplicate", {}, "去重"),
    ]
    resp = client.post("/api/chat", json={"file_id": file_id, "instruction": "清洗数据"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert isinstance(data["steps"], list)
    assert len(data["steps"]) == 2


def test_chat_invalid_file_id():
    resp = client.post("/api/chat", json={"file_id": "nonexistent_id", "instruction": "删除年龄列"})
    assert resp.status_code == 400
