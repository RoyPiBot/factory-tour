"""
test_api.py - 測試 FastAPI 端點（不需要 Groq API Key）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """建立測試用 FastAPI client"""
    # 注意：這會嘗試初始化 agent，但因為是 test key 會失敗
    # 靜態端點仍然可以測試
    from main import app

    with TestClient(app) as c:
        yield c


class TestStaticEndpoints:
    """靜態資料端點測試（不需要 Agent）"""

    def test_root(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "工廠導覽" in res.text or "Factory" in res.text

    def test_areas(self, client):
        res = client.get("/areas")
        assert res.status_code == 200
        data = res.json()
        assert "areas" in data
        assert len(data["areas"]) >= 5
        # 檢查排序
        orders = [a["order"] for a in data["areas"]]
        assert orders == sorted(orders)

    def test_area_detail(self, client):
        res = client.get("/areas/lobby")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "大廳"
        assert "description" in data

    def test_area_detail_by_name(self, client):
        res = client.get("/areas/大廳")
        assert res.status_code == 200

    def test_area_not_found(self, client):
        res = client.get("/areas/不存在")
        assert res.status_code == 404

    def test_routes(self, client):
        res = client.get("/routes")
        assert res.status_code == 200
        data = res.json()
        assert len(data["routes"]) >= 2

    def test_faq(self, client):
        res = client.get("/faq")
        assert res.status_code == 200
        data = res.json()
        assert "faq" in data
        assert len(data["faq"]) > 0

    def test_health(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "agent_ready" in data
        assert "rag_ready" in data
        assert "languages" in data

    def test_i18n_zh(self, client):
        res = client.get("/i18n/zh-TW")
        assert res.status_code == 200
        data = res.json()
        assert data["strings"]["title"] == "工廠導覽系統"

    def test_i18n_en(self, client):
        res = client.get("/i18n/en")
        assert res.status_code == 200
        data = res.json()
        assert data["strings"]["title"] == "Factory Tour System"

    def test_i18n_ja(self, client):
        res = client.get("/i18n/ja")
        assert res.status_code == 200
        data = res.json()
        assert data["strings"]["title"] == "工場見学システム"


class TestTourEndpoints:
    """導覽流程端點測試"""

    def test_tour_routes(self, client):
        res = client.get("/tour/routes")
        assert res.status_code == 200
        data = res.json()
        assert len(data["routes"]) >= 2

    def test_start_tour(self, client):
        res = client.post(
            "/tour/start",
            json={"session_id": "test-api-tour", "language": "zh-TW"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "started"
        assert "message" in data

    def test_next_stop(self, client):
        client.post(
            "/tour/start",
            json={"session_id": "test-next", "language": "zh-TW"},
        )
        res = client.post("/tour/next", json={"session_id": "test-next"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "active"

    def test_tour_status(self, client):
        client.post(
            "/tour/start",
            json={"session_id": "test-status"},
        )
        res = client.get("/tour/status/test-status")
        assert res.status_code == 200
        data = res.json()
        assert "current_step" in data

    def test_tour_status_not_found(self, client):
        res = client.get("/tour/status/nonexistent")
        assert res.status_code == 404


class TestHistoryEndpoints:
    """歷史與統計端點測試"""

    def test_sessions(self, client):
        res = client.get("/sessions")
        assert res.status_code == 200
        assert "sessions" in res.json()

    def test_history_empty(self, client):
        res = client.get("/history/empty-session")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 0

    def test_stats(self, client):
        res = client.get("/stats")
        assert res.status_code == 200
        data = res.json()
        assert "database" in data
        assert "agents" in data
