"""
test_tour_flow.py - 測試互動式導覽流程
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tour_flow import TourSession, TourManager


class TestTourSession:
    """單一導覽 session 測試"""

    def test_create_session_default(self):
        session = TourSession("test-1")
        assert session.route_name == "標準導覽路線"
        assert session.current_step == 0
        assert session.current_area == "大廳"
        assert session.total_stops == 5
        assert session.completed is False

    def test_create_session_quick_route(self):
        session = TourSession("test-2", "快速導覽路線")
        assert session.route_name == "快速導覽路線"
        assert session.total_stops == 4
        assert "倉儲區" not in session.stops

    def test_create_session_with_language(self):
        session = TourSession("test-3", language="en")
        intro = session.get_current_intro()
        assert "Welcome" in intro or "Lobby" in intro or "Stop" in intro

    def test_advance_step(self):
        session = TourSession("test-4")
        assert session.current_area == "大廳"

        result = session.advance()
        assert result["status"] == "active"
        assert "大廳" in result["visited_areas"]
        assert session.current_step == 1
        assert session.current_area == "組裝線A"

    def test_full_tour(self):
        session = TourSession("test-5")
        total = session.total_stops

        for i in range(total):
            result = session.advance()

        assert result["status"] == "completed"
        assert session.completed is True
        assert len(session.visited_areas) == total

    def test_advance_after_completion(self):
        session = TourSession("test-6")
        for _ in range(session.total_stops):
            session.advance()

        result = session.advance()
        assert result["status"] == "completed"

    def test_progress_percent(self):
        session = TourSession("test-7")
        assert session.progress_percent == 0.0

        session.advance()
        assert session.progress_percent > 0

        for _ in range(session.total_stops - 1):
            session.advance()
        assert session.progress_percent == 100.0

    def test_to_dict(self):
        session = TourSession("test-8")
        d = session.to_dict()
        assert "session_id" in d
        assert "route_name" in d
        assert "current_step" in d
        assert "stops" in d
        assert "visited_areas" in d
        assert "progress_percent" in d
        assert "completed" in d


class TestTourManager:
    """TourManager 測試"""

    def test_start_tour(self):
        manager = TourManager()
        result = manager.start_tour("s1")
        assert result["status"] == "started"
        assert "message" in result
        assert result["current_step"] == 0

    def test_next_stop(self):
        manager = TourManager()
        manager.start_tour("s2")
        result = manager.next_stop("s2")
        assert result["status"] == "active"
        assert result["current_step"] == 1

    def test_next_stop_no_session(self):
        manager = TourManager()
        result = manager.next_stop("nonexistent")
        assert result["status"] == "error"

    def test_get_status(self):
        manager = TourManager()
        manager.start_tour("s3")
        status = manager.get_status("s3")
        assert status is not None
        assert "current_step" in status

    def test_get_status_nonexistent(self):
        manager = TourManager()
        assert manager.get_status("nope") is None

    def test_get_available_routes(self):
        manager = TourManager()
        routes = manager.get_available_routes()
        assert len(routes) >= 2
        names = [r["name"] for r in routes]
        assert "標準導覽路線" in names
        assert "快速導覽路線" in names

    def test_multilingual_tour(self):
        manager = TourManager()
        result_zh = manager.start_tour("zh-tour", language="zh-TW")
        result_en = manager.start_tour("en-tour", language="en")
        result_ja = manager.start_tour("ja-tour", language="ja")

        assert "歡迎" in result_zh["message"]
        assert "Welcome" in result_en["message"]
        assert "ようこそ" in result_ja["message"]
