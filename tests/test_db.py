"""
test_db.py - 測試 SQLite 資料庫模組
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatabaseMessages:
    """對話訊息 CRUD 測試"""

    def test_save_and_get_message(self, temp_db):
        temp_db.save_message("test-session", "user", "Hello", "zh-TW")
        temp_db.save_message("test-session", "assistant", "Hi!", "zh-TW", "tour_guide")

        history = temp_db.get_history("test-session")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["agent_name"] == "tour_guide"

    def test_get_history_empty(self, temp_db):
        history = temp_db.get_history("nonexistent")
        assert len(history) == 0

    def test_get_history_limit(self, temp_db):
        for i in range(10):
            temp_db.save_message("limit-test", "user", f"msg-{i}")

        history = temp_db.get_history("limit-test", limit=5)
        assert len(history) == 5

    def test_session_metadata(self, temp_db):
        temp_db.save_message("meta-test", "user", "Hello", "en")
        temp_db.save_message("meta-test", "assistant", "Hi", "en")

        sessions = temp_db.get_all_sessions()
        assert len(sessions) >= 1
        meta = next(s for s in sessions if s["session_id"] == "meta-test")
        assert meta["total_messages"] == 2
        assert meta["language"] == "en"

    def test_delete_session(self, temp_db):
        temp_db.save_message("delete-me", "user", "Bye")
        temp_db.delete_session("delete-me")

        history = temp_db.get_history("delete-me")
        assert len(history) == 0
        sessions = temp_db.get_all_sessions()
        assert not any(s["session_id"] == "delete-me" for s in sessions)


class TestTourState:
    """導覽狀態持久化測試"""

    def test_save_and_get_tour_state(self, temp_db):
        temp_db.save_tour_state(
            "tour-1", "標準導覽路線", 1, ["大廳"], "zh-TW", False
        )

        state = temp_db.get_tour_state("tour-1")
        assert state is not None
        assert state["route_name"] == "標準導覽路線"
        assert state["current_step"] == 1
        assert state["visited_areas"] == ["大廳"]
        assert not state["completed"]

    def test_update_tour_state(self, temp_db):
        temp_db.save_tour_state("tour-2", "快速導覽路線", 0, [], "en")
        temp_db.save_tour_state(
            "tour-2", "快速導覽路線", 2, ["大廳", "組裝線A"], "en", True
        )

        state = temp_db.get_tour_state("tour-2")
        assert state["current_step"] == 2
        assert len(state["visited_areas"]) == 2
        assert state["completed"]

    def test_get_tour_state_nonexistent(self, temp_db):
        state = temp_db.get_tour_state("no-such-tour")
        assert state is None


class TestStats:
    """統計資訊測試"""

    def test_get_stats(self, temp_db):
        temp_db.save_message("s1", "user", "Hello", "zh-TW")
        temp_db.save_message("s2", "user", "Hello", "en")
        temp_db.save_tour_state("s1", "標準導覽路線", 0, [])
        temp_db.save_tour_state("s3", "快速導覽路線", 4, ["a", "b"], completed=True)

        stats = temp_db.get_stats()
        assert stats["total_sessions"] >= 2
        assert stats["total_messages"] >= 2
        assert stats["total_tours"] >= 2
        assert stats["completed_tours"] >= 1
        assert "zh-TW" in stats["language_distribution"]
        assert "en" in stats["language_distribution"]
