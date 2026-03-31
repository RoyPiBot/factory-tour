"""
db.py - SQLite 持久化模組 v2.0
提供對話記錄、回饋評分、測驗、訪客記憶、分析事件的儲存與管理

v2.0 新增：
  - feedback 表（導覽評分 + 留言）
  - quiz_results 表（測驗答題紀錄）
  - visitor_profiles 表（跨 session 訪客記憶）
  - analytics_events 表（行為分析事件）
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# 資料庫路徑
DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "factory_tour.db"


def get_connection() -> sqlite3.Connection:
    """取得 SQLite 連線（每個 thread 獨立）"""
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# 使用 thread-local storage 管理連線
_local = threading.local()


def get_db() -> sqlite3.Connection:
    """取得當前 thread 的資料庫連線"""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = get_connection()
    return _local.conn


def init_db():
    """初始化資料庫 schema"""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            language TEXT DEFAULT 'zh-TW',
            agent_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations(session_id, created_at);

        CREATE TABLE IF NOT EXISTS tour_sessions (
            session_id TEXT PRIMARY KEY,
            route_name TEXT,
            current_step INTEGER DEFAULT 0,
            visited_areas TEXT DEFAULT '[]',
            language TEXT DEFAULT 'zh-TW',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed BOOLEAN DEFAULT FALSE
        );

        CREATE TABLE IF NOT EXISTS session_metadata (
            session_id TEXT PRIMARY KEY,
            language TEXT DEFAULT 'zh-TW',
            total_messages INTEGER DEFAULT 0,
            first_message_at TIMESTAMP,
            last_message_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            comment TEXT,
            areas_visited TEXT DEFAULT '[]',
            quiz_score INTEGER DEFAULT 0,
            language TEXT DEFAULT 'zh-TW',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);

        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            area_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            answer TEXT NOT NULL,
            correct BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_quiz_session ON quiz_results(session_id);

        CREATE TABLE IF NOT EXISTS visitor_profiles (
            session_id TEXT PRIMARY KEY,
            visitor_name TEXT,
            interests TEXT DEFAULT '[]',
            preferences TEXT DEFAULT '{}',
            visit_count INTEGER DEFAULT 1,
            last_areas_visited TEXT DEFAULT '[]',
            quiz_total_score INTEGER DEFAULT 0,
            feedback_given BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analytics_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_type TEXT NOT NULL,
            event_data TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_analytics_type ON analytics_events(event_type, created_at);
        CREATE INDEX IF NOT EXISTS idx_analytics_session ON analytics_events(session_id);
    """)
    conn.commit()
    conn.close()


def save_message(
    session_id: str,
    role: str,
    content: str,
    language: str = "zh-TW",
    agent_name: str | None = None,
):
    """儲存一筆對話訊息"""
    conn = get_db()
    conn.execute(
        """INSERT INTO conversations (session_id, role, content, language, agent_name)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, role, content, language, agent_name),
    )
    # 更新 session metadata
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO session_metadata (session_id, language, total_messages, first_message_at, last_message_at)
           VALUES (?, ?, 1, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               total_messages = total_messages + 1,
               last_message_at = ?,
               language = ?""",
        (session_id, language, now, now, now, language),
    )
    conn.commit()


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    """取得某 session 的對話歷史（按時間正序）"""
    conn = get_db()
    rows = conn.execute(
        """SELECT role, content, agent_name, created_at
           FROM conversations
           WHERE session_id = ?
           ORDER BY created_at ASC
           LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "agent_name": r["agent_name"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def get_all_sessions(limit: int = 100) -> list[dict]:
    """取得所有 session 的摘要"""
    conn = get_db()
    rows = conn.execute(
        """SELECT session_id, language, total_messages, first_message_at, last_message_at
           FROM session_metadata
           ORDER BY last_message_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str):
    """刪除某 session 的所有資料"""
    conn = get_db()
    conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM tour_sessions WHERE session_id = ?", (session_id,))
    conn.execute(
        "DELETE FROM session_metadata WHERE session_id = ?", (session_id,)
    )
    conn.commit()


# ─── Tour Session 管理 ───
def save_tour_state(
    session_id: str,
    route_name: str,
    current_step: int,
    visited_areas: list[str],
    language: str = "zh-TW",
    completed: bool = False,
):
    """儲存/更新導覽進度"""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO tour_sessions (session_id, route_name, current_step, visited_areas, language, completed, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               current_step = ?,
               visited_areas = ?,
               completed = ?,
               updated_at = ?""",
        (
            session_id,
            route_name,
            current_step,
            json.dumps(visited_areas, ensure_ascii=False),
            language,
            completed,
            now,
            current_step,
            json.dumps(visited_areas, ensure_ascii=False),
            completed,
            now,
        ),
    )
    conn.commit()


def get_tour_state(session_id: str) -> dict | None:
    """取得導覽進度"""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM tour_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row:
        d = dict(row)
        d["visited_areas"] = json.loads(d["visited_areas"])
        return d
    return None


def get_stats() -> dict:
    """取得統計資訊"""
    conn = get_db()
    total_sessions = conn.execute(
        "SELECT COUNT(*) FROM session_metadata"
    ).fetchone()[0]
    total_messages = conn.execute(
        "SELECT COALESCE(SUM(total_messages), 0) FROM session_metadata"
    ).fetchone()[0]
    total_tours = conn.execute(
        "SELECT COUNT(*) FROM tour_sessions"
    ).fetchone()[0]
    completed_tours = conn.execute(
        "SELECT COUNT(*) FROM tour_sessions WHERE completed = TRUE"
    ).fetchone()[0]
    lang_stats = conn.execute(
        "SELECT language, COUNT(*) as cnt FROM session_metadata GROUP BY language"
    ).fetchall()
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_tours": total_tours,
        "completed_tours": completed_tours,
        "language_distribution": {r["language"]: r["cnt"] for r in lang_stats},
    }


# ─── 回饋評分 ───

def save_feedback(
    session_id: str,
    rating: int,
    comment: str = "",
    areas_visited: list[str] | None = None,
    quiz_score: int = 0,
    language: str = "zh-TW",
) -> int:
    """儲存導覽回饋評分，回傳 feedback id"""
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO feedback (session_id, rating, comment, areas_visited, quiz_score, language)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            rating,
            comment,
            json.dumps(areas_visited or [], ensure_ascii=False),
            quiz_score,
            language,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_feedback_stats() -> dict:
    """取得回饋統計"""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    if total == 0:
        return {"total": 0, "average_rating": 0, "distribution": {}}
    avg = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0]
    dist = conn.execute(
        "SELECT rating, COUNT(*) as cnt FROM feedback GROUP BY rating ORDER BY rating"
    ).fetchall()
    recent = conn.execute(
        """SELECT session_id, rating, comment, created_at
           FROM feedback ORDER BY created_at DESC LIMIT 10"""
    ).fetchall()
    return {
        "total": total,
        "average_rating": round(avg, 2),
        "distribution": {r["rating"]: r["cnt"] for r in dist},
        "recent": [dict(r) for r in recent],
    }


# ─── 測驗答題 ───

def save_quiz_answer(
    session_id: str, area_id: str, question_id: str, answer: str, correct: bool
):
    """儲存測驗答案"""
    conn = get_db()
    conn.execute(
        """INSERT INTO quiz_results (session_id, area_id, question_id, answer, correct)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, area_id, question_id, answer, correct),
    )
    conn.commit()


def get_quiz_score(session_id: str) -> dict:
    """取得某 session 的測驗成績"""
    conn = get_db()
    total = conn.execute(
        "SELECT COUNT(*) FROM quiz_results WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    correct = conn.execute(
        "SELECT COUNT(*) FROM quiz_results WHERE session_id = ? AND correct = TRUE",
        (session_id,),
    ).fetchone()[0]
    by_area = conn.execute(
        """SELECT area_id,
                  COUNT(*) as total,
                  SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct
           FROM quiz_results WHERE session_id = ? GROUP BY area_id""",
        (session_id,),
    ).fetchall()
    return {
        "total_answered": total,
        "total_correct": correct,
        "score_pct": round(correct / total * 100, 1) if total > 0 else 0,
        "by_area": {r["area_id"]: {"total": r["total"], "correct": r["correct"]} for r in by_area},
    }


# ─── 訪客記憶 ───

def get_visitor_profile(session_id: str) -> dict | None:
    """取得訪客資料"""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM visitor_profiles WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row:
        d = dict(row)
        d["interests"] = json.loads(d["interests"])
        d["preferences"] = json.loads(d["preferences"])
        d["last_areas_visited"] = json.loads(d["last_areas_visited"])
        return d
    return None


def save_visitor_profile(
    session_id: str,
    visitor_name: str = "",
    interests: list[str] | None = None,
    preferences: dict | None = None,
    areas_visited: list[str] | None = None,
):
    """建立或更新訪客資料"""
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO visitor_profiles
               (session_id, visitor_name, interests, preferences, last_areas_visited, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               visitor_name = COALESCE(NULLIF(?, ''), visitor_name),
               interests = ?,
               preferences = ?,
               last_areas_visited = ?,
               visit_count = visit_count + 1,
               updated_at = ?""",
        (
            session_id,
            visitor_name or "",
            json.dumps(interests or [], ensure_ascii=False),
            json.dumps(preferences or {}, ensure_ascii=False),
            json.dumps(areas_visited or [], ensure_ascii=False),
            now,
            visitor_name or "",
            json.dumps(interests or [], ensure_ascii=False),
            json.dumps(preferences or {}, ensure_ascii=False),
            json.dumps(areas_visited or [], ensure_ascii=False),
            now,
        ),
    )
    conn.commit()


# ─── 分析事件 ───

def log_event(session_id: str, event_type: str, event_data: dict | None = None):
    """記錄分析事件"""
    conn = get_db()
    conn.execute(
        """INSERT INTO analytics_events (session_id, event_type, event_data)
           VALUES (?, ?, ?)""",
        (session_id, event_type, json.dumps(event_data or {}, ensure_ascii=False)),
    )
    conn.commit()


def get_analytics_summary() -> dict:
    """取得分析摘要"""
    conn = get_db()
    total_events = conn.execute("SELECT COUNT(*) FROM analytics_events").fetchone()[0]
    event_types = conn.execute(
        """SELECT event_type, COUNT(*) as cnt
           FROM analytics_events GROUP BY event_type ORDER BY cnt DESC"""
    ).fetchall()
    # 最常問的問題（從 chat_message 事件取）
    top_questions = conn.execute(
        """SELECT event_data FROM analytics_events
           WHERE event_type = 'chat_message' ORDER BY created_at DESC LIMIT 20"""
    ).fetchall()
    return {
        "total_events": total_events,
        "event_distribution": {r["event_type"]: r["cnt"] for r in event_types},
        "recent_questions": [
            json.loads(r["event_data"]).get("message", "") for r in top_questions
        ],
    }
