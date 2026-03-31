"""
db.py - SQLite 對話歷史持久化模組
提供對話記錄的儲存、查詢和管理功能
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
    conn = get_connection()
    try:
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
    finally:
        conn.close()


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    """取得某 session 的對話歷史"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT role, content, agent_name, created_at
               FROM conversations
               WHERE session_id = ?
               ORDER BY created_at DESC
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
            for r in reversed(rows)
        ]
    finally:
        conn.close()


def get_all_sessions(limit: int = 100) -> list[dict]:
    """取得所有 session 的摘要"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT session_id, language, total_messages, first_message_at, last_message_at
               FROM session_metadata
               ORDER BY last_message_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_session(session_id: str):
    """刪除某 session 的所有資料"""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM tour_sessions WHERE session_id = ?", (session_id,))
        conn.execute(
            "DELETE FROM session_metadata WHERE session_id = ?", (session_id,)
        )
        conn.commit()
    finally:
        conn.close()


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
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    try:
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
    finally:
        conn.close()


def get_tour_state(session_id: str) -> dict | None:
    """取得導覽進度"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM tour_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["visited_areas"] = json.loads(d["visited_areas"])
            return d
        return None
    finally:
        conn.close()


def get_stats() -> dict:
    """取得統計資訊"""
    conn = get_connection()
    try:
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
    finally:
        conn.close()
