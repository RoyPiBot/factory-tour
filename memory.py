"""
memory.py - 訪客記憶系統
追蹤訪客的歷史互動、興趣偏好和參觀記錄，為 LLM 提供個人化上下文。
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# 資料庫路徑（與 db.py 共用同一個資料庫）
DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "factory_tour.db"

# 興趣關鍵字對照表：關鍵字 -> 興趣標籤
INTEREST_KEYWORDS: dict[str, str] = {
    # 製造與生產
    "SMT": "SMT 貼片技術",
    "貼片": "SMT 貼片技術",
    "回焊": "回焊製程",
    "PCB": "PCB 電路板",
    "電路板": "PCB 電路板",
    "產線": "生產線管理",
    "產能": "產能規劃",
    "自動化": "自動化製造",
    "組裝": "組裝製程",
    # 品質管理
    "品管": "品質管理",
    "品質": "品質管理",
    "AOI": "AOI 光學檢測",
    "X-ray": "X-ray 檢測",
    "不良率": "良率管控",
    "良率": "良率管控",
    "檢測": "檢測技術",
    "測試": "功能測試",
    # 倉儲與物流
    "倉庫": "倉儲管理",
    "倉儲": "倉儲管理",
    "WMS": "WMS 系統",
    "FIFO": "庫存管理",
    "物流": "物流管理",
    "棧板": "倉儲容量",
    # 安全與環境
    "安全": "工廠安全",
    "防靜電": "ESD 防護",
    "靜電": "ESD 防護",
    "無塵": "無塵室規範",
    "溫濕度": "環境控制",
    "溫度": "環境控制",
    "濕度": "環境控制",
    # 商業合作
    "合作": "商業合作",
    "報價": "商業合作",
    "交期": "交期管理",
    "客製": "客製化服務",
    "OEM": "OEM 代工",
    "ODM": "ODM 設計代工",
    # 技術
    "設備": "設備技術",
    "機台": "設備技術",
    "IoT": "IoT 物聯網",
    "AI": "AI 應用",
    "數據": "數據分析",
}

# thread-local storage 管理連線
_local = threading.local()


def _get_db() -> sqlite3.Connection:
    """取得當前 thread 的資料庫連線"""
    if not hasattr(_local, "conn") or _local.conn is None:
        DB_DIR.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def _init_visitor_tables():
    """初始化訪客記憶相關的資料表（若舊 schema 不符則重建）"""
    conn = _get_db()

    # 檢查既有 schema 是否相容
    existing_cols = {
        r[1]
        for r in conn.execute("PRAGMA table_info(visitor_profiles)").fetchall()
    }
    if existing_cols and "first_visit_at" not in existing_cols:
        conn.executescript("""
            DROP TABLE IF EXISTS visitor_profiles;
            DROP TABLE IF EXISTS visitor_area_visits;
        """)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS visitor_profiles (
            session_id TEXT PRIMARY KEY,
            visit_count INTEGER DEFAULT 1,
            interests TEXT DEFAULT '[]',
            preferences TEXT DEFAULT '{}',
            visited_areas TEXT DEFAULT '[]',
            quiz_scores TEXT DEFAULT '{}',
            first_visit_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_visit_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS visitor_area_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            area_id TEXT NOT NULL,
            visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES visitor_profiles(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_area_visits_session
            ON visitor_area_visits(session_id, visited_at);
    """)
    conn.commit()


# 模組載入時初始化資料表
_init_visitor_tables()


class VisitorMemory:
    """訪客記憶系統 — 追蹤訪客互動歷史並產生個人化上下文"""

    def get_or_create_profile(self, session_id: str) -> dict:
        """
        取得訪客資料，若不存在則建立新的。
        回傳 dict 包含訪客的完整 profile。
        """
        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM visitor_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if row:
            profile = dict(row)
            profile["interests"] = json.loads(profile["interests"])
            profile["preferences"] = json.loads(profile["preferences"])
            profile["visited_areas"] = json.loads(profile["visited_areas"])
            profile["quiz_scores"] = json.loads(profile["quiz_scores"])
            return profile

        # 建立新訪客
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO visitor_profiles
               (session_id, first_visit_at, last_visit_at)
               VALUES (?, ?, ?)""",
            (session_id, now, now),
        )
        conn.commit()

        return {
            "session_id": session_id,
            "visit_count": 1,
            "interests": [],
            "preferences": {},
            "visited_areas": [],
            "quiz_scores": {},
            "first_visit_at": now,
            "last_visit_at": now,
            "notes": "",
        }

    def get_context_prompt(self, session_id: str) -> str:
        """
        產生一段摘要文字，描述此訪客的歷史資訊，
        可直接嵌入 LLM 的 system prompt 或 context 中。
        """
        profile = self.get_or_create_profile(session_id)

        parts: list[str] = []
        parts.append("=== 訪客記憶 ===")

        # 造訪次數
        count = profile["visit_count"]
        if count > 1:
            parts.append(f"這位訪客是回訪者，已來過 {count} 次。")
        else:
            parts.append("這位訪客是第一次來訪。")

        # 已參觀區域
        visited = profile["visited_areas"]
        if visited:
            areas_str = "、".join(visited)
            parts.append(f"已參觀區域：{areas_str}")
        else:
            parts.append("尚未參觀任何區域。")

        # 興趣偏好
        interests = profile["interests"]
        if interests:
            interests_str = "、".join(interests)
            parts.append(f"感興趣的主題：{interests_str}")

        # 偏好設定
        preferences = profile["preferences"]
        if preferences:
            pref_items = [f"{k}: {v}" for k, v in preferences.items()]
            parts.append(f"偏好設定：{'；'.join(pref_items)}")

        # 測驗成績
        quiz_scores = profile["quiz_scores"]
        if quiz_scores:
            score_items = [
                f"{area}: {score}" for area, score in quiz_scores.items()
            ]
            parts.append(f"測驗成績：{'；'.join(score_items)}")

        # 備註
        if profile["notes"]:
            parts.append(f"備註：{profile['notes']}")

        parts.append("=== 結束 ===")
        return "\n".join(parts)

    def update_from_conversation(
        self, session_id: str, user_msg: str, ai_reply: str
    ) -> list[str]:
        """
        從對話中擷取訪客的興趣與偏好（關鍵字比對）。
        回傳新發現的興趣列表。
        """
        profile = self.get_or_create_profile(session_id)
        current_interests: list[str] = profile["interests"]
        new_interests: list[str] = []

        # 合併使用者訊息與 AI 回覆進行分析
        combined_text = f"{user_msg} {ai_reply}"

        for keyword, interest_label in INTEREST_KEYWORDS.items():
            if keyword in combined_text and interest_label not in current_interests:
                current_interests.append(interest_label)
                new_interests.append(interest_label)

        # 偵測語言偏好
        preferences = profile["preferences"]
        if any(w in user_msg for w in ["English", "english", "英文"]):
            preferences["language"] = "en"
        elif any(w in user_msg for w in ["日本語", "Japanese", "日文"]):
            preferences["language"] = "ja"

        # 偵測詳細程度偏好
        if any(w in user_msg for w in ["詳細", "深入", "技術細節", "more detail"]):
            preferences["detail_level"] = "detailed"
        elif any(w in user_msg for w in ["簡單", "簡短", "quick", "簡要"]):
            preferences["detail_level"] = "brief"

        # 寫回資料庫
        conn = _get_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE visitor_profiles
               SET interests = ?, preferences = ?, last_visit_at = ?
               WHERE session_id = ?""",
            (
                json.dumps(current_interests, ensure_ascii=False),
                json.dumps(preferences, ensure_ascii=False),
                now,
                session_id,
            ),
        )
        conn.commit()

        return new_interests

    def record_area_visit(self, session_id: str, area_id: str):
        """
        記錄訪客參觀了某個區域。
        同時更新 profile 中的 visited_areas 列表（不重複）。
        """
        profile = self.get_or_create_profile(session_id)
        conn = _get_db()
        now = datetime.now(timezone.utc).isoformat()

        # 寫入明細記錄（每次造訪都記）
        conn.execute(
            """INSERT INTO visitor_area_visits (session_id, area_id, visited_at)
               VALUES (?, ?, ?)""",
            (session_id, area_id, now),
        )

        # 更新 profile 的 visited_areas（去重）
        visited: list[str] = profile["visited_areas"]
        if area_id not in visited:
            visited.append(area_id)
            conn.execute(
                """UPDATE visitor_profiles
                   SET visited_areas = ?, last_visit_at = ?
                   WHERE session_id = ?""",
                (
                    json.dumps(visited, ensure_ascii=False),
                    now,
                    session_id,
                ),
            )

        conn.commit()

    def increment_visit_count(self, session_id: str) -> int:
        """
        為回訪者遞增造訪次數。
        回傳更新後的造訪次數。
        """
        profile = self.get_or_create_profile(session_id)
        new_count = profile["visit_count"] + 1

        conn = _get_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE visitor_profiles
               SET visit_count = ?, last_visit_at = ?
               WHERE session_id = ?""",
            (new_count, now, session_id),
        )
        conn.commit()

        return new_count

    def record_quiz_score(
        self, session_id: str, area_id: str, score: str
    ):
        """記錄某區域的測驗成績（額外功能）"""
        profile = self.get_or_create_profile(session_id)
        quiz_scores: dict = profile["quiz_scores"]
        quiz_scores[area_id] = score

        conn = _get_db()
        conn.execute(
            """UPDATE visitor_profiles
               SET quiz_scores = ?
               WHERE session_id = ?""",
            (json.dumps(quiz_scores, ensure_ascii=False), session_id),
        )
        conn.commit()

    def get_area_visit_history(self, session_id: str) -> list[dict]:
        """取得訪客的區域造訪明細（含時間戳記）"""
        conn = _get_db()
        rows = conn.execute(
            """SELECT area_id, visited_at
               FROM visitor_area_visits
               WHERE session_id = ?
               ORDER BY visited_at ASC""",
            (session_id,),
        ).fetchall()
        return [{"area_id": r["area_id"], "visited_at": r["visited_at"]} for r in rows]
