"""
conftest.py - pytest 共用 fixtures
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 確保專案根目錄在 path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 設定測試用環境變數（避免真正呼叫 API）
os.environ.setdefault("GROQ_API_KEY", "test-key-for-testing")
os.environ["SKIP_RAG"] = "1"  # 跳過 RAG 初始化（避免下載 embedding model）


@pytest.fixture
def knowledge_data():
    """提供測試用的知識庫資料"""
    return {
        "areas": [
            {
                "id": "lobby",
                "name": "大廳",
                "description": "歡迎來到工廠！大廳設有訪客登記處。",
                "safety_notes": ["請配戴訪客證", "手機請切靜音"],
                "technical_specs": {},
                "tour_order": 0,
            },
            {
                "id": "assembly_a",
                "name": "組裝線A",
                "description": "主要 PCB 組裝產線，日產能可達 5,000 片。",
                "safety_notes": ["必須穿戴防靜電手環", "禁止攜帶飲料"],
                "technical_specs": {
                    "daily_capacity": 5000,
                    "equipment": ["SMT 貼片機 x3"],
                    "temperature": "25±2°C",
                },
                "tour_order": 1,
            },
        ],
        "routes": [
            {
                "name": "標準導覽路線",
                "duration": "約 45 分鐘",
                "stops": ["大廳", "組裝線A"],
                "description": "完整參觀所有主要區域",
            }
        ],
        "emergency": {
            "exit_locations": "綠色標示",
            "assembly_point": "大門口停車場",
            "emergency_contact": "119",
            "aed_location": "一樓大廳",
            "first_aid": "每區域出入口旁",
        },
    }


@pytest.fixture
def faq_data():
    """提供測試用 FAQ 資料"""
    return [
        {
            "id": "visiting_hours",
            "question": "參觀時間是什麼時候？",
            "answer": "週一至週五 09:00-16:00",
            "category": "visit_info",
            "keywords": ["時間", "開放", "hours"],
        },
        {
            "id": "parking",
            "question": "有停車場嗎？",
            "answer": "有訪客專用停車場",
            "category": "visit_info",
            "keywords": ["停車", "parking"],
        },
    ]


@pytest.fixture
def temp_db():
    """提供臨時測試資料庫"""
    with tempfile.TemporaryDirectory() as tmpdir:
        import db

        original_dir = db.DB_DIR
        original_path = db.DB_PATH
        db.DB_DIR = Path(tmpdir)
        db.DB_PATH = Path(tmpdir) / "test.db"
        db.init_db()
        yield db
        db.DB_DIR = original_dir
        db.DB_PATH = original_path
