"""
tour_flow.py - 互動式導覽流程管理
按步驟帶訪客走完整路線，每站自動介紹
"""
import json
from pathlib import Path
from typing import Optional

from i18n import get_prompt

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


def _load_knowledge() -> dict:
    areas_file = KNOWLEDGE_DIR / "areas.json"
    if areas_file.exists():
        with open(areas_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"areas": [], "routes": [], "emergency": {}}


KNOWLEDGE = _load_knowledge()
AREA_MAP = {area["name"]: area for area in KNOWLEDGE.get("areas", [])}
ROUTE_MAP = {route["name"]: route for route in KNOWLEDGE.get("routes", [])}

# ─── 各站介紹文字（多語言）───
AREA_INTROS = {
    "大廳": {
        "zh-TW": (
            "🏢 歡迎來到大廳！\n\n"
            "這裡是訪客登記處，也是我們的起點。請在此領取訪客證和安全裝備。\n"
            "牆上展示了公司從創立至今的發展歷程。\n\n"
            "⚠️ 安全提醒：請配戴訪客證，手機請切靜音。\n\n"
            "準備好了嗎？輸入「下一站」繼續導覽！"
        ),
        "en": (
            "🏢 Welcome to the Lobby!\n\n"
            "This is the visitor registration area and our starting point. "
            "Please pick up your visitor badge and safety equipment here.\n"
            "The wall displays our company's history from founding to present.\n\n"
            "⚠️ Safety reminder: Please wear your visitor badge and silence your phone.\n\n"
            "Ready? Type 'next' to continue the tour!"
        ),
        "ja": (
            "🏢 ロビーへようこそ！\n\n"
            "ここは訪問者受付エリアで、見学の出発点です。\n"
            "訪問者バッジと安全装備をこちらでお受け取りください。\n\n"
            "⚠️ 安全注意：訪問者バッジを着用し、携帯電話はマナーモードにしてください。\n\n"
            "準備はいいですか？「次へ」と入力して見学を続けましょう！"
        ),
    },
    "組裝線A": {
        "zh-TW": (
            "⚙️ 歡迎來到組裝線A！\n\n"
            "這是我們的主力 PCB 組裝產線，日產能可達 5,000 片電路板。\n\n"
            "📋 主要設備：\n"
            "  • 3 台高速 SMT 貼片機（每小時 80,000 個元件）\n"
            "  • 1 台無鉛回焊爐\n"
            "  • 2 台 AOI 自動光學檢測\n\n"
            "🌡️ 環境控制：溫度 25±2°C，濕度 45-65% RH\n\n"
            "⚠️ 安全規範：\n"
            "  • 必須穿戴防靜電手環\n"
            "  • 必須配戴護目鏡\n"
            "  • 禁止攜帶飲料和食物\n"
            "  • 請勿觸碰運行中的設備"
        ),
        "en": (
            "⚙️ Welcome to Assembly Line A!\n\n"
            "This is our main PCB assembly line with a daily capacity of 5,000 boards.\n\n"
            "📋 Key Equipment:\n"
            "  • 3 high-speed SMT placement machines (80,000 components/hour)\n"
            "  • 1 lead-free reflow oven\n"
            "  • 2 AOI inspection machines\n\n"
            "🌡️ Environment: 25±2°C, 45-65% RH\n\n"
            "⚠️ Safety Rules:\n"
            "  • ESD wristband required\n"
            "  • Safety goggles required\n"
            "  • No food or drinks\n"
            "  • Do not touch running equipment"
        ),
        "ja": (
            "⚙️ 組立ラインAへようこそ！\n\n"
            "PCB組立の主力ラインです。日産能力は5,000枚。\n\n"
            "📋 主要設備：SMT実装機×3、リフロー炉×1、AOI検査機×2\n"
            "🌡️ 環境：25±2°C、45-65% RH\n\n"
            "⚠️ 安全規則：ESDリストバンド着用、保護メガネ着用、飲食禁止"
        ),
    },
    "品管室": {
        "zh-TW": (
            "🔍 歡迎來到品管室！\n\n"
            "品質管制中心，我們的不良率控制在 0.1% 以下（業界平均 0.5%）。\n\n"
            "📋 檢測設備：\n"
            "  • AOI 自動光學檢測機\n"
            "  • X-ray 檢測設備\n"
            "  • 4 座功能測試站\n\n"
            "每片電路板必須通過至少 3 道檢測程序。\n\n"
            "⚠️ 安全提醒：請穿無塵衣，X-ray 區域請勿靠近。"
        ),
        "en": (
            "🔍 Welcome to the QC Room!\n\n"
            "Our quality control center maintains a defect rate below 0.1% (industry avg: 0.5%).\n\n"
            "📋 Equipment: AOI, X-ray, and 4 test stations.\n"
            "Every board passes at least 3 inspection stages.\n\n"
            "⚠️ Safety: Wear clean room gear. Stay away from X-ray zone."
        ),
        "ja": (
            "🔍 品質管理室へようこそ！\n\n"
            "不良率0.1%以下（業界平均0.5%）。AOI、X線、テストステーション×4。\n"
            "⚠️ クリーンルーム着用、X線エリア立入禁止。"
        ),
    },
    "倉儲區": {
        "zh-TW": (
            "📦 歡迎來到倉儲區！\n\n"
            "自動化立體倉庫，可容納約 10,000 個棧板位。\n\n"
            "📋 管理系統：\n"
            "  • WMS 倉儲管理系統\n"
            "  • FIFO 先進先出原則\n"
            "  • 24 小時溫濕度監控\n\n"
            "⚠️ 安全提醒：\n"
            "  • 注意堆高機通行，請走標示的人行通道\n"
            "  • 不要在貨架間停留\n"
            "  • 聽從現場人員指揮"
        ),
        "en": (
            "📦 Welcome to the Warehouse!\n\n"
            "Automated storage with ~10,000 pallet positions.\n"
            "WMS system, FIFO management, 24/7 monitoring.\n\n"
            "⚠️ Watch for forklifts, stay in marked corridors."
        ),
        "ja": (
            "📦 倉庫エリアへようこそ！\n\n"
            "自動倉庫、約10,000パレット。WMS、FIFO管理、24時間監視。\n"
            "⚠️ フォークリフトに注意、通路を歩いてください。"
        ),
    },
    "會議室": {
        "zh-TW": (
            "🪑 歡迎來到會議室！\n\n"
            "導覽的最後一站。這裡備有茶水和休息空間。\n"
            "您可以在此提問、討論合作事宜，或回顧今天的參觀內容。\n\n"
            "會議室配備投影設備和視訊會議系統。\n\n"
            "🎉 感謝您的參觀！如有任何問題，隨時可以詢問我們的 AI 導覽系統。"
        ),
        "en": (
            "🪑 Welcome to the Conference Room!\n\n"
            "This is our final stop. Refreshments are available here.\n"
            "Feel free to ask questions or discuss collaboration.\n\n"
            "🎉 Thank you for visiting! Feel free to ask our AI guide anything."
        ),
        "ja": (
            "🪑 会議室へようこそ！\n\n"
            "最後の見学地です。お茶と休憩スペースをご用意しています。\n\n"
            "🎉 ご見学ありがとうございました！"
        ),
    },
}


class TourSession:
    """單一導覽 session 的狀態管理"""

    def __init__(
        self,
        session_id: str,
        route_name: str = "標準導覽路線",
        language: str = "zh-TW",
    ):
        self.session_id = session_id
        self.language = language

        route = ROUTE_MAP.get(route_name)
        if not route:
            route = KNOWLEDGE["routes"][0] if KNOWLEDGE["routes"] else None

        self.route_name = route["name"] if route else "標準導覽路線"
        self.stops: list[str] = route["stops"] if route else []
        self.duration = route["duration"] if route else "N/A"
        self.current_step: int = 0
        self.visited_areas: list[str] = []
        self.completed: bool = False

    @property
    def current_area(self) -> str | None:
        if 0 <= self.current_step < len(self.stops):
            return self.stops[self.current_step]
        return None

    @property
    def total_stops(self) -> int:
        return len(self.stops)

    @property
    def progress_percent(self) -> float:
        if not self.stops:
            return 0.0
        return len(self.visited_areas) / len(self.stops) * 100

    def get_current_intro(self) -> str:
        """取得當前站點的介紹文字"""
        area_name = self.current_area
        if not area_name:
            return self._completion_message()

        intros = AREA_INTROS.get(area_name, {})
        intro = intros.get(self.language, intros.get("zh-TW", f"歡迎來到{area_name}！"))

        # 加上進度資訊
        progress = self._progress_header()
        return f"{progress}\n\n{intro}"

    def advance(self) -> dict:
        """前進到下一站"""
        if self.completed:
            return {
                "status": "completed",
                "message": self._completion_message(),
                "current_step": self.current_step,
                "visited_areas": self.visited_areas,
            }

        # 記錄當前區域為已參觀
        current = self.current_area
        if current and current not in self.visited_areas:
            self.visited_areas.append(current)

        # 前進
        self.current_step += 1

        if self.current_step >= len(self.stops):
            self.completed = True
            return {
                "status": "completed",
                "message": self._completion_message(),
                "current_step": self.current_step,
                "visited_areas": self.visited_areas,
            }

        return {
            "status": "active",
            "message": self.get_current_intro(),
            "current_step": self.current_step,
            "current_area": self.current_area,
            "visited_areas": self.visited_areas,
            "remaining": len(self.stops) - self.current_step,
        }

    def _progress_header(self) -> str:
        step = self.current_step + 1
        total = len(self.stops)
        bar_filled = "●" * step
        bar_empty = "○" * (total - step)
        route_display = " → ".join(
            f"**{s}**" if i == self.current_step else s
            for i, s in enumerate(self.stops)
        )

        if self.language == "en":
            return f"📍 Stop {step}/{total} {bar_filled}{bar_empty}\n🗺️ Route: {route_display}"
        elif self.language == "ja":
            return f"📍 {step}/{total}番目 {bar_filled}{bar_empty}\n🗺️ ルート: {route_display}"
        return f"📍 第 {step}/{total} 站 {bar_filled}{bar_empty}\n🗺️ 路線：{route_display}"

    def _completion_message(self) -> str:
        if self.language == "en":
            return (
                "🎉 Tour Complete!\n\n"
                f"You visited {len(self.visited_areas)} areas: {', '.join(self.visited_areas)}\n"
                f"Route: {self.route_name} ({self.duration})\n\n"
                "Thank you for visiting! Feel free to ask any questions."
            )
        elif self.language == "ja":
            return (
                "🎉 見学完了！\n\n"
                f"見学エリア: {', '.join(self.visited_areas)}\n"
                f"ルート: {self.route_name}（{self.duration}）\n\n"
                "ご見学ありがとうございました！ご質問がありましたらどうぞ。"
            )
        return (
            "🎉 導覽完成！\n\n"
            f"您參觀了 {len(self.visited_areas)} 個區域：{', '.join(self.visited_areas)}\n"
            f"路線：{self.route_name}（{self.duration}）\n\n"
            "感謝您的參觀！如有任何問題，隨時可以繼續詢問。"
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "route_name": self.route_name,
            "current_step": self.current_step,
            "current_area": self.current_area,
            "total_stops": self.total_stops,
            "stops": self.stops,
            "visited_areas": self.visited_areas,
            "progress_percent": self.progress_percent,
            "completed": self.completed,
            "duration": self.duration,
        }


class TourManager:
    """管理所有導覽 session"""

    def __init__(self):
        self.sessions: dict[str, TourSession] = {}

    def start_tour(
        self,
        session_id: str,
        route_name: str = "標準導覽路線",
        language: str = "zh-TW",
    ) -> dict:
        session = TourSession(session_id, route_name, language)
        self.sessions[session_id] = session
        return {
            "status": "started",
            "message": session.get_current_intro(),
            **session.to_dict(),
        }

    def next_stop(self, session_id: str) -> dict:
        session = self.sessions.get(session_id)
        if not session:
            return {"status": "error", "message": "尚未開始導覽，請先開始一個導覽行程。"}
        return session.advance()

    def get_status(self, session_id: str) -> dict | None:
        session = self.sessions.get(session_id)
        if session:
            return session.to_dict()
        return None

    def get_available_routes(self) -> list[dict]:
        return [
            {
                "name": r["name"],
                "duration": r["duration"],
                "stops": r["stops"],
                "description": r["description"],
            }
            for r in KNOWLEDGE.get("routes", [])
        ]
