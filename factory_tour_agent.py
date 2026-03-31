"""
factory_tour_agent.py - 工廠導覽 Multi-Agent 系統
在 Raspberry Pi 5 上運行，使用 LangGraph + Groq (Llama 3.3)

作者：Roy (YORROY123)
建立：2026-03-30
"""
import os
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ─── 路徑設定 ───
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# ─── 載入知識庫 ───
def load_knowledge() -> dict:
    """從 JSON 載入工廠知識庫"""
    areas_file = KNOWLEDGE_DIR / "areas.json"
    if areas_file.exists():
        with open(areas_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"areas": [], "routes": [], "emergency": {}}

KNOWLEDGE = load_knowledge()

# 建立快速查找字典
AREA_MAP = {area["name"]: area for area in KNOWLEDGE.get("areas", [])}
ROUTE_MAP = {route["name"]: route for route in KNOWLEDGE.get("routes", [])}
EMERGENCY = KNOWLEDGE.get("emergency", {})


# ─── 模型初始化 ───
def get_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0.7):
    """初始化 Groq LLM"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "未設定 GROQ_API_KEY！\n"
            "請在 .env 檔案中設定，或執行：\n"
            "  export GROQ_API_KEY='your-key-here'\n"
            "API Key 申請：https://console.groq.com/keys"
        )
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=api_key,
    )


# ─── 工具定義 ───
@tool
def get_factory_info(area_name: str) -> str:
    """取得工廠特定區域的介紹資訊。

    Args:
        area_name: 廠區名稱，例如 "組裝線A"、"品管室"、"倉儲區"、"大廳"、"會議室"
    """
    area = AREA_MAP.get(area_name)
    if area:
        info = f"【{area['name']}】\n{area['description']}"
        if area.get("technical_specs"):
            specs = area["technical_specs"]
            info += "\n\n技術規格："
            for key, value in specs.items():
                if isinstance(value, list):
                    info += f"\n  - {key}: {', '.join(value)}"
                else:
                    info += f"\n  - {key}: {value}"
        return info
    available = ", ".join(AREA_MAP.keys())
    return f"找不到「{area_name}」的資訊。可用區域：{available}"


@tool
def get_all_areas() -> str:
    """取得所有廠區的列表和簡介"""
    result = "工廠區域一覽：\n"
    for area in sorted(KNOWLEDGE.get("areas", []), key=lambda x: x["tour_order"]):
        result += f"\n{area['tour_order']+1}. 【{area['name']}】— {area['description'][:50]}..."
    return result


@tool
def get_route_info(route_name: Optional[str] = None) -> str:
    """取得導覽路線資訊。

    Args:
        route_name: 路線名稱。不指定則列出所有路線。
    """
    if route_name and route_name in ROUTE_MAP:
        route = ROUTE_MAP[route_name]
        stops = " → ".join(route["stops"])
        return f"【{route['name']}】\n{route['description']}\n時間：{route['duration']}\n路線：{stops}"

    result = "可用導覽路線：\n"
    for route in KNOWLEDGE.get("routes", []):
        stops = " → ".join(route["stops"])
        result += f"\n• {route['name']}（{route['duration']}）：{stops}"
    return result


@tool
def get_safety_rules(area_name: str) -> str:
    """取得特定區域的安全規範。

    Args:
        area_name: 廠區名稱，例如 "組裝線A"、"品管室"、"倉儲區"
    """
    area = AREA_MAP.get(area_name)
    if area:
        notes = area.get("safety_notes", [])
        if notes:
            rules = "\n".join(f"  ⚠️ {note}" for note in notes)
            return f"【{area['name']}】安全規範：\n{rules}"
        return f"【{area['name']}】目前沒有特殊安全規範。"
    available = ", ".join(AREA_MAP.keys())
    return f"找不到「{area_name}」的安全規範。可用區域：{available}"


@tool
def get_all_safety_rules() -> str:
    """取得所有區域的安全規範總覽"""
    result = "全廠安全規範總覽：\n"
    for area in KNOWLEDGE.get("areas", []):
        notes = area.get("safety_notes", [])
        if notes:
            result += f"\n【{area['name']}】"
            for note in notes:
                result += f"\n  ⚠️ {note}"
    return result


@tool
def get_emergency_info() -> str:
    """取得緊急應變資訊，包括緊急出口、集合點、聯絡電話等"""
    if not EMERGENCY:
        return "緊急資訊尚未設定。"
    result = "🚨 緊急應變資訊：\n"
    result += f"\n🚪 緊急出口：{EMERGENCY.get('exit_locations', 'N/A')}"
    result += f"\n📍 集合點：{EMERGENCY.get('assembly_point', 'N/A')}"
    result += f"\n📞 緊急聯絡：{EMERGENCY.get('emergency_contact', 'N/A')}"
    result += f"\n💓 AED 位置：{EMERGENCY.get('aed_location', 'N/A')}"
    result += f"\n🩹 急救箱：{EMERGENCY.get('first_aid', 'N/A')}"
    return result


# ─── Agent 定義 ───
def create_tour_guide(llm):
    """建立導覽 Agent"""
    return create_react_agent(
        model=llm,
        tools=[get_factory_info, get_all_areas, get_route_info],
        name="tour_guide",
        prompt=(
            "你是工廠導覽員，負責介紹工廠各區域、設備和導覽路線。\n"
            "用友善、專業的語氣說明，讓第一次參觀的訪客也能輕鬆理解。\n"
            "回答要簡潔但資訊豐富。使用繁體中文。\n"
            "如果訪客問的區域不存在，先用 get_all_areas 工具列出所有區域供參考。"
        ),
    )


def create_safety_expert(llm):
    """建立安全專家 Agent"""
    return create_react_agent(
        model=llm,
        tools=[get_safety_rules, get_all_safety_rules, get_emergency_info],
        name="safety_expert",
        prompt=(
            "你是工廠安全專家，負責回答安全規範、個人防護裝備和緊急應變問題。\n"
            "語氣嚴謹但不令人緊張，確保訪客了解重要的安全資訊。\n"
            "使用繁體中文。\n"
            "安全永遠是第一優先，如果有任何疑慮，建議訪客聯繫現場安全人員。"
        ),
    )


# ─── 建構 Multi-Agent 系統 ───
def create_factory_tour_app(checkpointer=None):
    """建立並回傳工廠導覽 Multi-Agent 應用"""
    llm = get_llm()

    tour_guide = create_tour_guide(llm)
    safety_expert = create_safety_expert(llm)

    workflow = create_supervisor(
        agents=[tour_guide, safety_expert],
        model=llm,
        prompt=(
            "你是工廠導覽系統的總管。根據訪客的問題，決定由哪位專家回答：\n"
            "- tour_guide：廠區介紹、設備說明、導覽路線、一般問題\n"
            "- safety_expert：安全規範、防護裝備、緊急應變、安全相關\n\n"
            "規則：\n"
            "1. 如果問題同時涉及多個領域，先處理安全相關的部分\n"
            "2. 用繁體中文回覆\n"
            "3. 如果訪客只是打招呼，由 tour_guide 回應\n"
            "4. 如果不確定歸屬，交給 tour_guide"
        ),
    )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# ─── CLI 互動模式 ───
def main():
    """命令列互動模式"""
    print("=" * 50)
    print("  🏭 工廠導覽 Multi-Agent 系統")
    print("  Powered by LangGraph + Groq")
    print("=" * 50)
    print("輸入 'quit' 或 'q' 結束\n")

    try:
        app = create_factory_tour_app()
    except ValueError as e:
        print(f"❌ 初始化失敗：{e}")
        return

    config = {"configurable": {"thread_id": "cli-tour-001"}}

    while True:
        try:
            user_input = input("👤 訪客: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n感謝參觀，再見！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("感謝參觀，再見！👋")
            break
        if not user_input:
            continue

        try:
            result = app.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            ai_message = result["messages"][-1]
            print(f"\n🤖 導覽系統: {ai_message.content}\n")
        except Exception as e:
            print(f"\n❌ 發生錯誤：{e}\n")


if __name__ == "__main__":
    main()
