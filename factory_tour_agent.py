"""
factory_tour_agent.py - 工廠導覽 Multi-Agent 系統 v2.1
在 Raspberry Pi 5 上運行，使用 LangGraph + Groq (Llama 4 Scout)

功能：
  - 5 個 Agent（導覽員、安全專家、技術專家、QA 客服、知識檢索）
  - 多語言支援（繁中/英文/日文）
  - RAG 知識檢索（工廠知識 + 自訂文件）
  - 互動式導覽流程
  - SQLite 對話持久化

v2.1 — 整合東海大學 RAG 專案（方案A: 知識檢索 Agent）

作者：Roy (YORROY123)
建立：2026-03-30
更新：2026-03-31 (RAG 整合)
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver

from i18n import (
    get_prompt,
    SUPERVISOR_PROMPTS,
    TOUR_GUIDE_PROMPTS,
    SAFETY_EXPERT_PROMPTS,
    TECH_EXPERT_PROMPTS,
    QA_AGENT_PROMPTS,
    KNOWLEDGE_AGENT_PROMPTS,
    DEFAULT_LANGUAGE,
)

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


def load_faq() -> list[dict]:
    """載入 FAQ 知識庫"""
    faq_file = KNOWLEDGE_DIR / "faq.json"
    if faq_file.exists():
        with open(faq_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("faq", [])
    return []


KNOWLEDGE = load_knowledge()
FAQ_DATA = load_faq()

# 建立快速查找字典
AREA_MAP = {area["name"]: area for area in KNOWLEDGE.get("areas", [])}
ROUTE_MAP = {route["name"]: route for route in KNOWLEDGE.get("routes", [])}
EMERGENCY = KNOWLEDGE.get("emergency", {})
FAQ_MAP = {item["id"]: item for item in FAQ_DATA}


# ─── 模型初始化 ───
def get_llm(model: str = None, temperature: float = 0.3):
    """初始化 Groq LLM

    模型選擇注意事項：
    - 預設使用 meta-llama/llama-4-scout-17b-16e-instruct（tool calling 穩定）
    - ⚠️ llama-3.3-70b-versatile 有已知 bug：會生成 <function=...> XML 格式
      的 function call，導致 Groq API 400 tool_use_failed 錯誤，請勿使用。
    - 可透過環境變數 GROQ_MODEL 切換模型。
    - temperature 建議 ≤ 0.3 以確保 tool calling 格式穩定。
    """
    if model is None:
        model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
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


# ═══════════════════════════════════════════
# 導覽員 Agent 工具
# ═══════════════════════════════════════════

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
                    info += f"\n  - {key}: {', '.join(str(v) for v in value)}"
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
        result += (
            f"\n{area['tour_order']+1}. 【{area['name']}】— {area['description'][:50]}..."
        )
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
        return (
            f"【{route['name']}】\n{route['description']}\n"
            f"時間：{route['duration']}\n路線：{stops}"
        )

    result = "可用導覽路線：\n"
    for route in KNOWLEDGE.get("routes", []):
        stops = " → ".join(route["stops"])
        result += f"\n• {route['name']}（{route['duration']}）：{stops}"
    return result


# ═══════════════════════════════════════════
# 安全專家 Agent 工具
# ═══════════════════════════════════════════

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


# ═══════════════════════════════════════════
# 技術問答專家 Agent 工具（新增）
# ═══════════════════════════════════════════

@tool
def get_equipment_details(area_name: str) -> str:
    """取得特定區域的詳細設備規格和技術參數。

    Args:
        area_name: 廠區名稱，例如 "組裝線A"、"品管室"、"倉儲區"
    """
    area = AREA_MAP.get(area_name)
    if not area:
        available = ", ".join(AREA_MAP.keys())
        return f"找不到「{area_name}」。可用區域：{available}"

    result = f"【{area['name']}】設備與技術詳情：\n\n"
    result += f"描述：{area['description']}\n"

    specs = area.get("technical_specs", {})
    if specs:
        result += "\n📊 技術規格：\n"
        for key, value in specs.items():
            if isinstance(value, list):
                result += f"  • {key}:\n"
                for item in value:
                    result += f"    - {item}\n"
            else:
                result += f"  • {key}: {value}\n"
    else:
        result += "\n（此區域無特殊技術規格）\n"

    return result


@tool
def get_production_metrics() -> str:
    """取得工廠的整體產能、品質和效率指標"""
    metrics = []

    for area in KNOWLEDGE.get("areas", []):
        specs = area.get("technical_specs", {})
        if specs:
            metrics.append(f"【{area['name']}】")
            for key, value in specs.items():
                if isinstance(value, list):
                    metrics.append(f"  {key}: {', '.join(str(v) for v in value)}")
                else:
                    metrics.append(f"  {key}: {value}")
            metrics.append("")

    if not metrics:
        return "目前沒有可用的生產指標數據。"

    return "📊 工廠生產指標總覽：\n\n" + "\n".join(metrics)


@tool
def compare_areas(area1: str, area2: str) -> str:
    """比較兩個區域的設備和規格差異。

    Args:
        area1: 第一個區域名稱
        area2: 第二個區域名稱
    """
    a1 = AREA_MAP.get(area1)
    a2 = AREA_MAP.get(area2)

    if not a1 or not a2:
        available = ", ".join(AREA_MAP.keys())
        return f"找不到指定區域。可用區域：{available}"

    result = f"📊 {a1['name']} vs {a2['name']} 比較：\n\n"

    # 描述比較
    result += f"【{a1['name']}】\n{a1['description'][:100]}...\n\n"
    result += f"【{a2['name']}】\n{a2['description'][:100]}...\n\n"

    # 技術規格比較
    specs1 = a1.get("technical_specs", {})
    specs2 = a2.get("technical_specs", {})
    all_keys = set(list(specs1.keys()) + list(specs2.keys()))

    if all_keys:
        result += "技術規格對比：\n"
        for key in sorted(all_keys):
            v1 = specs1.get(key, "—")
            v2 = specs2.get(key, "—")
            if isinstance(v1, list):
                v1 = ", ".join(str(x) for x in v1)
            if isinstance(v2, list):
                v2 = ", ".join(str(x) for x in v2)
            result += f"  {key}: {v1} | {v2}\n"

    # 安全規範比較
    s1 = len(a1.get("safety_notes", []))
    s2 = len(a2.get("safety_notes", []))
    result += f"\n安全規範數量：{a1['name']}({s1}項) vs {a2['name']}({s2}項)"

    return result


# ═══════════════════════════════════════════
# QA Agent 工具（新增）
# ═══════════════════════════════════════════

@tool
def search_faq(keyword: str) -> str:
    """搜尋常見問題（FAQ）知識庫。

    Args:
        keyword: 搜尋關鍵字，例如 "停車"、"拍照"、"時間"
    """
    keyword_lower = keyword.lower()
    matches = []

    for item in FAQ_DATA:
        # 比對關鍵字
        if any(kw in keyword_lower for kw in item.get("keywords", [])):
            matches.append(item)
        elif keyword_lower in item["question"].lower():
            matches.append(item)
        elif keyword_lower in item["answer"].lower():
            matches.append(item)

    if not matches:
        return f"找不到與「{keyword}」相關的常見問題。您可以嘗試其他關鍵字，或直接描述您的問題。"

    result = f"找到 {len(matches)} 個相關問題：\n\n"
    for m in matches[:5]:
        result += f"❓ {m['question']}\n💡 {m['answer']}\n\n"
    return result


@tool
def get_all_faq() -> str:
    """列出所有常見問題的標題"""
    if not FAQ_DATA:
        return "目前沒有設定常見問題。"

    result = "📋 常見問題列表：\n\n"
    for i, item in enumerate(FAQ_DATA, 1):
        result += f"{i}. {item['question']}\n"
    result += f"\n共 {len(FAQ_DATA)} 個問題。請告訴我您想了解哪一個！"
    return result


@tool
def get_visitor_guidelines() -> str:
    """取得訪客須知的完整資訊"""
    guidelines = []
    for item in FAQ_DATA:
        if item.get("category") == "rules":
            guidelines.append(f"• {item['question']}: {item['answer']}")

    if not guidelines:
        return "目前沒有設定訪客規範資訊。"

    return "📜 訪客須知：\n\n" + "\n\n".join(guidelines)


# ═══════════════════════════════════════════
# RAG 工具（新增）
# ═══════════════════════════════════════════

@tool
def rag_knowledge_search(query: str) -> str:
    """使用向量檢索搜尋工廠知識庫，適合模糊或複雜問題。

    Args:
        query: 搜尋查詢，自然語言描述要找的資訊
    """
    try:
        from rag_engine import rag_search

        return rag_search(query, n_results=3)
    except ImportError:
        logger.info("RAG 引擎未安裝，fallback 到一般搜尋")
        return "RAG 引擎未安裝，請使用其他工具搜尋。"
    except Exception as e:
        logger.error(f"RAG 搜尋失敗: {e}")
        return f"RAG 搜尋暫時無法使用，請使用其他工具查詢。"


# ═══════════════════════════════════════════
# Knowledge Agent 工具（RAG 整合 — 方案A）
# ═══════════════════════════════════════════

@tool
def search_custom_knowledge(query: str) -> str:
    """搜尋自訂文件知識庫（外部匯入的 Markdown 文件）。

    Args:
        query: 搜尋查詢，自然語言描述要找的資訊
    """
    try:
        from rag_engine import rag_search_custom

        return rag_search_custom(query, n_results=3)
    except ImportError:
        return "RAG 引擎未安裝。"
    except Exception as e:
        logger.error(f"自訂知識搜尋失敗: {e}")
        return "搜尋暫時無法使用。"


@tool
def search_all_knowledge(query: str) -> str:
    """同時搜尋工廠知識和自訂文件知識庫，適合跨領域的問題。

    Args:
        query: 搜尋查詢，自然語言描述要找的資訊
    """
    try:
        from rag_engine import rag_search_all

        return rag_search_all(query, n_results=5)
    except ImportError:
        return "RAG 引擎未安裝。"
    except Exception as e:
        logger.error(f"全域知識搜尋失敗: {e}")
        return "搜尋暫時無法使用。"


@tool
def list_knowledge_documents() -> str:
    """列出所有已匯入的自訂文件清單"""
    try:
        from rag_engine import get_rag_engine

        engine = get_rag_engine()
        docs = engine.list_custom_documents()
        if not docs:
            return "目前沒有匯入任何自訂文件。"

        result = f"📚 已匯入 {len(docs)} 份自訂文件：\n\n"
        for doc in docs:
            result += f"  • {doc['source_file']} ({doc['chunks']} 段落)\n"
        return result
    except Exception as e:
        logger.error(f"列出文件失敗: {e}")
        return "無法取得文件清單。"


# ═══════════════════════════════════════════
# Agent 定義
# ═══════════════════════════════════════════

def create_tour_guide(llm, language: str = DEFAULT_LANGUAGE):
    """建立導覽 Agent"""
    tools = [get_factory_info, get_all_areas, get_route_info, rag_knowledge_search]
    return create_react_agent(
        model=llm,
        tools=tools,
        name="tour_guide",
        prompt=get_prompt(TOUR_GUIDE_PROMPTS, language),
    )


def create_safety_expert(llm, language: str = DEFAULT_LANGUAGE):
    """建立安全專家 Agent"""
    tools = [get_safety_rules, get_all_safety_rules, get_emergency_info]
    return create_react_agent(
        model=llm,
        tools=tools,
        name="safety_expert",
        prompt=get_prompt(SAFETY_EXPERT_PROMPTS, language),
    )


def create_tech_expert(llm, language: str = DEFAULT_LANGUAGE):
    """建立技術問答專家 Agent"""
    tools = [
        get_equipment_details,
        get_production_metrics,
        compare_areas,
        rag_knowledge_search,
    ]
    return create_react_agent(
        model=llm,
        tools=tools,
        name="tech_expert",
        prompt=get_prompt(TECH_EXPERT_PROMPTS, language),
    )


def create_qa_agent(llm, language: str = DEFAULT_LANGUAGE):
    """建立 QA 客服 Agent"""
    tools = [search_faq, get_all_faq, get_visitor_guidelines, rag_knowledge_search]
    return create_react_agent(
        model=llm,
        tools=tools,
        name="qa_agent",
        prompt=get_prompt(QA_AGENT_PROMPTS, language),
    )


def create_knowledge_agent(llm, language: str = DEFAULT_LANGUAGE):
    """建立知識檢索 Agent（RAG 整合 — 方案A）

    整合東海大學 RAG 專案的核心能力，支援：
    - 搜尋自訂文件知識庫
    - 跨集合全域搜尋（工廠 + 自訂）
    - 列出已匯入的文件
    """
    tools = [search_custom_knowledge, search_all_knowledge, list_knowledge_documents]
    return create_react_agent(
        model=llm,
        tools=tools,
        name="knowledge_agent",
        prompt=get_prompt(KNOWLEDGE_AGENT_PROMPTS, language),
    )


# ─── 建構 Multi-Agent 系統 ───
def create_factory_tour_app(
    checkpointer=None, language: str = DEFAULT_LANGUAGE
):
    """建立並回傳工廠導覽 Multi-Agent 應用 v2.0

    包含 5 個 Agent：
    1. tour_guide - 導覽員
    2. safety_expert - 安全專家
    3. tech_expert - 技術專家
    4. qa_agent - QA 客服
    5. knowledge_agent - 知識檢索（RAG 整合）

    Args:
        checkpointer: LangGraph checkpointer（預設 InMemorySaver）
        language: 語言代碼 (zh-TW, en, ja)
    """
    llm = get_llm()

    tour_guide = create_tour_guide(llm, language)
    safety_expert = create_safety_expert(llm, language)
    tech_expert = create_tech_expert(llm, language)
    qa_agent = create_qa_agent(llm, language)
    knowledge_agent = create_knowledge_agent(llm, language)

    workflow = create_supervisor(
        agents=[tour_guide, safety_expert, tech_expert, qa_agent, knowledge_agent],
        model=llm,
        prompt=get_prompt(SUPERVISOR_PROMPTS, language),
    )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# ─── CLI 互動模式 ───
def main():
    """命令列互動模式"""
    print("=" * 50)
    print("  🏭 工廠導覽 Multi-Agent 系統 v2.0")
    print("  Powered by LangGraph + Groq")
    print("  4 Agents | 3 Languages | RAG")
    print("=" * 50)
    print()
    print("語言選擇 / Language / 言語:")
    print("  1. 繁體中文 (預設)")
    print("  2. English")
    print("  3. 日本語")

    lang_choice = input("選擇 [1/2/3]: ").strip()
    lang_map = {"1": "zh-TW", "2": "en", "3": "ja"}
    language = lang_map.get(lang_choice, "zh-TW")

    print(f"\n使用語言: {language}")
    print("輸入 'quit' 或 'q' 結束\n")

    try:
        app = create_factory_tour_app(language=language)
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
