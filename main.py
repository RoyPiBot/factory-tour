"""
main.py - 工廠導覽 Multi-Agent Web API v2.1
使用 FastAPI 提供 RESTful 介面
AI 後端：Groq (Llama 4 Scout / 可透過 GROQ_MODEL 環境變數切換)

功能：
  - 5 Agent 系統（導覽/安全/技術/QA/知識檢索）
  - 多語言支援 (zh-TW/en/ja)
  - 互動式導覽流程
  - 對話歷史持久化 (SQLite)
  - RAG 知識檢索（工廠知識 + 自訂文件）
  - 文件管理 API（匯入/列表/刪除 Markdown 文件）

v2.1 — 整合東海大學 RAG 專案（方案A）

作者：Roy (YORROY123)
建立：2026-03-30
更新：2026-03-31 (RAG 整合)

啟動方式：
    cd /home/pi/factory-tour
    source /home/pi/factory-tour-env/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000

API 端點：
    POST /chat            - 對話（支援多語言）
    POST /tour/start      - 開始互動導覽
    POST /tour/next       - 下一站
    GET  /tour/status/:id - 導覽進度
    GET  /tour/routes     - 可用路線
    GET  /areas           - 列出所有廠區
    GET  /areas/:name     - 區域詳情
    GET  /routes          - 導覽路線
    GET  /faq             - 常見問題
    POST /documents       - 匯入文件（Markdown 文字或檔案路徑）
    GET  /documents       - 列出自訂文件
    DELETE /documents/:name - 刪除自訂文件
    GET  /history/:id     - 對話歷史
    GET  /sessions        - 所有 session
    GET  /stats           - 系統統計
    GET  /health          - 健康檢查
    GET  /i18n/:lang      - 多語言字串
"""
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from factory_tour_agent import create_factory_tour_app, KNOWLEDGE
from tour_flow import TourManager
from i18n import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, UI_STRINGS
import db as database

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 路徑設定 ───
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

# ─── 全域變數 ───
agent_apps: dict = {}  # language -> agent_app
_agent_lock = threading.Lock()  # 保護 agent_apps 的並發初始化
tour_manager = TourManager()
rag_ready = False

MAX_MESSAGE_LENGTH = 2000  # 使用者訊息最大長度


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動/關閉時執行"""
    global rag_ready

    # 初始化資料庫
    database.init_db()
    logger.info("✅ SQLite 資料庫初始化完成")

    # 初始化預設語言的 Agent
    try:
        agent_apps[DEFAULT_LANGUAGE] = create_factory_tour_app(
            language=DEFAULT_LANGUAGE
        )
        logger.info("✅ Multi-Agent 系統初始化完成 (zh-TW)")
    except ValueError as e:
        logger.warning(f"⚠️ Agent 初始化失敗：{e}")
        logger.warning("API 將在無 Agent 模式下運行（僅靜態資料可用）")

    # 初始化 RAG（可選，可透過 SKIP_RAG=1 跳過）
    try:
        if os.getenv("SKIP_RAG"):
            raise RuntimeError("SKIP_RAG is set")
        from rag_engine import get_rag_engine

        engine = get_rag_engine()
        rag_ready = engine.ready
        if rag_ready:
            logger.info(
                f"✅ RAG 引擎就緒 ({engine.get_stats()['total_documents']} 筆文件)"
            )
        else:
            logger.info("ℹ️ RAG 引擎未就緒（chromadb 未安裝或初始化失敗）")
    except Exception as e:
        logger.info(f"ℹ️ RAG 引擎停用: {e}")

    yield


app = FastAPI(
    title="工廠導覽 Multi-Agent API",
    description="基於 LangGraph + Groq 的智慧多語言工廠導覽系統",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 取得/建立 Agent App ───
def get_agent(language: str = DEFAULT_LANGUAGE):
    """取得指定語言的 Agent，若不存在則建立（執行緒安全）"""
    if language in agent_apps:
        return agent_apps[language]
    with _agent_lock:
        # Double-check：取得鎖後再次確認，避免重複建立
        if language not in agent_apps:
            try:
                agent_apps[language] = create_factory_tour_app(language=language)
                logger.info(f"✅ Agent 初始化完成 ({language})")
            except ValueError as e:
                raise HTTPException(503, f"Agent 初始化失敗：{e}")
    return agent_apps[language]


# ─── 資料模型 ───
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    language: str = DEFAULT_LANGUAGE

    @field_validator("message")
    @classmethod
    def message_not_empty_or_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("訊息不得為空")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"訊息過長，上限 {MAX_MESSAGE_LENGTH} 字元")
        return v


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    agent_name: str | None = None


class TourStartRequest(BaseModel):
    session_id: str = "default"
    route_name: str = "標準導覽路線"
    language: str = DEFAULT_LANGUAGE


class TourNextRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    areas_loaded: int
    rag_ready: bool
    languages: list[str]
    total_sessions: int


# ═══════════════════════════════════════════
# 頁面端點
# ═══════════════════════════════════════════


@app.get("/", response_class=HTMLResponse)
async def root():
    """首頁 - 全螢幕 RPG 遊戲導覽"""
    game_file = BASE_DIR / "static" / "game.html"
    if game_file.exists():
        return HTMLResponse(game_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Game file not found</h1>", status_code=500)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """舊版 Dashboard 介面"""
    html_file = TEMPLATES_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Template not found</h1>", status_code=500)


# ═══════════════════════════════════════════
# 對話 API
# ═══════════════════════════════════════════


def _find_best_reply(messages) -> tuple[str, str | None]:
    """從 message chain 中找出最佳回覆（優先取 agent 的回覆，而非 supervisor 的摘要）"""
    from langchain_core.messages import AIMessage

    # 只考慮 AIMessage（跳過 HumanMessage、ToolMessage）
    ai_messages = [
        m for m in messages
        if isinstance(m, AIMessage) and m.content and m.content.strip()
    ]

    # 從後往前找，找有實質內容的 agent 回覆（非 supervisor、非 transfer 訊息）
    for msg in reversed(ai_messages):
        name = getattr(msg, "name", None)
        content = msg.content.strip()
        # 跳過 transfer 相關的訊息和空洞的 supervisor 摘要
        if "transfer" in content.lower():
            continue
        if name and name not in ("supervisor",):
            return content, name

    # 退而求其次：取 supervisor 的回覆
    for msg in reversed(ai_messages):
        content = msg.content.strip()
        if content and "transfer" not in content.lower():
            return content, getattr(msg, "name", None)

    # 最終兜底
    return "抱歉，系統暫時無法回覆。請稍後再試。", None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """對話端點（支援多語言，含 tool_use_failed 自動重試）"""
    language = (
        req.language if req.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    )

    agent = get_agent(language)

    config = {"configurable": {"thread_id": req.session_id}}
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
        )
        reply, agent_name = _find_best_reply(result["messages"])

        # 儲存對話到 SQLite
        try:
            database.save_message(req.session_id, "user", req.message, language)
            database.save_message(
                req.session_id, "assistant", reply, language, agent_name
            )
        except Exception as e:
            logger.warning(f"對話儲存失敗: {e}")

        return ChatResponse(
            reply=reply, session_id=req.session_id, agent_name=agent_name
        )
    except Exception as e:
        error_str = str(e)

        # 偵測 Groq tool_use_failed 錯誤（部分模型的已知 bug）
        if "tool_use_failed" in error_str:
            current_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
            logger.warning(
                f"⚠️ Groq tool_use_failed 錯誤（模型 {current_model} function calling 格式問題）。"
                f"建議切換至 meta-llama/llama-4-scout-17b-16e-instruct"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"目前使用的模型 ({current_model}) 有已知的 function calling 格式問題。"
                    "請在 .env 中設定 GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct"
                ),
            )

        logger.error(f"Agent 錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 錯誤：{error_str}")


# ═══════════════════════════════════════════
# 導覽流程 API
# ═══════════════════════════════════════════


@app.post("/tour/start")
async def start_tour(req: TourStartRequest):
    """開始互動式導覽"""
    language = (
        req.language if req.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    )
    result = tour_manager.start_tour(req.session_id, req.route_name, language)

    # 儲存導覽狀態到 SQLite
    try:
        database.save_tour_state(
            req.session_id,
            req.route_name,
            result.get("current_step", 0),
            result.get("visited_areas", []),
            language,
        )
    except Exception as e:
        logger.warning(f"導覽狀態儲存失敗: {e}")

    return result


@app.post("/tour/next")
async def next_stop(req: TourNextRequest):
    """前進到下一站"""
    result = tour_manager.next_stop(req.session_id)

    # 更新 SQLite
    try:
        state = tour_manager.get_status(req.session_id)
        if state:
            database.save_tour_state(
                req.session_id,
                state.get("route_name", ""),
                state.get("current_step", 0),
                state.get("visited_areas", []),
                completed=state.get("completed", False),
            )
    except Exception as e:
        logger.warning(f"導覽狀態更新失敗: {e}")

    return result


@app.get("/tour/status/{session_id}")
async def tour_status(session_id: str):
    """查詢導覽進度"""
    state = tour_manager.get_status(session_id)
    if state:
        return state
    db_state = database.get_tour_state(session_id)
    if db_state:
        return db_state
    raise HTTPException(404, "找不到此導覽 session")


@app.get("/tour/routes")
async def get_tour_routes():
    """取得可用導覽路線"""
    return {"routes": tour_manager.get_available_routes()}


# ═══════════════════════════════════════════
# 資料 API
# ═══════════════════════════════════════════


@app.get("/areas")
async def list_areas():
    """列出所有廠區"""
    return {
        "areas": [
            {
                "name": a["name"],
                "description": a["description"][:100] + "...",
                "order": a["tour_order"],
            }
            for a in sorted(
                KNOWLEDGE.get("areas", []), key=lambda x: x["tour_order"]
            )
        ]
    }


@app.get("/areas/{area_name}")
async def get_area_detail(area_name: str):
    """取得特定區域詳情（支援 name 或 id）"""
    for area in KNOWLEDGE.get("areas", []):
        if area["name"] == area_name or area["id"] == area_name:
            return area
    raise HTTPException(404, f"找不到區域: {area_name}")


@app.get("/routes")
async def list_routes():
    """列出導覽路線"""
    return {"routes": KNOWLEDGE.get("routes", [])}


@app.get("/faq")
async def list_faq():
    """列出常見問題"""
    from factory_tour_agent import FAQ_DATA

    return {"faq": FAQ_DATA}


# ═══════════════════════════════════════════
# 文件管理 API（RAG 整合 — 方案A）
# ═══════════════════════════════════════════


class DocumentUploadRequest(BaseModel):
    """文件匯入請求"""
    content: str | None = None  # Markdown 文字內容（與 file_path 二擇一）
    file_path: str | None = None  # Markdown 檔案路徑（與 content 二擇一）
    name: str | None = None  # 文件名稱（僅 content 模式需要）

    @field_validator("content", "file_path")
    @classmethod
    def at_least_one(cls, v, info):
        return v  # 在 endpoint 中驗證


@app.post("/documents")
async def upload_document(req: DocumentUploadRequest):
    """匯入 Markdown 文件到自訂知識庫

    兩種模式：
    1. 直接傳送 Markdown 文字：提供 content + name
    2. 指定 Pi 上的檔案路徑：提供 file_path
    """
    try:
        if os.getenv("SKIP_RAG"):
            raise HTTPException(503, "RAG 引擎已停用 (SKIP_RAG=1)")

        from rag_engine import get_rag_engine

        engine = get_rag_engine()
        if not engine.ready:
            raise HTTPException(503, "RAG 引擎未就緒")

        if req.file_path:
            # 模式 2: 從檔案匯入
            chunks = engine.add_markdown_file(req.file_path)
            if chunks == 0:
                raise HTTPException(400, f"無法匯入檔案: {req.file_path}")
            return {
                "status": "ok",
                "source": req.file_path,
                "chunks_added": chunks,
            }
        elif req.content:
            # 模式 1: 從文字匯入
            doc_name = req.name or "unnamed_document"
            doc_id = f"doc_{doc_name}"
            chunks = engine.add_document_from_text(
                doc_id, req.content, metadata={"source_file": f"{doc_name}.md"}
            )
            if chunks == 0:
                raise HTTPException(400, "文件內容過短或無效")
            return {
                "status": "ok",
                "name": doc_name,
                "chunks_added": chunks,
            }
        else:
            raise HTTPException(400, "請提供 content 或 file_path")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件匯入失敗: {e}")
        raise HTTPException(500, f"文件匯入失敗: {e}")


@app.get("/documents")
async def list_documents():
    """列出所有自訂文件"""
    try:
        if os.getenv("SKIP_RAG"):
            return {"documents": [], "message": "RAG 已停用"}
        from rag_engine import get_rag_engine

        engine = get_rag_engine()
        docs = engine.list_custom_documents()
        stats = engine.get_stats()
        return {
            "documents": docs,
            "total_custom_chunks": stats.get("custom_documents", 0),
            "embedding_model": stats.get("embedding_model", "unknown"),
        }
    except Exception as e:
        logger.error(f"列出文件失敗: {e}")
        return {"documents": [], "error": str(e)}


@app.delete("/documents/{source_file}")
async def delete_document(source_file: str):
    """刪除指定的自訂文件"""
    try:
        if os.getenv("SKIP_RAG"):
            raise HTTPException(503, "RAG 已停用")
        from rag_engine import get_rag_engine

        engine = get_rag_engine()
        removed = engine.remove_document(source_file)
        if removed:
            return {"status": "deleted", "source_file": source_file}
        raise HTTPException(404, f"找不到文件: {source_file}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"刪除失敗: {e}")


# ═══════════════════════════════════════════
# 歷史 & 統計 API
# ═══════════════════════════════════════════


@app.get("/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """取得對話歷史"""
    history = database.get_history(session_id, limit)
    return {"session_id": session_id, "messages": history, "count": len(history)}


@app.get("/sessions")
async def list_sessions(limit: int = 50):
    """列出所有對話 session"""
    sessions = database.get_all_sessions(limit)
    return {"sessions": sessions}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """刪除對話 session"""
    database.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/stats")
async def get_stats():
    """取得系統統計"""
    db_stats = database.get_stats()
    rag_stats = {}
    try:
        if os.getenv("SKIP_RAG"):
            raise RuntimeError("SKIP_RAG")
        from rag_engine import get_rag_engine

        rag_stats = get_rag_engine().get_stats()
    except Exception:
        rag_stats = {"ready": False}

    return {
        "database": db_stats,
        "rag": rag_stats,
        "agents": {
            "loaded_languages": list(agent_apps.keys()),
            "total_agents_per_language": 5,
        },
    }


# ═══════════════════════════════════════════
# 健康檢查 & i18n
# ═══════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康檢查"""
    try:
        db_stats = database.get_stats()
        total_sessions = db_stats.get("total_sessions", 0)
    except Exception:
        total_sessions = 0

    return HealthResponse(
        status="ok",
        agent_ready=len(agent_apps) > 0,
        areas_loaded=len(KNOWLEDGE.get("areas", [])),
        rag_ready=rag_ready,
        languages=list(agent_apps.keys()),
        total_sessions=total_sessions,
    )


@app.get("/i18n/{language}")
async def get_i18n(language: str):
    """取得前端多語言字串"""
    strings = UI_STRINGS.get(language, UI_STRINGS[DEFAULT_LANGUAGE])
    return {"language": language, "strings": strings}


# 掛載靜態檔案（放在所有路由之後）
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
