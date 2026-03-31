"""
main.py - 工廠導覽 Multi-Agent Web API
使用 FastAPI 提供 RESTful 介面
AI 後端：Groq (Llama 3.3 70B)

作者：Roy (YORROY123)
建立：2026-03-30

啟動方式：
    cd /home/pi/factory-tour
    source /home/pi/factory-tour-env/bin/activate
    uvicorn main:app --host 0.0.0.0 --port 8000

API 端點：
    POST /chat          - 對話
    GET  /areas         - 列出所有廠區
    GET  /routes        - 列出導覽路線
    GET  /health        - 健康檢查
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from factory_tour_agent import create_factory_tour_app, KNOWLEDGE

load_dotenv()

# ─── 全域變數 ───
agent_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動/關閉時執行"""
    global agent_app
    try:
        agent_app = create_factory_tour_app()
        print("✅ Multi-Agent 系統初始化完成")
    except ValueError as e:
        print(f"⚠️ 初始化失敗：{e}")
        print("API 將在無 Agent 模式下運行（僅靜態資料可用）")
    yield


app = FastAPI(
    title="工廠導覽 Multi-Agent API",
    description="基於 LangGraph + Groq 的工廠導覽系統",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 設定（允許前端跨域存取）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 資料模型 ───
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
    areas_loaded: int


# ─── API 端點 ───
@app.get("/", response_class=HTMLResponse)
async def root():
    """首頁 - 簡易測試介面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>工廠導覽系統</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                   background: #f0f2f5; display: flex; justify-content: center; padding: 20px; }
            .container { max-width: 600px; width: 100%; }
            h1 { text-align: center; margin: 20px 0; color: #1a1a1a; }
            .chat-box { background: white; border-radius: 12px; padding: 20px;
                        min-height: 400px; max-height: 600px; overflow-y: auto;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 12px; }
            .msg { margin: 8px 0; padding: 10px 14px; border-radius: 18px; max-width: 85%; }
            .msg.user { background: #0084ff; color: white; margin-left: auto; text-align: right; }
            .msg.bot { background: #e4e6eb; color: #1a1a1a; }
            .input-row { display: flex; gap: 8px; }
            input { flex: 1; padding: 12px 16px; border: 1px solid #ddd; border-radius: 24px;
                    font-size: 16px; outline: none; }
            input:focus { border-color: #0084ff; }
            button { padding: 12px 24px; background: #0084ff; color: white; border: none;
                     border-radius: 24px; font-size: 16px; cursor: pointer; }
            button:hover { background: #0073e6; }
            button:disabled { background: #ccc; }
            .status { text-align: center; color: #888; font-size: 14px; margin-top: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏭 工廠導覽系統</h1>
            <div class="chat-box" id="chat"></div>
            <div class="input-row">
                <input id="input" placeholder="請輸入您的問題..." autofocus
                       onkeydown="if(event.key==='Enter')send()">
                <button onclick="send()" id="btn">送出</button>
            </div>
            <div class="status">Powered by LangGraph + Groq on Raspberry Pi 5</div>
        </div>
        <script>
            const chat = document.getElementById('chat');
            const input = document.getElementById('input');
            const btn = document.getElementById('btn');
            function addMsg(text, cls) {
                const d = document.createElement('div');
                d.className = 'msg ' + cls;
                d.textContent = text;
                chat.appendChild(d);
                chat.scrollTop = chat.scrollHeight;
            }
            async function send() {
                const msg = input.value.trim();
                if (!msg) return;
                addMsg(msg, 'user');
                input.value = '';
                btn.disabled = true;
                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: msg})
                    });
                    const data = await res.json();
                    addMsg(data.reply || data.detail, 'bot');
                } catch(e) { addMsg('連線錯誤：' + e.message, 'bot'); }
                btn.disabled = false;
                input.focus();
            }
            addMsg('歡迎來到工廠導覽系統！請問有什麼我可以幫您的嗎？', 'bot');
        </script>
    </body>
    </html>
    """


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """對話端點"""
    if agent_app is None:
        raise HTTPException(
            status_code=503,
            detail="Agent 尚未初始化。請確認已設定 GROQ_API_KEY。"
        )

    config = {"configurable": {"thread_id": req.session_id}}
    try:
        result = agent_app.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
        )
        ai_message = result["messages"][-1]
        return ChatResponse(reply=ai_message.content, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 錯誤：{str(e)}")


@app.get("/areas")
async def list_areas():
    """列出所有廠區"""
    return {
        "areas": [
            {"name": a["name"], "description": a["description"][:100] + "...", "order": a["tour_order"]}
            for a in sorted(KNOWLEDGE.get("areas", []), key=lambda x: x["tour_order"])
        ]
    }


@app.get("/routes")
async def list_routes():
    """列出導覽路線"""
    return {"routes": KNOWLEDGE.get("routes", [])}


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康檢查"""
    return HealthResponse(
        status="ok",
        agent_ready=agent_app is not None,
        areas_loaded=len(KNOWLEDGE.get("areas", [])),
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
