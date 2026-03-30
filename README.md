# 工廠導覽 Multi-Agent 系統

基於 LangGraph + Gemini 的智慧工廠導覽系統，運行在 Raspberry Pi 5 上。

## 架構

- **Supervisor Agent** — 路由決策，分派任務給專家
- **Tour Guide Agent** — 廠區介紹、設備說明、導覽路線
- **Safety Expert Agent** — 安全規範、防護裝備、緊急應變

## 快速開始

```bash
# 1. 進入專案目錄
cd /home/pi/factory-tour

# 2. 啟用虛擬環境
source /home/pi/factory-tour-env/bin/activate

# 3. 設定 API Key
cp .env.example .env
# 編輯 .env，填入你的 Google Gemini API Key

# 4a. 命令列模式
python factory_tour_agent.py

# 4b. Web API 模式
uvicorn main:app --host 0.0.0.0 --port 8000
# 然後用瀏覽器打開 http://<pi-ip>:8000
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | Web 聊天介面 |
| POST | `/chat` | 對話 API |
| GET | `/areas` | 廠區列表 |
| GET | `/routes` | 導覽路線 |
| GET | `/health` | 健康檢查 |

## 知識庫

工廠資訊存放在 `knowledge/areas.json`，可自行編輯新增區域。

## 技術棧

- Python 3.13
- LangGraph 1.1 + langgraph-supervisor
- LangChain + Google Gemini 2.0 Flash
- FastAPI + Uvicorn
- SQLite (對話持久化，選配)
