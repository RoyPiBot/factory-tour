# 工廠導覽 Multi-Agent 系統

基於 LangGraph + Groq (Llama 3.3 70B) 的智慧工廠導覽系統，運行在 Raspberry Pi 5 上。

## 架構總覽

### 系統架構圖

```mermaid
flowchart TB
    subgraph Entry["🖥️ 使用者介面"]
        CLI["CLI 互動模式<br/><code>factory_tour_agent.py</code>"]
        WEB["Web 聊天介面<br/><code>main.py (FastAPI)</code>"]
    end

    subgraph API["🌐 FastAPI REST API"]
        direction LR
        EP1["POST /chat"]
        EP2["GET /areas"]
        EP3["GET /routes"]
        EP4["GET /health"]
    end

    subgraph LangGraph["🧠 LangGraph Multi-Agent 系統"]
        SUP["🎯 Supervisor Agent<br/>路由決策・任務分派"]

        subgraph Agents["專家 Agents"]
            TG["🏭 Tour Guide<br/>導覽員"]
            SE["🛡️ Safety Expert<br/>安全專家"]
        end

        subgraph Tools_TG["導覽工具"]
            T1["get_factory_info"]
            T2["get_all_areas"]
            T3["get_route_info"]
        end

        subgraph Tools_SE["安全工具"]
            T4["get_safety_rules"]
            T5["get_all_safety_rules"]
            T6["get_emergency_info"]
        end

        SUP -->|廠區・路線問題| TG
        SUP -->|安全・緊急問題| SE
        TG --> T1 & T2 & T3
        SE --> T4 & T5 & T6
    end

    subgraph Data["💾 資料層"]
        KB["knowledge/areas.json<br/>工廠知識庫"]
        MEM["InMemorySaver<br/>對話記憶"]
    end

    subgraph Infra["⚙️ 基礎設施"]
        GEMINI["☁️ Groq Llama 3.3<br/>(Google AI)"]
        PI["🍓 Raspberry Pi 5<br/>16GB RAM"]
    end

    CLI --> SUP
    WEB --> API --> SUP
    T1 & T2 & T3 & T4 & T5 & T6 --> KB
    SUP --> MEM
    SUP & TG & SE -.->|LLM 呼叫| GEMINI
    LangGraph ~~~ PI

    style Entry fill:#e3f2fd,stroke:#1976d2
    style LangGraph fill:#fff3e0,stroke:#f57c00
    style Data fill:#e8f5e9,stroke:#388e3c
    style Infra fill:#fce4ec,stroke:#c62828
    style SUP fill:#ff9800,color:#fff,stroke:#e65100
    style TG fill:#42a5f5,color:#fff
    style SE fill:#ef5350,color:#fff
```

### 專案檔案結構

```
factory-tour/
├── main.py                    # FastAPI Web 伺服器 + REST API + 2D RPG 前端
├── factory_tour_agent.py      # Multi-Agent 核心（Supervisor + Agents + Tools）
├── rag_engine.py              # RAG 知識檢索引擎（SentenceTransformer + ChromaDB）
├── db.py                      # SQLite 對話持久化
├── i18n.py                    # 多語系支援（中/英/日）
├── tour_flow.py               # 導覽流程控制
├── knowledge/
│   ├── areas.json             # 工廠知識庫（區域、路線、安全規範）
│   └── faq.json               # 常見問題集
├── documents/                 # RAG 用文件資料夾
├── static/
│   ├── game.html              # 2D RPG 工廠導覽遊戲介面
│   ├── js/                    # 前端 JavaScript（dialog.js, npc.js 等）
│   ├── css/                   # 樣式表
│   └── img/                   # 遊戲素材圖片
├── templates/
│   └── index.html             # Web 聊天介面模板
├── tests/                     # 測試
├── data/                      # 運行時資料（ChromaDB 向量庫等）
├── requirements.txt           # Python 套件依賴
├── .env                       # 環境變數（API Keys）
├── .env.example               # 環境變數範本
├── .gitignore
└── README.md
```

### 對話處理流程圖

```mermaid
sequenceDiagram
    actor User as 訪客
    participant Web as FastAPI<br/>Web 介面
    participant Sup as Supervisor<br/>Agent
    participant LLM as Gemini 2.0<br/>Flash
    participant TG as Tour Guide<br/>Agent
    participant SE as Safety Expert<br/>Agent
    participant KB as Knowledge Base<br/>areas.json

    User->>Web: POST /chat<br/>「組裝線A的設備有哪些？」
    Web->>Sup: invoke(messages)

    Note over Sup,LLM: Step 1: 路由決策
    Sup->>LLM: 分析問題類型
    LLM-->>Sup: → 廠區問題 → tour_guide

    Note over Sup,TG: Step 2: 分派給專家
    Sup->>TG: 轉交問題
    TG->>LLM: 決定使用哪個工具
    LLM-->>TG: → get_factory_info("組裝線A")

    Note over TG,KB: Step 3: 查詢知識庫
    TG->>KB: 讀取 areas.json
    KB-->>TG: 組裝線A 的完整資料

    Note over TG,LLM: Step 4: 生成回覆
    TG->>LLM: 結合資料生成自然語言回答
    LLM-->>TG: 友善的導覽說明
    TG-->>Sup: 回傳結果
    Sup-->>Web: 最終回覆
    Web-->>User: 顯示回答
```

### 導覽路線地圖

```mermaid
graph LR
    A["🏢 大廳<br/><small>訪客登記・領取裝備</small>"]
    B["⚙️ 組裝線A<br/><small>SMT 貼片・回焊・AOI</small>"]
    C["🔍 品管室<br/><small>AOI・X-ray・功能測試</small>"]
    D["📦 倉儲區<br/><small>自動化立體倉庫</small>"]
    E["🪑 會議室<br/><small>Q&A・合作討論</small>"]

    A -->|"標準 & 快速"| B
    B -->|"標準 & 快速"| C
    C -->|"標準路線"| D
    C -->|"快速路線"| E
    D -->|"標準路線"| E

    style A fill:#bbdefb,stroke:#1565c0
    style B fill:#fff9c4,stroke:#f9a825
    style C fill:#c8e6c9,stroke:#2e7d32
    style D fill:#ffe0b2,stroke:#ef6c00
    style E fill:#e1bee7,stroke:#7b1fa2
```

### 角色說明

| Agent | 職責 | 工具 |
|-------|------|------|
| **Supervisor** | 分析訪客問題，路由給對應的專家 Agent | — |
| **Tour Guide** | 廠區介紹、設備說明、導覽路線 | `get_factory_info`, `get_all_areas`, `get_route_info` |
| **Safety Expert** | 安全規範、防護裝備、緊急應變 | `get_safety_rules`, `get_all_safety_rules`, `get_emergency_info` |
| **Knowledge Agent** | RAG 知識檢索，回答文件相關問題 | SentenceTransformer + ChromaDB |

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

## 功能亮點

- 🎮 **2D RPG 導覽介面** — 像素風遊戲地圖，點擊 NPC 觸發對話
- 🧠 **RAG 知識檢索** — 上傳文件自動建立向量索引，支援語意搜尋
- 🌐 **多語系** — 支援中文、英文、日文切換
- 💾 **對話記憶** — SQLite 持久化，跨 session 保留對話歷史
- 🛡️ **安全導覽** — 獨立的安全專家 Agent 處理緊急情況

## 技術棧

- Python 3.13
- LangGraph 1.1 + langgraph-supervisor
- LangChain + Groq (Llama 3.3 70B)
- FastAPI + Uvicorn
- SentenceTransformers (all-MiniLM-L6-v2) + ChromaDB
- SQLite (對話持久化)
- 2D RPG 前端 (HTML5 Canvas + JavaScript)
