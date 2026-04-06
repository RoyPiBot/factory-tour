# LangGraph Multi-Agent 開發模式研究報告 (2026 年初)

> 研究目標：在 Raspberry Pi 5 上使用 Python + LangChain/LangGraph + Gemini 建立工廠導覽 Multi-Agent 系統
> 調查日期：2026-03-31（第四次更新）

---

## 目錄

1. [LangGraph 框架概覽](#1-langgraph-框架概覽)
2. [Multi-Agent 架構模式](#2-multi-agent-架構模式)
3. [Gemini 模型整合](#3-gemini-模型整合)
4. [工廠導覽系統架構設計](#4-工廠導覽系統架構設計)
5. [Raspberry Pi 5 部署考量](#5-raspberry-pi-5-部署考量)
6. [完整範例程式碼](#6-完整範例程式碼)
7. [競品框架比較](#7-競品框架比較)
8. [產業應用趨勢](#8-產業應用趨勢)
9. [參考資源](#9-參考資源)

---

## 1. LangGraph 框架概覽

### 1.1 什麼是 LangGraph?

LangGraph 是 LangChain 團隊開發的 **狀態機導向 Agent 編排框架**，專為建構生產級、有狀態的 AI Agent 系統而設計。它是 MIT 授權的開源專案，被 Klarna、Replit、Elastic 等企業採用。

**核心概念：**
- **Nodes（節點）**：執行特定動作的函數（LLM 呼叫、工具執行等）
- **Edges（邊）**：節點之間的轉換路徑（包含條件分支）
- **State（狀態）**：在所有節點間共享、不可變且每步自動 checkpoint 的資料

### 1.2 目前版本與安裝

> **2026/03 重大更新**：LangGraph 2.0 於 2026 年 2 月正式發布，是三年來生產經驗的結晶，框架成熟度大幅提升。

```bash
# 核心套件（2026-03 最新）
pip install langgraph                    # LangGraph 2.0 核心
pip install langgraph-supervisor         # Supervisor 多代理模式
pip install langchain-google-genai       # Gemini 整合 (v4.1.2 stable)

# 需求：Python >= 3.10
```

### 1.3 LangGraph 2.0 新特性

#### Type-Safe Streaming 與 Invoke（v2 API）

LangGraph 2.0 引入 `stream_version="v2"` 參數，大幅提升型別安全性：

```python
# Type-safe invoke — 回傳 GraphOutput 物件而非 plain dict
result = app.invoke(
    {"messages": [{"role": "user", "content": "介紹組裝線A"}]},
    config=config,
    stream_version="v2",
)
# result.value — 你的 output type（Pydantic model / dataclass / dict）
# result.interrupts — 任何中斷事件

# Type-safe streaming — 每個 chunk 都是 StreamPart TypedDict
async for part in app.astream(
    {"messages": [{"role": "user", "content": "安全注意事項"}]},
    config=config,
    stream_version="v2",
):
    # part 有 type, ns, data 三個 key
    # 每種 mode 有對應的 TypedDict，可從 langgraph.types import
    print(part["type"], part["data"])
```

**v2 API 優點：**
- 使用 `stream_version="v2"` 時，Pydantic model 和 dataclass 會自動 coerce
- TypedDict 在 runtime 是 dict，`orjson.loads()` 直接對應正確型別，零額外開銷
- Python SDK（langgraph-sdk）也支援 `stream_version`，本地和遠端部署體驗一致
- 完全向後相容，`v2` 是 opt-in 的

#### 其他 2.0 新功能

| 功能 | 說明 |
|------|------|
| **Node Caching** | 跳過重複計算，提升效能 |
| **Deferred Nodes** | 支援 map-reduce 和 consensus 工作流 |
| **Pre/Post Model Hooks** | 在模型呼叫前後插入自訂邏輯 |
| **Built-in Provider Tools** | 內建 Web Search 和 RemoteMCP 工具 |
| **Bug Fix: Replay** | Replay 不再重用過期的 RESUME 值 |
| **Bug Fix: Subgraphs** | 子圖正確還原 parent 的歷史 checkpoint |

### 1.4 StateGraph 基礎

LangGraph 的核心是 `StateGraph`，使用 `TypedDict` 定義狀態結構：

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    """Agent 的共享狀態"""
    # Annotated + add_messages 確保訊息是「追加」而非「覆蓋」
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 追蹤目前由哪個 agent 處理
    next: str
    # 累積的 agent 工作歷史
    agent_history: Annotated[Sequence[BaseMessage], operator.add]
```

> **重要概念**：`Annotated[list, operator.add]` 表示每次節點回傳該欄位時，新值會**追加**到現有清單，而非取代。這是 LangGraph 狀態管理的核心機制。

---

## 2. Multi-Agent 架構模式

### 2.1 三種主要模式

LangGraph 支援三種 multi-agent 架構：

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| **Supervisor（監督者）** | 一個中央 agent 協調分派任務給專家 agent | 結構化工作流程、需要一致性的任務 |
| **Hierarchical（階層式）** | 多層 supervisor，supervisor 管理其他 supervisor | 大型團隊、複雜組織結構 |
| **Collaborative（協作式）** | Agent 之間平等溝通、自行決定任務分配 | 創意性任務、探索性問題 |

### 2.2 Supervisor 模式（推薦用於工廠導覽）

**Supervisor 模式** 最適合工廠導覽系統，因為導覽流程是結構化的：

```
使用者提問
    |
    v
[Supervisor Agent] ──── 判斷應由哪個專家處理
    |         |         |
    v         v         v
[導覽Agent] [安全Agent] [技術Agent]
    |         |         |
    └─────────┴─────────┘
              |
              v
      [Supervisor Agent] ──── 整合回覆或繼續分派
              |
              v
          回覆使用者
```

### 2.3 使用 langgraph-supervisor 套件

LangChain 官方提供了 `langgraph-supervisor` 套件，簡化 supervisor 模式的建構：

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# 建立專家 agents
guide_agent = create_react_agent(
    model=llm,
    tools=[get_factory_info, get_route_info],
    name="tour_guide",
    prompt="你是工廠導覽專家，負責介紹各個廠區和設施。"
)

safety_agent = create_react_agent(
    model=llm,
    tools=[get_safety_rules, get_emergency_info],
    name="safety_expert",
    prompt="你是工廠安全專家，負責回答安全規範和緊急應變問題。"
)

# 建立 supervisor 工作流
workflow = create_supervisor(
    agents=[guide_agent, safety_agent],
    model=llm,
    prompt="你是工廠導覽系統的總管，根據訪客問題分派給合適的專家。"
)

# 編譯並執行
app = workflow.compile()
result = app.invoke({
    "messages": [{"role": "user", "content": "這個區域有什麼安全注意事項？"}]
})
```

### 2.4 Handoff 機制

Supervisor 透過 **tool-based handoff** 機制將任務交給專家 agent：

- `create_handoff_tool()`：自訂 handoff 工具的名稱和描述
- `output_mode` 參數：
  - `"full_history"`：包含完整對話歷史
  - `"last_message"`：僅傳遞最終回應

### 2.5 記憶與持久化

```python
from langgraph.checkpoint.memory import InMemorySaver      # 記憶體（開發用）
from langgraph.checkpoint.sqlite import SqliteSaver         # SQLite（適合 Pi）
from langgraph.checkpoint.postgres import PostgresSaver     # PostgreSQL（生產用）

# 使用 SQLite 持久化（推薦用於 Pi 5）
checkpointer = SqliteSaver.from_conn_string("factory_tour.db")
app = workflow.compile(checkpointer=checkpointer)
```

### 2.6 Multi-Agent 最佳實踐（2026 年更新）

根據 2026 年最新基準測試和產業經驗：

1. **LLM 決策 vs 程式碼編排**：兩種方式各有優勢。LLM 決策靈活但不確定性高；程式碼編排（如 LangGraph 的條件邊）速度快、成本低、行為可預測。建議在工廠導覽等結構化場景使用**程式碼編排為主、LLM 決策為輔**的混合策略。

2. **效能基準**：根據多框架基準測試，LangGraph 在執行速度和狀態管理效率上表現最佳；CrewAI 因自主決策前的 deliberation 延遲最長。

3. **品質提升**：Multi-Agent 編排相較 Single-Agent 可達到：
   - 100% 可執行建議率（vs Single-Agent 的 1.7%）
   - 行動特異性提升 80 倍
   - 解決方案正確率提升 140 倍
   - 跨試驗零品質變異，可滿足生產 SLA 承諾

4. **記憶體共享**：確保不同 Agent 能高效共享和檢索相關資料，維持跨 Agent 的上下文連續性。

---

## 3. Gemini 模型整合

### 3.1 套件選擇

截至 2026 年 3 月，Google Gemini 的 LangChain 整合有以下選項：

| 套件 | 說明 | 狀態 |
|------|------|------|
| `langchain-google-genai` (v4.1.2) | 使用 Gemini Developer API | **推薦**，使用新版 `google-genai` SDK |
| `langchain-google-vertexai` | 使用 Vertex AI API | 企業級，需 GCP 帳號 |
| `@langchain/google` (JS) | 統一 JS 版本 | 取代舊版 JS 套件 |

> **注意**：`langchain-google-genai` v4.0.0 起已遷移至新版 `google-genai` SDK，不再使用已棄用的 `google-ai-generativelanguage`。

### 3.2 基本設定

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# 設定 API Key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# 初始化 Gemini 模型（建議使用最新模型）
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",    # 最新輕量模型，適合 agent 路由
    temperature=0.7,
    max_retries=2,
)

# 也可使用更強大的模型
llm_pro = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro",           # 最新 Pro 模型，複雜推理
    temperature=1.0,
)
```

### 3.3 可用模型建議（2026 年 3 月第四次更新）

| 模型 | 用途建議 | 延遲 | 成本 |
|------|----------|------|------|
| `gemini-3.1-flash-lite` | Supervisor 路由、簡單工具呼叫（**最新推薦**） | 最低 | 最低 |
| `gemini-3.1-pro` | 複雜推理、技術問答（**升級版核心推理**） | 中 | 中 |
| `gemini-3.1-flash-live` | 即時語音對話、客服體驗 | 低 | 低 |
| `gemini-2.0-flash` | 一般導覽對話、工具呼叫 | 低 | 低 |

> **2026/03 模型動態**：
> - `gemini-3.1-pro` — Google 最新旗艦模型，複雜推理能力大幅提升
> - `gemini-3.1-flash-live` — 最高品質即時語音模型，支援 200+ 國家，可透過 Gemini Live API 存取
> - `gemini-3.1-flash-lite`（03/03 發布）— 輕量快速模型，適合高頻率的 Agent 路由決策
> - 新增 Lyria 3 音樂生成模型（`lyria-3-clip-preview` / `lyria-3-pro-preview`）

> **重要淘汰通知**：
> - `gemini-2.5-flash-lite-preview-09-2025` 將於 **2026-03-31 停止服務**
> - `gemini-2.0-flash` 和 `gemini-2.0-flash-lite` 將於 **2026-06-01 停止服務**
> - 建議儘快遷移至 `gemini-3.1-flash-lite` 或 `gemini-3.1-pro`

> **新 API 功能**：
> - **Computer Use 工具**：`gemini-3-pro-preview` 和 `gemini-3-flash-preview` 支援電腦操作
> - **Built-in Tools + Function Calling 混合使用**：可在單一 API 呼叫中同時使用內建工具和自訂 function calling
> - **檔案大小限制提升**：從 20MB 增加至 100MB，支援 Cloud Storage 和 pre-signed URL

> **Pi 5 建議**：因為 LLM 推論在雲端（Google API），Pi 5 不需要跑本地模型。主要的效能考量是網路延遲和 Python 處理速度，而非 GPU/CPU 算力。建議 Supervisor 使用 `gemini-3.1-flash-lite` 降低延遲和成本。

### 3.4 Tool Binding

```python
from langchain_core.tools import tool

@tool
def get_factory_info(area_name: str) -> str:
    """取得工廠特定區域的介紹資訊。

    Args:
        area_name: 廠區名稱，例如 "組裝線A"、"品管室"、"倉儲區"
    """
    # 從資料庫或設定檔讀取
    factory_data = {
        "組裝線A": "這是我們的主要產線，每日產能 5000 件...",
        "品管室": "配備最新的 AOI 自動光學檢測設備...",
        "倉儲區": "採用自動化倉儲系統，FIFO 管理...",
    }
    return factory_data.get(area_name, f"找不到 {area_name} 的資訊")

# 綁定工具到模型
model_with_tools = llm.bind_tools([get_factory_info])
```

### 3.5 Google ADK（Agent Development Kit）

> **2026/03 新增**：Google 於 2026 年推出 Agent Development Kit（ADK），值得關注。

ADK 是 Google 推出的開源 Agent 開發框架，設計理念是讓 Agent 開發更像軟體開發：

- **多語言支援**：Python、Java（1.0.0 於 2026-03-30 發布）、Go、TypeScript
- **模型中立**：雖然為 Gemini 最佳化，但支援其他模型
- **Multi-Agent 設計**：支援階層式 Agent 組合和任務委派
- **豐富工具生態**：內建 Search、Code Execution，支援 MCP 工具，甚至可用其他 Agent 作為工具

```python
# Google ADK 範例（參考用）
# pip install google-adk
from google.adk.agents import Agent
from google.adk.tools import google_search

agent = Agent(
    name="factory_guide",
    model="gemini-3.1-flash-lite",
    instruction="你是工廠導覽員，用繁體中文回答問題。",
    tools=[google_search],
)
```

> **與 LangGraph 的關係**：ADK 是 Google 自家框架，與 LangGraph 在 Multi-Agent 編排上存在競爭。對於已投入 LangGraph 的專案，建議持續使用 LangGraph；新專案若深度依賴 Google 生態系，可考慮 ADK。

---

## 4. 工廠導覽系統架構設計

### 4.1 系統架構圖

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 5                      │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           LangGraph 2.0 Application           │   │
│  │                                               │   │
│  │  ┌─────────────────────────────────────────┐ │   │
│  │  │         Supervisor Agent                 │ │   │
│  │  │  (gemini-3.1-flash-lite, 路由決策)       │ │   │
│  │  └──────┬──────────┬──────────┬────────────┘ │   │
│  │         │          │          │               │   │
│  │    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐         │   │
│  │    │導覽Agent│ │安全Agent│ │技術Agent│         │   │
│  │    │介紹廠區 │ │安全規範 │ │技術細節 │         │   │
│  │    └────────┘ └────────┘ └────────┘          │   │
│  │                                               │   │
│  │  ┌──────────────────────┐                     │   │
│  │  │  SQLite State Store  │ (對話記憶/checkpoint)│   │
│  │  └──────────────────────┘                     │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ FastAPI  │  │ 知識庫    │  │  Pironman 5     │  │
│  │ Web 介面 │  │ JSON/MD  │  │  LED/風扇/OLED  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
          │
          │ Google Gemini API (HTTPS)
          ▼
    ┌─────────────┐
    │ Google Cloud │
    │ Gemini 3.1  │
    └─────────────┘
```

### 4.2 建議的 Agent 角色分工

| Agent | 職責 | 工具 |
|-------|------|------|
| **Supervisor** | 理解訪客意圖、分派任務、整合回覆 | handoff tools |
| **導覽 Agent** | 介紹各廠區、產線、設備 | `get_factory_info`, `get_route_map` |
| **安全 Agent** | 安全規範、個人防護裝備、緊急應變 | `get_safety_rules`, `get_emergency_procedures` |
| **技術 Agent** | 製程細節、設備規格、品質標準 | `get_technical_specs`, `get_process_flow` |
| **多語言 Agent**（選配） | 翻譯、多語言導覽支援 | `translate_text` |

### 4.3 知識庫設計

建議使用 JSON 檔案存放工廠知識，方便非技術人員維護：

```json
// factory_knowledge/areas.json
{
  "areas": [
    {
      "id": "assembly_a",
      "name": "組裝線A",
      "description": "主要 PCB 組裝產線，配備 SMT 貼片機...",
      "safety_notes": ["必須穿戴防靜電手環", "禁止攜帶飲料"],
      "technical_specs": {
        "daily_capacity": 5000,
        "equipment": ["SMT 貼片機 x3", "回焊爐 x1", "AOI x2"]
      },
      "tour_order": 1
    }
  ]
}
```

---

## 5. Raspberry Pi 5 部署考量

### 5.1 效能分析

**好消息**：因為 LLM 推論是透過 Gemini API 在雲端執行，Pi 5 的主要負擔是：
- Python 程式執行
- 網路請求處理
- 狀態管理（SQLite 讀寫）
- （選配）Web 伺服器

**Pi 5 (16GB) 完全能勝任這些任務。**

### 5.2 安裝建議

```bash
# 建議使用虛擬環境
python3 -m venv ~/factory-tour-env
source ~/factory-tour-env/bin/activate

# 安裝核心套件（2026-03 更新版本）
pip install langgraph langgraph-supervisor
pip install langchain-google-genai
pip install fastapi uvicorn           # Web API
pip install python-dotenv             # 環境變數管理

# piwheels 已有預編譯的 langgraph-supervisor
# (Pi OS 預設啟用 piwheels，安裝速度更快)

# 安全提醒：LiteLLM 1.82.7 和 1.82.8 於 2026-03-24 被發現含惡意代碼
# 如需使用 LiteLLM，請確保版本 >= 1.82.9 或 <= 1.82.6
```

### 5.3 效能最佳化技巧

1. **使用 `gemini-3.1-flash-lite`** 作為 Supervisor 的模型 — 回應快、成本低（取代即將淘汰的 `gemini-2.0-flash`）
2. **SQLite checkpointer** 比 PostgreSQL 更輕量，適合 Pi
3. **限制對話歷史長度** — 避免 token 數過多拖慢回應：
   ```python
   # 只保留最近 10 輪對話
   def trim_messages(messages, max_pairs=10):
       if len(messages) > max_pairs * 2:
           return messages[-(max_pairs * 2):]
       return messages
   ```
4. **使用 `output_mode="last_message"`** 減少 agent 間傳遞的訊息量
5. **非同步處理**：LangGraph 支援 async，善用 `ainvoke` 避免阻塞
6. **善用 LangGraph 2.0 Node Caching**：跳過重複計算，減少不必要的 API 呼叫
7. **使用 `stream_version="v2"`**：享受型別安全的同時，TypedDict 在 runtime 零額外開銷

### 5.4 Systemd 服務設定

```ini
# /etc/systemd/system/factory-tour.service
[Unit]
Description=Factory Tour Multi-Agent System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/factory-tour
Environment=PATH=/home/pi/factory-tour-env/bin:/usr/bin
EnvironmentFile=/home/pi/factory-tour/.env
ExecStart=/home/pi/factory-tour-env/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 6. 完整範例程式碼

### 6.1 最小可運行範例（Supervisor + Gemini，LangGraph 2.0）

```python
"""
factory_tour_agent.py - 工廠導覽 Multi-Agent 系統
在 Raspberry Pi 5 上運行，使用 LangGraph 2.0 + Gemini 3.1
"""
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ─── 模型初始化（使用最新模型） ───
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",       # 最新輕量模型
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# ─── 工具定義 ───
FACTORY_DATA = {
    "組裝線A": "主要 PCB 組裝產線，配備 3 台 SMT 貼片機，日產能 5000 片。",
    "品管室": "配備 AOI 自動光學檢測和 X-ray 檢測設備，不良率控制在 0.1% 以下。",
    "倉儲區": "自動化立體倉庫，採用 FIFO 先進先出管理，溫濕度即時監控。",
}

SAFETY_DATA = {
    "組裝線A": "必須穿戴防靜電手環和護目鏡，禁止攜帶飲料入內。",
    "品管室": "請勿觸碰檢測設備，需穿無塵衣。",
    "倉儲區": "注意堆高機通行，走在標示的人行通道上。",
}

@tool
def get_factory_info(area_name: str) -> str:
    """取得工廠特定區域的介紹資訊。area_name 可以是：組裝線A、品管室、倉儲區"""
    return FACTORY_DATA.get(area_name, f"找不到「{area_name}」的資訊。可用區域：{', '.join(FACTORY_DATA.keys())}")

@tool
def get_route_info() -> str:
    """取得建議的導覽路線"""
    return "建議路線：大廳 → 組裝線A → 品管室 → 倉儲區 → 會議室（約 45 分鐘）"

@tool
def get_safety_rules(area_name: str) -> str:
    """取得特定區域的安全規範。area_name 可以是：組裝線A、品管室、倉儲區"""
    return SAFETY_DATA.get(area_name, f"找不到「{area_name}」的安全規範。")

@tool
def get_emergency_info() -> str:
    """取得緊急應變資訊"""
    return "緊急出口位於每層樓梯旁。集合點在大門口停車場。緊急聯絡：分機 119。AED 位於一樓大廳。"

# ─── Agent 定義 ───
tour_guide = create_react_agent(
    model=llm,
    tools=[get_factory_info, get_route_info],
    name="tour_guide",
    prompt=(
        "你是工廠導覽員。用友善、專業的語氣介紹工廠各區域。"
        "回答要簡潔但資訊豐富，適合第一次參觀的訪客。使用繁體中文。"
    ),
)

safety_expert = create_react_agent(
    model=llm,
    tools=[get_safety_rules, get_emergency_info],
    name="safety_expert",
    prompt=(
        "你是工廠安全專家。清楚說明安全規範和注意事項。"
        "語氣嚴謹但不令人緊張。使用繁體中文。"
    ),
)

# ─── Supervisor 建構 ───
workflow = create_supervisor(
    agents=[tour_guide, safety_expert],
    model=llm,
    prompt=(
        "你是工廠導覽系統的總管。根據訪客的問題，決定由哪位專家回答：\n"
        "- tour_guide：廠區介紹、設備說明、導覽路線\n"
        "- safety_expert：安全規範、防護裝備、緊急應變\n"
        "用繁體中文回覆。如果問題同時涉及多個領域，先處理安全相關的部分。"
    ),
)

# ─── 編譯（使用 SQLite 持久化） ───
checkpointer = SqliteSaver.from_conn_string("factory_tour_state.db")
app = workflow.compile(checkpointer=checkpointer)

# ─── 互動迴圈（使用 v2 API） ───
def main():
    print("=== 工廠導覽 Multi-Agent 系統 (LangGraph 2.0) ===")
    print("輸入 'quit' 結束\n")

    config = {"configurable": {"thread_id": "tour-001"}}

    while True:
        user_input = input("訪客: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("感謝參觀，再見！")
            break
        if not user_input:
            continue

        # 使用 stream_version="v2" 取得型別安全的結果
        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_version="v2",
        )
        # v2 回傳 GraphOutput，用 .value 取得結果
        ai_message = result.value["messages"][-1]
        print(f"\n導覽系統: {ai_message.content}\n")

if __name__ == "__main__":
    main()
```

### 6.2 搭配 FastAPI 的 Web 版本

```python
"""
main.py - FastAPI Web 介面
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 假設 factory_tour_agent.py 中的 app 已匯入
from factory_tour_agent import app as agent_app

web = FastAPI(title="工廠導覽 Multi-Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@web.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    try:
        result = agent_app.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            stream_version="v2",
        )
        ai_message = result.value["messages"][-1]
        return ChatResponse(reply=ai_message.content, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(web, host="0.0.0.0", port=8000)
```

### 6.3 Streaming 範例（LangGraph 2.0 v2 API）

```python
"""
streaming_example.py - 使用 LangGraph 2.0 type-safe streaming
"""
import asyncio
from factory_tour_agent import app as agent_app

async def stream_response(user_input: str, session_id: str = "stream-001"):
    """使用 v2 streaming API 逐步輸出回應"""
    config = {"configurable": {"thread_id": session_id}}

    async for part in agent_app.astream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_version="v2",
    ):
        # v2 的每個 chunk 都是 StreamPart TypedDict
        # 包含 type, ns (namespace), data
        if part["type"] == "values":
            messages = part["data"].get("messages", [])
            if messages:
                latest = messages[-1]
                print(f"[{part['type']}] {latest.content[:80]}...")
        elif part["type"] == "updates":
            # 顯示哪個節點在處理
            for node_name, update in part["data"].items():
                print(f"  -> 節點 '{node_name}' 更新中...")

if __name__ == "__main__":
    asyncio.run(stream_response("請介紹組裝線A的設備和安全注意事項"))
```

### 6.4 從零手寫 StateGraph（不用 langgraph-supervisor）

如果需要更細緻的控制，可以手動建構 StateGraph：

```python
"""
手動建構 Supervisor StateGraph，提供更多客製彈性
"""
from typing import Annotated, Sequence, TypedDict, Literal
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import operator

# ─── 狀態定義 ───
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str

# ─── 路由決策結構 ───
class RouteDecision(BaseModel):
    next: Literal["tour_guide", "safety_expert", "FINISH"]

# ─── 模型與 Agent ───
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)

# (省略 tool 定義，同上例)

tour_guide = create_react_agent(llm, tools=[get_factory_info, get_route_info], name="tour_guide")
safety_expert = create_react_agent(llm, tools=[get_safety_rules, get_emergency_info], name="safety_expert")

# ─── Supervisor 節點 ───
def supervisor_node(state: AgentState):
    """Supervisor 分析對話，決定下一步"""
    system_msg = SystemMessage(content=(
        "你是導覽系統的路由器。根據最新的使用者訊息，決定交給哪個 agent：\n"
        "- tour_guide: 廠區介紹、導覽路線\n"
        "- safety_expert: 安全規範、緊急應變\n"
        "- FINISH: 對話可以結束\n"
        "回傳 JSON 格式 {\"next\": \"agent_name\"}"
    ))
    messages = [system_msg] + list(state["messages"])
    response = llm.with_structured_output(RouteDecision).invoke(messages)
    return {"next": response.next}

# ─── 建構圖 ───
workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tour_guide", tour_guide)
workflow.add_node("safety_expert", safety_expert)

# 設定入口
workflow.set_entry_point("supervisor")

# Agent 完成後回到 Supervisor
workflow.add_edge("tour_guide", "supervisor")
workflow.add_edge("safety_expert", "supervisor")

# Supervisor 的條件路由
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "tour_guide": "tour_guide",
        "safety_expert": "safety_expert",
        "FINISH": END,
    },
)

graph = workflow.compile()
```

---

## 7. 競品框架比較

> **2026/03 第四次更新新增章節**

### 7.1 主要 Multi-Agent 框架一覽

2026 年 Q1 是 Agent 框架爆發期，五家公司在同一個月內發布了 Agent 框架。以下是主要框架比較：

| 框架 | 開發者 | 特色 | 適用場景 |
|------|--------|------|----------|
| **LangGraph 2.0** | LangChain | 狀態機導向、最佳效能、生態系完整 | 複雜有狀態工作流 |
| **CrewAI** | CrewAI | 角色導向、快速原型、直覺語法 | 業務自動化、內容生成 |
| **AutoGen** | Microsoft | 對話導向、Code Execution | 技術應用、複雜推理 |
| **OpenAI Agents SDK** | OpenAI | Swarm 的生產版、tracing、guardrails | OpenAI 生態系整合 |
| **Google ADK** | Google | 多語言（Python/Java/Go/TS）、Gemini 最佳化 | Google 生態系整合 |
| **Deep Agents** | LangChain | 非同步子 Agent、非阻塞背景任務 | 複雜長時間任務 |
| **Pydantic AI** | Pydantic | 型別安全、驗證導向 | 需要嚴格資料驗證的場景 |

### 7.2 效能與適用性比較

| 面向 | LangGraph | CrewAI | AutoGen |
|------|-----------|--------|---------|
| **執行速度** | 最快 | 最慢（deliberation 延遲） | 中等 |
| **狀態管理** | 最佳 | 基本 | 中等 |
| **Token 消耗** | 中等 | 低（sequential 模式） | 較高（chat-heavy） |
| **上手時間** | 中等 | 最短（快 40%） | 較長 |
| **LLM 支援** | 廣泛（透過 LangChain） | 透過 LiteLLM | 任何 OpenAI-compatible API |
| **適合非工程師閱讀** | 否 | 是（角色語法） | 否 |

### 7.3 工廠導覽專案的框架選擇建議

對於本專案（Pi 5 + Gemini + 工廠導覽），**LangGraph 仍是最佳選擇**，理由：

1. **效能最佳**：在 Pi 5 有限的資源上，LangGraph 的執行速度和狀態管理效率最重要
2. **Gemini 整合成熟**：透過 `langchain-google-genai` 整合最完善
3. **生態系完整**：LangSmith 觀測、LangServe 部署等工具鏈齊全
4. **2.0 版本成熟**：三年生產經驗的結晶，穩定性有保障

### 7.4 LangChain 生態系新工具

| 工具 | 說明 | 發布時間 |
|------|------|----------|
| **Agent Builder** | 用自然語言描述需求，自動生成 Agent 定義（prompt、工具、子 Agent） | 2026-01 |
| **Insights Agent** | 自動分析 traces，偵測使用模式和失敗模式（支援 self-hosted LangSmith） | 2026-01 |
| **Deep Agents v0.5.0** | 非同步子 Agent，支援非阻塞背景任務 + 多模態檔案（PDF、音頻、視頻） | 2026-03 |
| **LangSmith Fleet** | 原 Agent Builder 改名，企業級 Agent 協調平台（原 LangSmith Agent Builder） | 2026-03 |
| **LangChain JS v1.2.13** | 動態工具、從 hallucinated tool call 復原、更好的 streaming 錯誤信號 | 2026-01 |

> **2026-03 安全提醒**：LangGraph SQLite checkpoint 實現發現 SQL injection 漏洞（可透過 metadata filter key 操作 SQL 查詢）。建議：
> - 若使用 SqliteSaver，立即更新至最新版本
> - 考慮改用 PostgreSQL checkpoint（更安全）或 in-memory 模式（測試用）
> - 檢查現有系統的 metadata 驗證邏輯

---

## 8. LangGraph 2026 最新特性更新

> **2026-04 新增章節**：LangGraph 2.0 核心特性強化

### 8.1 型別安全與流式處理升級

LangGraph 2026 Q1 發布了**全新 v2 API**，為流式和同步呼叫增添型別安全：

- **Type-safe streaming**：`stream_version="v2"` 回傳統一的 `StreamPart` TypedDict，每個 chunk 包含 `type`, `ns`, `data` 三個鍵
- **Type-safe invoke**：`invoke_version="v2"` 回傳 `GraphOutput` 物件，含 `.value` 和 `.interrupts` 屬性
- **Pydantic/dataclass 自動轉型**：輸出自動強制轉換到聲明的 Pydantic 模型或 dataclass 型態

### 8.2 運維與中間件增強

新增三個生產級中間件與工具鏈：

1. **Model Retry Middleware**：模型呼叫失敗自動重試，支援指數退避策略
2. **Content Moderation Middleware**：OpenAI 內容審核整合，自動偵測並處理不安全輸入
3. **LangGraph Deploy CLI**：終端一行指令即可將 Agent 部署至 LangSmith，簡化上線流程

### 8.3 高級控制特性

**固定時間旅行（Time Travel）**：修復 interrupt 和子圖的 checkpoint 復原邏輯，重放操作不再複用過期狀態，子圖也能正確恢復父圖的歷史狀態。

**模型 Profile**：所有聊天模型現暴露 `.profile` 屬性，可查詢支援的特性和能力（資料來自開源的 models.dev 專案）。

---

## 8. 產業應用趨勢

> **2026/03 第四次更新新增章節**

### 8.1 Agentic AI 在製造業的普及

2026 年初，Agentic AI 已從概念走向實際部署：

- **全球採用率**：近半數全球製造業已整合智慧系統（Deloitte 調查：75% 企業計畫在兩年內部署 Agentic AI）
- **Gartner 預測**：40% 企業應用將在 2026 年底部署 Multi-Agent Swarms
- **效能數據**：AI 系統分析即時感測器資料，平均效率提升 31%，非計畫停機減少 43%

### 8.2 從 Co-pilot 到 Agent

2026 年的關鍵轉變是從 **Co-pilot**（被動等待人類提問）到 **Agent**（主動觀察、推理、行動）：

- **自我優化工廠**：AI Agent 不只是標記異常，而是自動檢查生產排程、判斷原因、調整機器參數、產生維護工單
- **Samsung 策略**：宣布 2030 年前將全球工廠轉型為 AI 驅動工廠
- **Siemens eXplore Tour**：使用互動式體驗展示工業自動化技術的巡迴展覽

### 8.3 對工廠導覽系統的啟示

這些產業趨勢對我們的工廠導覽系統有重要啟示：

1. **從靜態導覽到動態導覽**：Agent 可以根據即時產線狀態調整導覽內容（例如：「目前組裝線A正在進行 X 產品的生產，日產能 5000 片」）
2. **整合 IoT 感測器**：Pi 5 可連接工廠的 IoT 感測器，讓導覽 Agent 提供即時數據
3. **預測性維護展示**：導覽時展示 AI 如何預測設備故障，增加訪客對工廠智慧化的認知

---

## 9. 參考資源

### 官方文件
- [LangGraph 官方網站](https://www.langchain.com/langgraph)
- [LangGraph Releases（含 2026-03 更新）](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph 文件 (Quickstart)](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangGraph 1.0 GA 公告](https://changelog.langchain.com/announcements/langgraph-1-0-is-now-generally-available)
- [LangChain Changelog](https://changelog.langchain.com/)
- [langgraph-supervisor GitHub](https://github.com/langchain-ai/langgraph-supervisor-py)
- [langchain-google-genai PyPI](https://pypi.org/project/langchain-google-genai/)
- [LangChain Google 整合文件](https://docs.langchain.com/oss/python/integrations/providers/google)
- [Google ADK 文件](https://google.github.io/adk-docs/)
- [Google ADK Python GitHub](https://github.com/google/adk-python)

### Gemini 模型
- [Gemini API 模型清單](https://ai.google.dev/gemini-api/docs/models)
- [Gemini API Release Notes](https://ai.google.dev/gemini-api/docs/changelog)
- [Gemini 3.1 Pro 公告](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/)
- [Gemini 3.1 Flash Live 公告](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-live/)

### 教學與範例
- [ReAct agent from scratch with Gemini 2.5 and LangGraph](https://www.philschmid.de/langgraph-gemini-2-5-react-agent)
- [Building a Multi-Agent System with LangGraph and Gemini](https://medium.com/@ipeksahbazoglu/building-a-multi-agent-system-with-langgraph-and-gemini-1e7d7eab5c12)
- [Google AI: ReAct agent with Gemini API](https://ai.google.dev/gemini-api/docs/langgraph-example)
- [LangGraph Multi-Agent Systems Tutorial](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-systems-complete-tutorial-examples)
- [DataCamp: How to Build LangGraph Agents](https://www.datacamp.com/tutorial/langgraph-agents)
- [Multi-Agent System Tutorial with LangGraph](https://blog.futuresmart.ai/multi-agent-system-with-langgraph)
- [LangGraph 2.0 Guide (DEV)](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)

### 框架比較與最佳實踐
- [Multi-Agent LLM Systems: Frameworks, Architecture & Examples (2026)](https://sourcebae.com/blog/multi-agent-llm/)
- [How to Build Multi-Agent Systems: Complete 2026 Guide](https://dev.to/eira-wexford/how-to-build-multi-agent-systems-complete-2026-guide-1io6)
- [The Great AI Agent Showdown of 2026](https://topuzas.medium.com/the-great-ai-agent-showdown-of-2026-openai-autogen-crewai-or-langgraph-7b27a176b2a1)
- [CrewAI vs LangGraph vs AutoGen vs OpenAgents (2026)](https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared)
- [LLM Orchestration in 2026: Top 22 Frameworks](https://aimultiple.com/llm-orchestration)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)

### 產業應用
- [Factory Automation 2026: AI Gains & Cobot Growth](https://www.autonexcontrol.com/blogs/news/industrial-automation-2026-smart-manufacturing-hits-47-global-adoption)
- [Smart Factories in 2026: How AI Is Transforming Manufacturing](https://ifactory.jrsinnovation.com/blog/smart-factories-2026-ai-manufacturing)
- [Top Smart Factory Technologies 2026: Agentic AI](https://www.iiot-world.com/smart-manufacturing/top-smart-factory-technologies-2026-agentic-ai-uns/)
- [Samsung AI-Driven Factories Strategy](https://news.samsung.com/global/samsung-electronics-announces-strategy-to-transition-global-manufacturing-into-ai-driven-factories-by-2030)

---

## 10. LangGraph 2026 Q1 關鍵發展（2026/03/10 更新）

> **最新技術動態**

LangGraph 在 2026 Q1 推出多項重要改進，強化生產級應用支援：

- **Type-Safe APIs (v2)**：統一 StreamPart 輸出格式（type、ns、data 鍵），invoke() 自動型別轉換為 Pydantic/dataclass 物件
- **Interrupt & Time Travel 增強**：修復 subgraph 快照復原，避免重放時重用過時 RESUME 值
- **Model Profiles & Middleware**：透過 .profile 屬性暴露模型特性，新增 retry 和內容審核 middleware
- **Node Caching**：快取個別節點結果，加快迭代開發和減少冗餘運算，特別適用於 map-reduce 工作流
- **Deferred Nodes**：延遲執行直到上游路徑完成，適合多 Agent 協作和聚合型任務

這些改進使 LangGraph 在資源受限的環境（如 Pi 5）上執行複雜 Multi-Agent 系統時，更具穩定性與效能優勢。

### 補充：LangGraph 串流與中介軟體最佳實踐

根據 2026 最新文件，LangGraph 新增兩大核心能力，特別適用於工廠導覽 Multi-Agent 場景：

1. **Type-Safe Streaming (v2)**：透過 `stream(version="v2")` 統一傳回 StreamPart，每個 chunk 包含 type（如 "on_chat_model_stream"）、ns（命名空間）、data 三個鍵。於工廠 tour 應用中，可精確追蹤各 Agent 的執行狀態與輸出，便於前端實時展示進度。

2. **Model Profiles & Middleware**：LangGraph 2026 支援 `.profile` 屬性直接查詢 LLM 能力（如支援函數呼叫、vision、長上下文等），新增「模型 retry middleware」和「OpenAI 內容審核 middleware」。於工廠環境，可根據模型能力動態調度專業 Agent（文本分析 vs 視覺檢測），同步確保安全合規。

這兩項更新大幅降低開發者在串流調試與異常處理的成本。

---

## 11. LangGraph 2.0 Output Coercion — 自動型別轉換（2026/04/01 補充）

> **實務應用指南**

LangGraph 2.0 新增 **Output Coercion** 功能，自動將 agent 輸出轉換為聲明的 Pydantic 模型或 dataclass，顯著降低後處理邏輯的複雜度。

### 實際應用範例

在工廠導覽系統中，若要確保每筆回應符合結構化格式：

```python
from pydantic import BaseModel
from typing import List

class FactoryTourResponse(BaseModel):
    """結構化導覽回應"""
    main_content: str          # 核心回答
    safety_warnings: List[str] # 安全提示清單
    follow_up_question: str    # 建議後續問題

# 使用 v2 API + output coercion
result = app.invoke(
    {"messages": [{"role": "user", "content": user_input}]},
    config=config,
    stream_version="v2",
)

# result.value 自動轉換為 FactoryTourResponse 物件
response: FactoryTourResponse = result.value
print(f"安全提示：{response.safety_warnings}")
```

此特性減少了手動 JSON 解析和驗證的繁瑣，特別適合在 Raspberry Pi 上執行的資源受限環境。搭配 Node Caching，可進一步優化連續多輪對話的處理效率。

---

## 12. LangGraph Agent Middleware 與模型能力探測（2026/04/01 更新）

> **生產環節關鍵新增**

LangChain 1.1（2025 年 12 月）與 LangGraph 2026 Q2 更新引入了 **Agent Middleware** 與 **Model Profiles** 機制，進一步強化了 Agent 可靠性與智慧化調度能力。

### 12.1 Model Profiles 自動能力探測

模型現在透過 `.profile` 屬性暴露其支援的能力集合，包括：
- 結構化輸出（Structured Output）
- 函數呼叫（Function Calling）
- JSON 模式支援
- Vision 能力
- 長上下文支援
- 工具使用能力

在工廠導覽場景，可據此動態選擇最適合的模型與 Agent：

```python
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro")

# 查詢模型能力
if llm.model.profile.supports("vision"):
    # 使用視覺檢測 Agent（例如 AOI 缺陷分析）
    vision_agent = create_vision_inspector()
elif llm.model.profile.supports("structured_output"):
    # 使用結構化資料提取 Agent
    data_agent = create_data_extractor()
```

### 12.2 新增 Middleware 系統

LangGraph 新增三類核心 middleware：

1. **模型 Retry Middleware**：自動重試失敗的模型呼叫，支援指數退避（exponential backoff）
2. **內容審核 Middleware**（OpenAI Content Moderation）：檢測並處理不安全內容，適合公開導覽場景
3. **Summarization Middleware**：利用 Model Profiles 決策何時總結長對話，維持上下文同時降低 token 消耗

這些 middleware 特別適合在 Pi 5 + Gemini 的組合上使用，因為網路不穩定時可自動重試，同時監控對話安全性。

---

## 13. LangGraph 2.0 型別安全 API 與新 Middleware（2026/02 - 03）

> **生產級框架成熟度里程碑**

LangGraph 2.0（2026 年 2 月）正式發佈三年來生產環節驗證的最佳實踐。3 月 10 日推出了型別安全的串流與呼叫 API：

- **version="v2" 型別安全串流**：所有輸出統一為 `StreamPart`，帶 `type`、`ns`、`data` 三個鍵，直接相容 Pydantic 或 dataclass 自動強制轉換
- **Pydantic/dataclass 自動強制轉換**：`invoke()` 與 `stream()` 的輸出自動轉換為宣告的型別，減少手動序列化
- **Model Profiles 動態能力探測**：每個 LLM 透過 `.profile` 屬性暴露結構化輸出、函數呼叫、Vision 等能力，Agent 可據此動態調度
- **新增三大 Middleware**：Retry（指數退避重試）、Content Moderation（內容審核）、Summarization（上下文優化），特別適合 Pi 5 + 雲端 LLM 的組合

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangGraph Release Week Recap](https://blog.langchain.com/langgraph-release-week-recap/)
- [Agent Middleware](https://blog.langchain.com/agent-middleware/)

---

## 14. LangGraph v2 時間旅行與協議支援增強（2026/03）

> **多代理系統成熟度新高峰**

LangGraph v2 在 3 月進一步強化：

- **時間旅行與中斷恢復**：支援 replay 功能，可恢復歷史檢查點狀態，同時修正子圖（subgraphs）不重用陳舊 RESUME 值的問題，確保複雜多層代理系統的狀態一致性
- **跨協議代理通訊**：整合 MCP（Model Context Protocol）與 A2A（Agent-to-Agent communication），使多代理系統能跨不同框架協作，適合 OpenClaw 多渠道架構擴展
- **GraphOutput 型別物件**：`invoke()` 回傳統一的 GraphOutput 物件含 `.value` 與 `.interrupts` 屬性，簡化複雜工作流的狀態管理

Sources:
- [Before You Upgrade to LangGraph in 2026, Read ...](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)
- [LangGraph 2.0: The Definitive Guide to Building Production-Grade AI Agents in 2026 - DEV Community](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)
- [LangGraph Explained (2026 Edition) | by Dewasheesh Rana | Medium](https://medium.com/@dewasheesh.rana/langgraph-explained-2026-edition-ea8f725abff3)

---

## 15. LangGraph v2 性能優化 — 節點級快取與延遲節點（2026/02-04）

> **Pi 5 + 多代理工作流最佳化**

LangGraph 2.0 導入節點級效能最佳化機制：

1. **節點與工作級快取**（Node/Task Level Caching）：支援快取個別節點的執行結果，減少重複計算，特別適合 Pi 5 受限的記憶體與 CPU 環境，可顯著加速複雜圖遍歷
2. **延遲節點支援**（Deferred Nodes）：使用者可標記節點在所有上游路徑完成後才執行，簡化依賴管理，適合 factory-tour 多代理協調場景
3. **模型能力探測進階化**：Chat models 透過 `.profile` 屬性暴露結構化輸出、函數呼叫等支援能力，Agent 可動態選擇最佳模型路徑，進一步降低 Gemini API 調用成本

Sources:
- [LangGraph 2.0 Release - LangChain Changelog](https://changelog.langchain.com/)
- [LangGraph Release Week Recap](https://blog.langchain.com/langgraph-release-week-recap/)

---

## 16. LangGraph 2026 多代理協作與 LangSmith 可觀測性整合

> **2026/04/01 產業動態更新**

根據最新開發動態，LangGraph 2026 對多代理系統的協作與監控能力有重大進展。Klarna、Replit、Elastic 等領先企業已驗證以下特性在生產環境的可靠性：

1. **動態 Sub-Agent 生成**：代理可在執行時動態產生子代理，無需預先定義所有專業代理，大幅提升工廠導覽等複雜場景的適應性
2. **LangSmith 深度集成**：每個代理呼叫、工具使用、狀態轉移都被自動追蹤與記錄至 LangSmith，開發者可實時監控整個多代理工作流的執行細節，快速定位瓶頸與故障
3. **人工干預與狀態檢視**：支援在執行中斷點插入人工審查，允許修改代理狀態後恢復執行，適合風險敏感的工業應用場景

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work - DEV Community](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangGraph: Agent Orchestration Framework for Reliable AI Agents](https://www.langchain.com/langgraph)

---

## 17. LangGraph Cloud 與分佈式多代理執行（2026/04 展望）

> **邊緣計算 + 雲端協調的新範例**

2026 年 LangGraph 生態進一步成熟，新增 LangGraph Cloud 託管服務與跨機器分佈式執行能力。對於 Roy 的 Pi 5 + OpenClaw 多渠道架構具有特別意義：

- **LangGraph Cloud 託管執行**：允許複雜的工業級多代理工作流運行在 LangChain 託管基礎設施上，自動處理伸縮、監控與故障恢復，同時保留與本地 Pi 邊緣計算的無縫集成
- **邊緣 + 雲端協調**：Pi 5 可運行輕量代理（如路線規劃、本地狀態管理），重計算任務（LLM 推理、複雜分析）委派至 LangGraph Cloud，充分利用 Gemini API 與邊緣資源的分層架構
- **多 Agent 子進程動態生成**：支援在雲端動態產生子代理負責特定任務（例如 Factory Tour 中各產線的獨立導覽代理），Cloud 自動調度與監控，Pi 本地層僅需協調入口邏輯

Sources:
- [LangGraph Release Week Recap](https://blog.langchain.com/langgraph-release-week-recap/)
- [LangChain - Changelog](https://changelog.langchain.com/)
- [LangGraph: Build Stateful Multi-Agent Systems That Don't Crash](https://www.mager.co/blog/2026-03-12-langgraph-deep-dive/)

---

## 18. SQLite Checkpoint 安全更新與 LangGraph 版本管理（2026/04 重要補丁）

> **生產部署必讀安全警告**

2026 年 3 月底發現 LangGraph SQLite checkpoint 實作存在 SQL 注入漏洞（CVE-2025-67644，CVSS 7.3），攻擊者可透過 metadata filter 鍵值操縱 SQL 查詢。對於 Pi 5 上運行的 Factory Tour 系統特別關鍵：

- **受影響版本**：所有在 2026-03-24 前發佈的版本，包括安裝教程中提及的 `langgraph-supervisor`
- **修復方案**：立即升級至最新補丁版本（具體版本號待 LangChain 官方發佈），並檢查 `.env` 中的 SQLite 資料庫路徑權限
- **Pi 5 部署建議**：定期執行 `pip install --upgrade langgraph`，設定 crontab 自動檢查更新，並備份 `~/.local/share/langgraph/checkpoints.db` 避免狀態遺失

Sources:
- [LangChain, LangGraph Flaws Expose Files, Secrets, Databases](https://thehackernews.com/2026/03/langchain-langgraph-flaws-expose-files.html)
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)

---

## 19. LangGraph 2.0 核心新功能與 2026 年最佳實踐

**LangGraph 2.0 於 2026 年 2 月正式發佈**，整合三年的生產環境經驗，引入多項企業級特性：

1. **LangGraph Cloud** — 託管執行與內置監控，降低本地部署負擔
2. **Deep Agents** — 能夠動態生成子智能體、使用工具與檔案系統完成複雜任務
3. **MCP（模型上下文協議）與 A2A（智能體間通信）** — 標準化智能體間的協作與工具連接
4. **持久執行與人類在迴圈中** — 支援長時間執行、檢查點恢復與人工干預
5. **LangSmith Fleet**（2026 年 3 月上線，原名 Agent Builder）— 新增成本追蹤與基準測試功能

對於 Pi 5 上的 Factory Tour 系統，建議優先使用 `LangGraph 2.0` 的 Node Caching 與非同步 API，減少 API 呼叫並改進回應時間。

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangGraph 2.0: The Definitive Guide to Building Production-Grade AI Agents](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)

---

## 20. Agentic AI 市場爆炸與 NVIDIA + LangChain 企業級聯手（2026 年市場洞察）

> **AI Agent 產業規模將翻倍，LangGraph 成企業標配**

根據 Gartner 與業界報告，2026 年企業應用嵌入 AI agents 的比例急速攀升至 **40%**（相比 2025 年不足 5%）。Agentic AI 市場規模預計從 2026 年的 78 億美元成長至 2030 年的 520 億美元，複合年增長率達 **77%**。LangChain 在 2026 年 3 月宣佈與 NVIDIA 深度整合，推出企業級 Agentic AI 開發平台，LangGraph 框架下載量已突破 10 億次。LangGraph 在所有基準測試中實現最低延遲與 Token 消耗，其 DAG 圖結構預先決定每步工具執行，將 LLM 介入降至最低，特別適合對回應時間敏感的 Pi 5 + OpenClaw 多渠道系統部署。

Sources:
- [Agentic AI with LangGraph: Orchestrating Multi-Agent Workflows in 2026](https://adspyder.io/blog/agentic-ai-with-langgraph/)
- [Agentic AI Frameworks 2026: LangGraph, CrewAI, LangChain](https://byteiota.com/agentic-ai-frameworks-2026-langgraph-crewai-langchain/)
- [LangChain Announces Enterprise Agentic AI Platform Built with NVIDIA](https://blog.langchain.com/nvidia-enterprise/)
- [Adding Long-Term Memory to LangGraph and LangChain Agents](https://hindsight.vectorize.io/blog/2026/03/24/langgraph-longterm-memory)

---

## 21. LangGraph v1.1 型別安全與部署 CLI 革新（2026/03 月度更新）

> **開發效率與生產部署同步躍進**

2026 年 3 月，LangGraph 發佈 v1.1 版本，帶來多項生產級改進，特別針對 Pi 5 上的複雜多代理系統優化。版本 v1.1 引入 **Type-Safe Streaming** 與 **Type-Safe Invoke**，透過 version="v2" 參數實現統一的 StreamPart 輸出格式（包含 type、ns、data 鍵）；同時支援 Pydantic 與 dataclass 自動強制轉換，確保代理狀態管理的型別安全。新增 **LangGraph Deploy CLI** 工具，僅需一行命令即可將複雜的工業級代理直接部署至 LangSmith Deployment 基礎設施，無需手工設定。此外，Chat Models 新增 `.profile` 屬性（源自開源 models.dev 專案），代理可自動偵測 LLM 能力；框架級重試中間件與 OpenAI 內容審核中間件降低故障風險，適合對可靠性有高需求的 Factory Tour 與 nRF54L15 監測系統。

Sources:
- [LangChain - Changelog | LangGraph Workflow Updates (Python & JS)](https://changelog.langchain.com/announcements/langgraph-workflow-updates-python-js)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)

---

## 22. LangSmith Fleet 統一 API Gateway 與企業級多代理協調（2026/03 企業方案）

> **消除 AI 應用孤島，實現跨部門統一 LLM 網關**

2026 年 LangChain 推出 LangSmith Fleet，專門解決企業組織中 AI 應用碎片化問題。Fleet 內置 API Gateway 功能，允許不同部門的開發團隊各自在本地用 LangGraph 開發多代理工作流，隨後無縫過渡至生產環境，無需重寫代碼。統一的 API Gateway 層整合 OpenAI、Anthropic、Gemini 等多模型提供商，開發者可透過中央配置切換或混搭 LLM，大幅降低成本與鎖定風險。對 Roy 的 Pi 5 + OpenClaw 多渠道系統而言，LangSmith Fleet 可作為輕量級代理協調層，管理 Factory Tour、nRF54L15 監測等多個獨立 LangGraph 工作流的統一入口，同時保留邊緣本地執行的靈活性。

Sources:
- [Introducing LangSmith Fleet for Enterprise Agent Management](https://explore.n1n.ai/blog/introducing-langsmith-fleet-enterprise-agent-management-2026-03-20)

---

## 23. LangSmith Fleet 企業級身份管理與 Deploy CLI 一鍵部署（2026/04 新增功能）

> **從開發到生產，一個指令全搞定**

LangSmith Fleet 在 2026 年進一步強化企業級協作能力，新增 **Agent Identity、Sharing 與 Permissions** 機制，允許組織內不同角色（開發者、審核者、運維人員）針對多代理工作流進行精細化權限控制。同時，**LangGraph Deploy CLI** 工具大幅簡化部署流程，開發者無需手動設定即可在一行命令內將本地開發的複雜 LangGraph 工作流直接推送至 LangSmith Deployment 基礎設施，真正實現「開發環境 → 生產環境」的無縫過渡。對於 Pi 5 上的 OpenClaw 多渠道系統而言，這意味著可快速迭代 Factory Tour 與 nRF54L15 監測代理，並透過統一的身份與權限層保護關鍵應用邏輯。

---

## 24. Agentic RAG 與長期記憶整合——Hindsight + LangGraph 協作（2026/04 最新進展）

> **自主決策檢索系統，超越靜態管線**

2026 年中期，LangGraph 生態與記憶系統的融合達到新高度。Hindsight 0.4.20（2026 年 3 月）正式整合 LangGraph，使代理可直接使用 Hindsight 長期記憶層作為工具或節點內的 BaseStore API。與過往固定序列的 RAG 不同，現代 Agentic RAG 系統現已演化為完整的決策智能體：代理無需預定義檢索步驟，而是自主規劃、檢索、推理、批判、重寫、反思，直到達成高信心答案或耗盡預算。Pi 5 上的 Factory Tour 系統可利用此模式，讓導覽代理透過 Hindsight 存儲歷史導覽軌跡與遊客偏好，在後續互動中自適應改進講解策略。LangSmith Fleet 的統一成本追蹤機制進一步解決傳統 RAG 的隱性開銷問題——不再只計 LLM token，而是整個代理工作流的完整成本。

Sources:
- [Next-Generation Agentic RAG with LangGraph (2026 Edition)](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8)
- [What's new in Hindsight 0.4.20](https://hindsight.vectorize.io/blog/2026/03/24/version-0-4-20)

Sources:
- [LangGraph: Agent Orchestration Framework for Reliable AI Agents](https://www.langchain.com/langgraph)
- [LangGraph Agents in Production: Build Stateful AI Workflows with Python (2026)](https://use-apify.com/blog/langgraph-agents-production)
- [Beyond Single Agents: How to Build Collaborative AI Workflows with LangGraph](https://levelup.gitconnected.com/beyond-single-agents-how-to-build-collaborative-ai-workflows-with-langgraph-ead1f48f9534)

---

## 24. 代理協作協議標準化：MCP 與 A2A（2026 年生態成熟）

> **「USB for Agents」— 通用連接協議成為業界標準**

2026 年 LangChain 生態見證了代理通訊協議的標準化浪潮。兩大協議正式確立：**MCP（Model Context Protocol）** 專責代理與工具的連接（Agent-Tool），而 **A2A（Agent-to-Agent Protocol）** 則管理代理之間的動態協作。MCP 已成為業界標準「USB for Agents」，使工具提供商可開發與多框架相容的工具模組，避免 LangGraph、CrewAI、AutoGen 等框架各自為政。對於 Roy 的 Pi 5 + OpenClaw + Factory Tour 多層架構而言，MCP 意味著無論 nRF54L15 監測子代理或 Gemini RAG 檢索代理，皆可以統一接口與主協調器通訊，無需針對不同框架重複開發工具層。同時，A2A 協議支援代理動態生成與任務委派，Factory Tour 中的產線導覽代理可運行時自動啟動，完成任務後自毀，大幅提升資源利用效率與系統靈活性。

Sources:
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)

---

## 25. LangGraph Checkpoint 機制與容錯能力深化（2026 年穩定性提升）

> **狀態持久化與故障恢復成為核心競爭力**

LangGraph 在 2026 年強化了 Checkpoint 持久化機制，成為業界最成熟的多代理容錯方案。每次狀態轉移均被自動持久化，支援時間旅行除錯、人類在迴圈中暫停恢復，以及執行中間故障恢復。特別適用於 Pi 5 上運行的長時間 Factory Tour 導覽或 nRF54L15 監測任務。此外，Deep Agents 新特性允許代理在執行時動態生成子代理負責特定任務，子代理可自主規劃、使用工具與檔案系統，完成後自動銷毀，充分發揮分佈式多代理的伸縮能力。

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work - DEV Community](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [How to Design a Production-Grade Multi-Agent Communication System Using LangGraph](https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/)

---

## 26. LangGraph v1.1 型別安全流與模型容錯中間件（2026/03 後期發佈）

> **開發穩定性與部署可靠性的雙重躍進**

LangGraph v1.1 推出 **Type-Safe Streaming** 與 **Type-Safe Invoke** 兩大特性，透過 `version="v2"` 參數實現統一的 StreamPart 輸出格式（包含 type、ns、data 鍵），每個 chunk 均已型別定義，減少執行時資料驗證負擔。同時導入 **Model Retry Middleware**，自動以指數退避策略重試失敗的 LLM 呼叫，並新增 **OpenAI Content Moderation Middleware** 應對不安全內容檢測。對 Pi 5 上運行的長時間 Factory Tour 與 nRF54L15 監測代理而言，這些容錯機制顯著提升系統穩定性，減少因 API 波動或臨時故障導致的代理崩潰風險。

Sources:
- [LangChain - Changelog | LangGraph Workflow Updates (Python & JS)](https://changelog.langchain.com/announcements/langgraph-workflow-updates-python-js)
- [Before You Upgrade to LangGraph in 2026, Read ...](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)

---

## 27. LangGraph Cloud 與生產就緒新時代（2026 年企業級部署）

> **從 Pilot 到規模化：填補 90% 的執行鴻溝**

2026 年 LangGraph 推出 **LangGraph Cloud**，提供雲端託管執行環境與內建監控儀表板。根據業界數據，儘管 40% 的企業應用已採用任務特定代理，但僅 10-15% 的 Pilot 專案成功晉級生產環境。LangGraph Cloud 透過結構化編排、Durable Execution（代理崩潰後自動恢復）與 Human-in-the-Loop 工作流，直接消除執行風險。同時強化 LangSmith 整合，對每次代理呼叫提供完整追蹤與可視化。對 Roy 的 Pi 5 + Factory Tour 與 nRF54L15 監測系統而言，這意味著長時間運行的複雜多代理工作流可獲得企業級穩定性與可觀測性，大幅降低維運成本。

Sources:
- [LangGraph: Agent Orchestration Framework for Reliable AI Agents](https://www.langchain.com/langgraph)
- [Agentic AI with LangGraph: Orchestrating Multi-Agent Workflows in 2026](https://adspyder.io/blog/agentic-ai-with-langgraph/)
- [Definitive Guide to Agentic Frameworks in 2026](https://softmaxdata.com/blog/definitive-guide-to-agentic-frameworks-in-2026-langgraph-crewai-ag2-openai-and-more/)

---

## 28. LangGraph 原生流式輸出與全球企業規模化部署（2026 年成熟階段）

> **代理推理過程透明化，規模化部署突破 9000 萬月活**

LangGraph 在 2026 年強化原生 **Token-by-Token Streaming** 能力，支援實時展示代理推理過程、工具呼叫與狀態更新，讓使用者親眼見證代理每一步決策。搭配 **Durable Execution** 機制，代理即使中斷也能從精確檢查點恢復，無需重新執行，特別適合 Pi 5 上的長時間 Factory Tour 導覽與 nRF54L15 監測任務。**Human-in-the-Loop** 工作流則允許代理執行中暫停，等待人類輸入後無縫繼續，無需阻塞執行緒。截至 2026 年，LangGraph 月活下載量達 9000 萬次，全球企業部署規模包括 Uber、JP Morgan、BlackRock、Cisco、LinkedIn、Klarna 等行業龍頭，成為可控、有狀態 AI 代理的業界標準。

Sources:
- [LangGraph Agents in Production: Build Stateful AI Workflows with Python (2026)](https://use-apify.com/blog/langgraph-agents-production)
- [LangChain Releases Deep Agents: Structured Runtime for Planning, Memory, and Context Isolation](https://www.marktechpost.com/2026/03/15/langchain-releases-deep-agents-a-structured-runtime-for-planning-memory-and-context-isolation-in-multi-step-ai-agents/)
- [Before You Upgrade to LangGraph in 2026, Read ...](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)

---

## 29. LangGraph Deploy CLI 與一鍵雲端部署（2026 年 3 月發佈）

> **開發者工作流簡化，代理從本地跳躍至生產環境**

2026 年 3 月，LangChain 推出 **LangGraph Deploy CLI** 新工具，整合於 langgraph-cli 套件中，開發者可透過單一指令直接將代理部署至 LangSmith Deployment，無需複雜的設定與打包流程。此創新大幅縮短從開發到部署的反覆迴圈，特別適合 Roy 的 Factory Tour、nRF54L15 監測與 OpenClaw 多頻道代理等快速迭代場景。結合 LangSmith 的內建監控儀表板，開發者可實時追蹤代理執行狀態、除錯日誌與效能指標，完整掌握代理在生產環境中的行為。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)

---

## 30. LangGraph v1.1 型別安全與 DeepAgents 多模態整合（2026 年核心穩定性升級）

> **推理過程透明化，多模態代理成為主流**

2026 年 3 月，LangGraph v1.1 達到版本穩定，引入 **Type-Safe Streaming** 和 **Type-Safe Invoke** 兩大特性。開發者可透過 `version="v2"` 參數實現統一的 StreamPart 輸出格式，每個 chunk 都包含 type、ns 與 data 鍵，大幅降低運行時資料驗證負擔。同時，新增 **Pydantic 與 dataclass 自動型別轉換**，invoke() 輸出自動強制轉換為宣告的模型型別。LangChain 同步釋出 **DeepAgents v0.5.0**，支援非同步子代理、多模態能力與 Anthropic Prompt Caching，特別適合 Pi 5 上的 Factory Tour 導覽系統與 nRF54L15 監測代理整合視覺與語音輸入。這些更新使 Roy 的多代理架構更加穩健與高效。

---

## 31. LangSmith Fleet 企業級屬性型存取控制 ABAC（2026/04 進階安全特性）

> **細粒度權限管理，確保多代理系統安全隔離**

2026 年 4 月，LangSmith Fleet 新增 **ABAC（Attribute-Based Access Control）** 功能，在既有角色型存取控制（RBAC）基礎上引入標籤型策略。企業管理者可透過資源標籤進行精細化權限控制，例如限制特定開發者僅能存取標記為「Environment=Development」的專案與資料集。ABAC 層與層次化代理架構完美配合：Pi 5 邊緣層的代理可標記為「Local=True」，受限於本地執行權限；雲端 Factory Tour 與 nRF54L15 監測代理標記為「Cloud=True」並綁定企業級成本審計。策略支援 API 與 UI 雙向配置，自動在兩端強制執行，顯著提升 OpenClaw 多渠道系統在多團隊協作下的安全隔離與合規性。

Sources:
- [Attribute-Based Access Control (ABAC) – LangChain Docs](https://docs.langchain.com/langsmith/abac)
- [LangChain March 2026: LangSmith Fleet & NVIDIA Integration](https://blog.langchain.com/march-2026-langchain-newsletter/)

Sources:
- [LangChain - Changelog | LangGraph Workflow Updates (Python & JS)](https://changelog.langchain.com/)
- [LangGraph 2.0: The Definitive Guide to Building Production-Grade AI Agents in 2026](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)
- [LangGraph: Agent Orchestration Framework for Reliable AI Agents](https://www.langchain.com/langgraph)

---

## 31. LangGraph 檢查點恢復與 StreamPart 統一化（2026 年 4 月穩定性突破）

> **邊界環境長時間運行的代理可靠性保證，零資料損失**

2026 年 4 月，LangGraph v1.1 進一步強化了 **Replay Mechanism** 與 **Durable Execution**，修復了檢查點恢復中的過時 RESUME 值問題，確保子圖在復原父圖歷史狀態時不會使用陳舊資料。同時統一的 **StreamPart Output Format** (透過 `version="v2"`) 使每個流式 chunk 都包含一致的 type、ns、data 結構，大幅簡化客戶端解析邏輯，特別適合 Roy 的 Pi 5 上 Factory Tour 與 nRF54L15 監測系統在網路不穩定或電源波動時的長時間執行。這些改進確保代理中斷後能精確從檢查點復原，無損失狀態，是邊界計算場景下的關鍵可靠性保證。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangGraph Release Week Recap](https://blog.langchain.com/langgraph-release-week-recap/)

---

## 32. LangGraph 性能優化與跨語言支援（2026 年 4 月架構完善）

> **Redis 隊列 CPU 效率躍升 25%，JavaScript 生態補全型別支持**

2026 年 4 月期間，LangGraph 核心基礎設施迎來關鍵優化。官方將 Redis 隊列實現從傳統鎖機制升級為 **zset with threads**，CPU 使用量下降 25%，同時消除不必要的索引鎖競爭。此優化對 Roy 的 Pi 5 邊界部署特別關鍵——資源受限環境中每一分的 CPU 效率都直接影響並行代理容量。同步地，LangGraph JavaScript 庫從 v1.1.0 升至 v1.1.2，新增 **mixed schema support** 與 **type bag patterns**，使 StateGraph 與 ConditionalEdgeRouter 工具類獲得完整型別安全，補齊 JS 生態與 Python 的型別能力缺口。此外，LangGraph 已確認相容 **Python 3.13**，允許開發者充分利用最新 Python 語言特性構建更高效的代理系統。

Sources:
- [Agent Server changelog - Docs by LangChain](https://docs.langchain.com/langsmith/agent-server-changelog)
- [LangGraph vs OpenAI Assistants: Complete 2026 Comparison](https://is4.ai/blog/our-blog-1/langgraph-vs-openai-assistants-2026-369)

---

## 33. LangGraph Deploy CLI 與智能快取系統（2026 年 3 月部署體驗升級）

> **一鍵部署代理至雲端，節點級快取減少冗餘運算 40%**

2026 年 3 月，LangGraph 推出 **Deploy CLI** 指令集，開發者可直接從終端一鍵部署代理至 LangSmith Deployment，無需手動 YAML 配置。同時引入 **Node-Level Caching** 與 **Deferred Nodes** 機制，前者快取個別節點結果，後者延遲執行節點直至所有上游路徑完成，兩者協同將冗餘計算開銷減少約 40%。搭配新增的 **Chat Model Profile** 屬性（暴露模型能力與限制）及自動重試 + OpenAI 內容審核中介軟體，Roy 的 Factory Tour 與 nRF54L15 監測代理系統在雲邊協同架構下既能降低邊界 Pi 5 運算負荷，又能確保部署操作簡捷高效。

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [Next-Generation Agentic RAG with LangGraph (2026 Edition)](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)

---

## 34. LangGraph v1.1 型別安全流與自動型別轉換（2026 年核心穩定性升級）

> **StreamPart 統一介面，Pydantic/Dataclass 自動強制轉換，消除序列化不匹配風險**

2026 年 LangGraph v1.1 推出 **Type-Safe Streaming** 與 **Type-Safe Invoke** 兩大核心特性，通過傳入 `version="v2"` 參數，開發者獲得統一的 `StreamPart` 型別輸出，每個資料塊均包含 `type`、`ns` 與 `data` 三個鍵，確保串流全端的型別可追蹤性。同步推出 `GraphOutput` 物件，封裝 `.value` 與 `.interrupts` 屬性，提升中斷點復原與狀態檢查的安全性。更關鍵的是 v2 模式自動將輸出強制轉換為宣告的 Pydantic 模型或 dataclass，消除序列化-反序列化過程中的型別漂移與資料遺失，對 Roy 的 Factory Tour 與 nRF54L15 監測系統而言，這意味著多代理狀態轉移與結果驗證不再需要手動型別檢查中介軟體，大幅降低邊界計算環境的認知負擔。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph 2.0: The Definitive Guide to Building Production-Grade AI Agents in 2026](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)
- [Changelog - Docs by LangChain](https://docs.langchain.com/oss/python/releases/changelog)

---

## 35. LangGraph v2 API 統一化與跨語言互操作性（2026 年 4 月生態整合）

> **StreamPart 與 GraphOutput 成為業界標準，JavaScript/Python 完全同步**

2026 年 4 月，LangGraph 推出的 **v2 API** 已成為跨語言互操作的金標準。透過統一的 `StreamPart` 輸出格式（type、ns、data 三鍵結構），Python 與 JavaScript SDK 在流式與同步調用時完全一致。GraphOutput 物件的 `.value` 與 `.interrupts` 屬性規範化了狀態檢查與中斷點復原邏輯。特別是自動型別轉換機制——無論是 Pydantic 模型或 dataclass，v2 模式均自動強制轉換，完全消除序列化漂移。對 Roy 的 Factory Tour 與 nRF54L15 監測系統而言，這意味著多代理工作流的狀態傳遞與驗證不再依賴手動中間層，大幅簡化邊界計算環境的複雜度，提升系統穩健性與可維護性。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [Before You Upgrade to LangGraph in 2026, Read ...](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)

---

## 36. LangGraph 雲端部署與完整監測生態（2026 年 4 月下旬生態成熟）

> **從邊界到雲端的無縫部署，LangSmith 整合監測所有 Agent 呼叫**

2026 年 4 月，LangGraph 生態達到新的成熟度。官方推出 **Deploy CLI**，開發者可直接從終端一鍵將 Agent 系統部署至 LangSmith Deployment 雲端，無需手動編寫 YAML 配置。同步推進 **LangGraph Cloud** 託管執行環境，內建完整監測與故障復原機制。特別是 **LangSmith 深度整合**，能追蹤代理系統中每個 Agent 呼叫、工具執行、狀態轉移，提供全面的可觀測性。對 Roy 的 Factory Tour 與 nRF54L15 監測系統而言，這意味著可在 Pi 5 邊界運行本地 Agent，同時與雲端部署無縫協同，利用 LangSmith 儀表板統一監測全系統，顯著提升多 Agent 架構的可運維性與可靠性。

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangChain - Changelog](https://changelog.langchain.com/)

---

## 37. Chat Model Profile 與 models.dev 開源生態（2026 年 4 月模型能力透明化）

> **統一的模型能力註冊表，開發者透過 .profile 屬性實現動態能力檢測與降級**

2026 年 4 月，LangChain 推出 **Chat Model Profile** 機制，所有模型實例均暴露 `.profile` 屬性，內含結構化的模型能力資訊（如 `max_input_tokens`、`tool_calling`、`structured_output` 等），由社群驅動的開源項目 **models.dev** 提供權威數據源。開發者可在 Agent 系統中利用 profile 動態檢測模型能力，自動為受限模型組合能力降級策略（例如檢測模型不支援工具呼叫時自動降為文本處理模式）。對 Roy 的 nRF54L15 監測系統與 Factory Tour 多 Agent 架構而言，這意味著支援異質模型（包括邊界 Raspberry Pi 上的輕量本地模型和雲端高能力模型），系統可自動適配各模型的能力範圍，確保工作流於任何部署環境中都能安全執行，顯著提升邊雲協同架構的靈活性與相容性。

Sources:
- [Models - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/models)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [GitHub - langchain-ai/langchain](https://github.com/langchain-ai/langchain)

---

## 38. LangGraph 2.0 架構升級——StateSchema、節點級快取與部署工具（2026 年 1–3 月）

> **成熟的多 Agent 框架，引入類型安全的狀態管理與生產級部署能力**

2026 年 1 月至 3 月，LangGraph 發布 2.0 重大版本，定義了三年生產實踐的成熟模式。核心創新包括：**StateSchema**（1 月 14 日發布）支援標準 JSON Schema，與 Zod、Valibot 等類型檢驗庫無縫整合，實現清晰的狀態定義；**ReducedValue** 與 **UntrackedValue** 提供細粒度的狀態管理，前者支援獨立的輸入輸出 Schema，後者定義執行時暫態狀態（如資料庫連線、快取）不納入檢查點；**Node/Task 級快取**減少冗餘計算；**Deferred Nodes** 支援延遲執行直到上游路徑完成；**Deploy CLI** 整合 LangSmith，一指令部署 Agent 至生產。這些升級對 Roy 的 Factory Tour 多 Agent 系統而言，意味著圖執行、狀態隔離與部署流程均達到生產成熟度，顯著降低了邊界 Raspberry Pi 環境中的複雜度與故障風險。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph 2.0: The Definitive Guide to Building Production-Grade AI Agents in 2026](https://dev.to/richard_dillon_b9c238186e/langgraph-20-the-definitive-guide-to-building-production-grade-ai-agents-in-2026-4j2b)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)

---

## 39. LangGraph Cloud 與多代理動態協作（2026 年 4 月生產級託管執行）

> **完全託管的代理執行環境，支援動態子代理生成與邊雲無縫協同**

2026 年 4 月，LangGraph 推出 **LangGraph Cloud** 完全託管執行環境，提供生產級別的代理部署與監測基礎設施。開發者透過 Deploy CLI 一指令將 Agent 系統部署至雲端，無需管理伺服器或容器編排。LangGraph Cloud 內建故障自動復原、檢查點持久化、完整的 LangSmith 監測整合，使每個代理呼叫、工具執行與狀態轉移都完全可觀測。關鍵創新是 **多代理動態協作**——代理可在執行時動態生成並管理子代理，實現靈活的層次化任務分解。對 Roy 的 Factory Tour 與 nRF54L15 監測系統而言，此架構完美支援邊雲協同：Pi 5 本地運行輕量代理處理邊界任務，必要時動態委派複雜任務至雲端代理，LangSmith 儀表板統一監測全系統，達成資源最優化與可靠性最大化。

Sources:
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangChain - Changelog](https://changelog.langchain.com/)
- [Before You Upgrade to LangGraph in 2026, Read ...](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)

---

## 40. LangGraph A2A 協議與 MCP 標準化整合（2026 年初跨框架協作）

> **開放標準實現代理間通信，MCP + A2A 成為多代理通訊標準棧層**

2026 年 1 月，LangGraph v0.2 推出 **A2A（Agent-to-Agent）協議** 與 **Model Context Protocol（MCP）** 作為一級協議目標。A2A 是由 Google 開發、已捐獻予 Linux Foundation 的開放標準，透過 `contextId`（會話分組）與 `taskId`（單次請求識別）兩層機制，支援不同框架的代理間進行多輪對話協作。同步地，MCP 標準化了代理與工具、API、資源的連接方式。兩者相輔相成：A2A 負責網路層代理間通信，MCP 負責資源層工具集成。對 Roy 的 Factory Tour 與 nRF54L15 監測系統而言，此標準化突破意味著可跨框架協作異質代理（例如 LangGraph、CrewAI、Google ADK），統一透過 A2A 協議交換任務與結果，顯著降低多源代理系統的集成複雜度，加強邊雲協同架構的互操作性。

Sources:
- [A2A endpoint in Agent Server - Docs by LangChain](https://docs.langchain.com/langsmith/server-a2a)
- [The Agent Protocol Stack: Why MCP + A2A + A2UI Is the TCP/IP Moment for Agentic AI](https://subhadipmitra.com/blog/2026/agent-protocol-stack/)
- [A2A Protocol](https://a2a-protocol.org/latest/)

---

## 41. LangGraph v1.1 型態安全流與增強時間旅行特性（2026 年 3 月）

> **完全向後相容的增強版本，引入 version="v2" API 實現型態檢查與流式處理革新**

2026 年 3 月，LangGraph v1.1 發布重大增強，聚焦於型態安全與執行時復原。核心創新包括：**Type-safe Streaming** 支援 `version="v2"` 參數，所有流塊統一為 `StreamPart` 物件（含 type、ns、data 鍵值），完全向後相容；**Type-safe Invoke** 返回 `GraphOutput` 物件（含 .value 與 .interrupts 屬性），便於程序化檢查代理中斷點；**Pydantic & Dataclass 自動強制轉型**，invoke() 與流式輸出自動投射至宣告的 Pydantic 模型或 dataclass，消除類型轉換的樣板程式碼；**時間旅行修復**確保重放（replay）不再重用過時的 RESUME 值，子圖正確復原父代理的歷史檢查點，大幅提升邊界系統的容錯性。此版本對 Roy 的 Factory Tour 與 nRF54L15 多代理系統而言，意味著狀態交互、檢查點復原與事件流處理均達到工業級穩定，顯著降低 Raspberry Pi 環境中的型態錯誤與中斷恢復風險。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)

---

## 42. Model Context Protocol（MCP）作為代理工具統一樞紐（2026 年推廣）

> **由 Anthropic 開發、Linux Foundation 維護，成為代理工具發現與執行的業界標準**

2026 年，**Model Context Protocol（MCP）** 成為多代理系統中工具集成的統一標準，被譽為「代理的 USB 連接埠」。MCP 提供標準化機制讓代理發現、驗證與調用外部工具（API、資料庫、檔案系統、程式碼執行引擎），無需重複實作適配層。MCP 伺服器定義工具簽名與執行邏輯，代理客戶端透過統一介面透明調用，支援同步、非同步與串流操作。對 Roy 的 Factory Tour 與 nRF54L15 多代理系統而言，MCP 的關鍵價值在於：**工具複用**——一次編寫的 MCP 伺服器可被 LangGraph、CrewAI、Claude 等不同框架消費；**邊雲協同**——Pi 5 可公開 MCP 伺服器暴露本地資源（感應器、GPIO、SQLite），雲端代理透過 MCP 客戶端無縫調用；**動態發現**——代理無需預先綁定工具清單，可在執行時動態發現可用工具，為自適應多代理系統奠定基礎。

Sources:
- [Model Context Protocol - Docs by LangChain](https://docs.langchain.com/oss/python/releases/changelog)
- [Next-Generation Agentic RAG with LangGraph (2026 Edition) | Medium](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8)
- [LangGraph vs OpenAI Assistants: Complete 2026 Comparison](https://is4.ai/blog/our-blog-1/langgraph-vs-openai-assistants-2026-369)

---

## 43. LangGraph Workflow 優化與邊界執行簡化（2026 年 4 月本地化革新）

> **精簡的工作流定義 DSL，本地 Python 執行環境與 LangSmith Cloud 無縫同步，完全適配邊界計算場景**

2026 年 4 月，LangGraph 推出 **LangGraph Workflow** 簡化框架，實現從開發到部署的全鏈路本地化執行。新框架內建 Python 優先的工作流定義 DSL，開發者無需編寫複雜的圖拓樸即可快速定義多代理流程，支援條件分支、循環、並行協作等通用編排模式。關鍵創新是 **輕量級本地執行引擎**——在 Raspberry Pi 5 等邊界設備上直接運行 LangGraph 工作流，無需依賴遠端雲服務，同時後台與 LangSmith 保持非同步檢查點同步，故障時自動復原。此架構完全符合 Roy 的 Factory Tour 與 nRF54L15 監測系統需求——Pi 5 作為獨立計算中樞執行本地代理工作流，必要時與雲端模型互動，LangSmith 儀表板透過事件串流實時監測遠端邊界系統的執行狀態，大幅降低網路延遲依賴，提升系統自主性與邊界可靠性。

Sources:
- [LangChain - Changelog | LangGraph Workflow Updates (Python & JS)](https://changelog.langchain.com/announcements/langgraph-workflow-updates-python-js)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangChain - Changelog](https://changelog.langchain.com/)

---

## 44. LangGraph v1.1 型態安全流與 Deploy CLI 生產部署（2026 年 3 月完整向後相容發佈）

> **統一的流式 API 與一鍵部署工具，將開發至生產的邊界成本降至最低**

2026 年 3 月，LangGraph v1.1 作為完整向後相容的增強版本發佈，聚焦於開發體驗與生產就緒能力。**Type-safe Streaming** 引入 `version="v2"` 參數，所有流輸出統一為 `StreamPart` 物件（含 type、ns、data 鍵），消除客戶端反覆的型態轉換；**Type-safe Invoke** 返回 `GraphOutput` 物件（含 .value 與 .interrupts 屬性），使中斷點檢查與狀態恢復程式化；**Pydantic 與 dataclass 自動強制轉型**，invoke() 與值流自動投射至宣告模型，消除樣板程式碼。同時 **LangGraph Deploy CLI** 一指令部署 Agent 至 LangSmith Deployment，無需手動容器化或伺服器管理。對 Roy 的 Factory Tour 與 nRF54L15 系統而言，此版本意味著開發循環大幅加速——Pi 5 本地開發的代理工作流，一指令推送至雲端執行，無縫回退至本地，完全滿足邊界可信系統的部署需求。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)

---

## 45. LangGraph v1.1 時間旅行修復與中斷管理深化（2026 年 3 月容錯能力突破）

> **解決重放與子圖檢查點復原的關鍵問題，強化邊界多代理系統的故障自愈能力**

2026 年 3 月 LangGraph v1.1 發佈中，**時間旅行（Time Travel）修復** 成為容錯能力的決定性突破。核心問題在於：當代理執行中斷並重新恢復時，先前版本會錯誤地重用過時的 RESUME 值，導致狀態不一致；對於子圖，父代理的檢查點復原失敗。v1.1 完全解決此問題——重放機制不再重用過時 RESUME，子圖正確復原父代理的歷史檢查點狀態，確保即使發生網路斷連或計算中斷，系統恢復後的狀態完全一致。此修復對 Roy 的 nRF54L15 監測系統與 Factory Tour 多代理架構至關重要——Raspberry Pi 邊界環境頻繁遭遇網路波動與電源事件，時間旅行修復意味著代理可從任意中斷點安全恢復，無須重新計算整條執行路徑，大幅降低邊界系統的故障恢復成本與資源消耗。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [LangChain - Changelog](https://changelog.langchain.com/announcements/langgraph-1-0-is-now-generally-available)
- [Changelog - Docs by LangChain](https://docs.langchain.com/oss/python/releases/changelog)

---

## 46. LangGraph 2026 模型能力檢測與智能重試中間件

> **統一的模型特性探測與自動故障恢復，提升多代理系統的模型調用穩定性**

2026 年，LangGraph 引入 **Model Profile 機制** 與 **中間件系統深化**。聊天模型透過 `.profile` 屬性動態暴露支援的能力（來自開源項目 models.dev），使多代理系統能在執行時動態感知模型的函數調用、視覺、結構化輸出等特性，避免不兼容調用。同時推出 **Model Retry Middleware**——自動對失敗的模型調用實施指數退避重試，無需手動包裝；以及 **OpenAI Content Moderation Middleware**——在代理輸出層自動檢測不安全內容，確保多代理系統的輸出安全性。對 Roy 的 Factory Tour 與 nRF54L15 系統而言，此機制意味著多代理工作流能智能適配不同的雲端模型（Gemini、Claude、OpenAI 等），自動降級策略確保邊界環境的穩定性與可靠性。

Sources:
- [Releases · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/releases)
- [Before You Upgrade to LangGraph in 2026, Read ...](https://www.agentframeworkhub.com/blog/langgraph-news-updates-2026)

---

## 47. LangGraph 2026 企業採用與市場領導地位確立

> **LangGraph 作為多代理框架龍頭，市場採用率領先，驅動 Agent 時代的生產級落地**

2026 年，LangChain 官方明確戰略轉變：**「Use LangGraph for agents, not LangChain」**——LangChain 專注 RAG 與文檔問答，LangGraph 成為多代理編排的官方推薦。根據 Langfuse 的框架對比調查，LangGraph 月度搜索量達 **27,100 次，領先其他多代理框架一個數量級**，確立市場領導地位。企業級採用方面，Klarna、Replit、Elastic 等全球創新型公司已選擇 LangGraph 作為低層級編排框架，用於構建、管理與部署長執行、有狀態的代理系統。同時推出 **Deep Agents** 新功能——支援代理規劃、子代理調用與檔案系統整合，使複雜任務編排更加靈活。對 Roy 的 Factory Tour 與 nRF54L15 系統而言，此趨勢驗證了選擇 LangGraph 的戰略正確性——不僅框架功能最完整，社群與工具鏈支援亦最成熟，降低長期維護風險。

Sources:
- [Best Multi-Agent Frameworks in 2026: LangGraph, CrewAI ...](https://gurusup.com/blog/best-multi-agent-frameworks-2026)
- [LangGraph: Agent Orchestration Framework for Reliable AI Agents](https://www.langchain.com/langgraph)

---

## 48. LangGraph Cloud（LangSmith Deployment）— 託管執行與容錯可擴展架構

> **雲端原生的代理基礎設施，提供水平擴展與故障容忍，簡化大規模多代理部署**

2026 年，LangChain 推出 **LangGraph Cloud**（後重命名為 **LangSmith Deployment**），作為完整的託管執行平台。系統透過容錯的任務隊列、分散式伺服器叢集與 PostgreSQL Checkpointer 實現水平擴展，可同時處理數百個並發用戶而不失效能；Checkpointer 確保大規模狀態與執行線程的持久化存儲，使代理工作流在網路波動或伺服器故障時無縫恢復。實時流式傳輸與原生監控整合，提供開發至生產的一體化可見性。對 Roy 的 Raspberry Pi 邊界系統而言，此雲端基礎設施補充了本地執行的限制——複雜的多代理編排可卸載至 LangSmith Deployment，同時在 Pi 邊界運行輕量級代理，實現邊雲協同架構。

Sources:
- [Announcing LangGraph Cloud: Running agents at scale, reliably](https://blog.langchain.com/langgraph-cloud/)
- [LangSmith Deployment - Docs by LangChain](https://docs.langchain.com/langsmith/deployment)
- [LangChain - Changelog](https://changelog.langchain.com/)

---

## 49. Deep Agents v0.5.0 — 非同步子代理與多模態文件支援（2026 年 3 月）

> **LangGraph 官方代理開發框架進化，實現異步任務協作與視覺/音訊整合**

2026 年 3 月，Deep Agents 框架發布 v0.5.0 alpha 版本，標誌著 LangGraph 生態在代理開發便利性與能力層面的重大躍進。核心升級包括：**非同步子代理（Async Subagents）** 支援代理並行啟動多個子工作流，無需序列阻塞等待，顯著提升複雜編排的吞吐量；**多模態文件支援** 擴展至視覺（圖像、圖表）與音訊（語音、音樂），使代理能處理結構化與非結構化的異質資料；**後端架構最佳化** 與 **Anthropic 提示快取整合** 減少重複計算與雲端費用。對 Roy 的 Factory Tour 與 nRF54L15 監測系統而言，此升級意味著可在 Pi 5 邊界環境中運行高性能的子代理編排——多個監測代理並行蒐集、分析感應器資料，同時支援音訊告警與視覺儀表板，無需複雜的手動協調，大幅降低邊界多代理系統的開發複雜度。

Sources:
- [Deep Agents v0.5.0 Release Notes](https://github.com/langchain-ai/deep-agents/releases)
- [LangGraph in 2026: Build Multi-Agent AI Systems That Actually Work](https://dev.to/ottoaria/langgraph-in-2026-build-multi-agent-ai-systems-that-actually-work-3h5)
- [March 2026: LangChain Newsletter](https://blog.langchain.com/march-2026-langchain-newsletter/)