"""
rag_engine.py - RAG 引擎（ChromaDB 向量檢索）
整合東海大學 RAG 專案的概念，用於工廠導覽知識檢索
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"

# 全域 RAG 引擎實例
_rag_instance: Optional["RAGEngine"] = None


class RAGEngine:
    """基於 ChromaDB 的 RAG 檢索引擎"""

    def __init__(self):
        self.collection = None
        self.ready = False
        self._init_chromadb()

    def _init_chromadb(self):
        """初始化 ChromaDB（使用內建 embedding）"""
        try:
            import chromadb

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.collection = self.client.get_or_create_collection(
                name="factory_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            # 載入知識庫
            self._load_knowledge()
            self.ready = True
            logger.info(
                f"RAG 引擎初始化完成，共 {self.collection.count()} 筆文件"
            )
        except ImportError:
            logger.warning("chromadb 未安裝，RAG 功能停用。執行: pip install chromadb")
            self.ready = False
        except Exception as e:
            logger.warning(f"RAG 引擎初始化失敗: {e}")
            self.ready = False

    def _load_knowledge(self):
        """載入所有知識到向量資料庫"""
        if not self.collection:
            return

        # 如果已有資料，跳過載入
        if self.collection.count() > 0:
            logger.info(f"ChromaDB 已有 {self.collection.count()} 筆資料，跳過載入")
            return

        documents = []
        metadatas = []
        ids = []

        # 載入廠區資料
        areas_file = KNOWLEDGE_DIR / "areas.json"
        if areas_file.exists():
            with open(areas_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for area in data.get("areas", []):
                # 主要描述
                doc = f"區域名稱：{area['name']}\n描述：{area['description']}"
                if area.get("technical_specs"):
                    specs = area["technical_specs"]
                    doc += "\n技術規格："
                    for k, v in specs.items():
                        if isinstance(v, list):
                            doc += f"\n  {k}: {', '.join(str(x) for x in v)}"
                        else:
                            doc += f"\n  {k}: {v}"
                documents.append(doc)
                metadatas.append(
                    {"type": "area", "area_id": area["id"], "area_name": area["name"]}
                )
                ids.append(f"area_{area['id']}")

                # 安全規範（獨立文件）
                if area.get("safety_notes"):
                    safety_doc = f"{area['name']}的安全規範：\n" + "\n".join(
                        f"- {note}" for note in area["safety_notes"]
                    )
                    documents.append(safety_doc)
                    metadatas.append(
                        {
                            "type": "safety",
                            "area_id": area["id"],
                            "area_name": area["name"],
                        }
                    )
                    ids.append(f"safety_{area['id']}")

            # 路線資料
            for i, route in enumerate(data.get("routes", [])):
                doc = (
                    f"導覽路線：{route['name']}\n"
                    f"說明：{route['description']}\n"
                    f"時間：{route['duration']}\n"
                    f"停靠站：{' → '.join(route['stops'])}"
                )
                documents.append(doc)
                metadatas.append({"type": "route", "route_name": route["name"]})
                ids.append(f"route_{i}")

            # 緊急資訊
            emergency = data.get("emergency", {})
            if emergency:
                doc = (
                    f"緊急應變資訊：\n"
                    f"緊急出口：{emergency.get('exit_locations', '')}\n"
                    f"集合點：{emergency.get('assembly_point', '')}\n"
                    f"緊急聯絡：{emergency.get('emergency_contact', '')}\n"
                    f"AED位置：{emergency.get('aed_location', '')}\n"
                    f"急救箱：{emergency.get('first_aid', '')}"
                )
                documents.append(doc)
                metadatas.append({"type": "emergency"})
                ids.append("emergency_info")

        # 載入 FAQ
        faq_file = KNOWLEDGE_DIR / "faq.json"
        if faq_file.exists():
            with open(faq_file, "r", encoding="utf-8") as f:
                faq_data = json.load(f)

            for item in faq_data.get("faq", []):
                doc = f"問題：{item['question']}\n回答：{item['answer']}"
                documents.append(doc)
                metadatas.append(
                    {
                        "type": "faq",
                        "category": item.get("category", "general"),
                        "faq_id": item["id"],
                    }
                )
                ids.append(f"faq_{item['id']}")

        # 批量寫入
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"已載入 {len(documents)} 筆知識文件到 ChromaDB")

    def search(
        self,
        query: str,
        n_results: int = 3,
        doc_type: str | None = None,
    ) -> list[dict]:
        """搜尋相關知識"""
        if not self.ready or not self.collection:
            return []

        where = {"type": doc_type} if doc_type else None
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error(f"RAG 搜尋錯誤: {e}")
            return []

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                item = {
                    "content": doc,
                    "metadata": (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    ),
                    "distance": (
                        results["distances"][0][i] if results.get("distances") else None
                    ),
                }
                items.append(item)
        return items

    def add_document(self, doc_id: str, content: str, metadata: dict | None = None):
        """動態新增文件"""
        if not self.ready or not self.collection:
            return False
        try:
            self.collection.add(
                documents=[content],
                metadatas=[metadata or {}],
                ids=[doc_id],
            )
            return True
        except Exception as e:
            logger.error(f"新增文件失敗: {e}")
            return False

    def get_stats(self) -> dict:
        """取得 RAG 引擎統計"""
        if not self.ready:
            return {"ready": False, "total_documents": 0}
        return {
            "ready": True,
            "total_documents": self.collection.count() if self.collection else 0,
        }


def get_rag_engine() -> RAGEngine:
    """取得全域 RAG 引擎實例（Singleton）"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance


def rag_search(query: str, n_results: int = 3, doc_type: str | None = None) -> str:
    """供 LangChain tool 使用的搜尋函數"""
    engine = get_rag_engine()
    if not engine.ready:
        return "RAG 引擎未就緒，無法進行知識檢索。"
    results = engine.search(query, n_results, doc_type)
    if not results:
        return f"找不到與「{query}」相關的資訊。"

    output = f"找到 {len(results)} 筆相關資料：\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- 結果 {i} ---\n{r['content']}\n\n"
    return output
