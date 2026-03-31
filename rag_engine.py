"""
rag_engine.py - RAG 引擎（ChromaDB 向量檢索）v2.0
整合東海大學 RAG 專案 (my_rag_pro) 的核心能力

v2.0 新增：
  - SentenceTransformer 嵌入模型 (all-MiniLM-L6-v2)
  - 自訂 Markdown 文件匯入（段落分段）
  - 雙集合架構：factory_knowledge + custom_knowledge
  - 文件管理 API（新增、列表、刪除）

作者：Roy (YORROY123)
建立：2026-03-30
更新：2026-03-31 (RAG 整合 — 方案A)
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_DIR = Path(__file__).parent / "data" / "chroma_db"
DOCUMENTS_DIR = Path(__file__).parent / "documents"

# 全域 RAG 引擎實例（使用 Lock 保護）
_rag_instance: Optional["RAGEngine"] = None
_rag_lock = threading.Lock()


class RAGEngine:
    """基於 ChromaDB + SentenceTransformer 的 RAG 檢索引擎

    雙集合架構：
    - factory_knowledge: 工廠知識（從 areas.json / faq.json 自動載入）
    - custom_knowledge: 自訂文件（Markdown 匯入，東海大學 RAG 風格）
    """

    def __init__(self):
        self.factory_collection = None
        self.custom_collection = None
        self.embed_model = None
        self.ready = False
        self._init_embedding()
        self._init_chromadb()

    def _init_embedding(self):
        """初始化 SentenceTransformer 嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer

            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer (all-MiniLM-L6-v2) 載入完成")
        except ImportError:
            logger.warning(
                "sentence-transformers 未安裝，使用 ChromaDB 內建嵌入。"
                "安裝方式: pip install sentence-transformers"
            )
            self.embed_model = None
        except Exception as e:
            logger.warning(f"SentenceTransformer 載入失敗: {e}，使用內建嵌入")
            self.embed_model = None

    def _init_chromadb(self):
        """初始化 ChromaDB（雙集合）"""
        try:
            import chromadb

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

            # 集合 1: 工廠知識
            self.factory_collection = self.client.get_or_create_collection(
                name="factory_knowledge",
                metadata={"hnsw:space": "cosine"},
            )

            # 集合 2: 自訂文件知識
            self.custom_collection = self.client.get_or_create_collection(
                name="custom_knowledge",
                metadata={"hnsw:space": "cosine"},
            )

            # 載入工廠知識庫
            self._load_factory_knowledge()

            # 載入 documents/ 目錄中的自訂文件
            self._load_documents_dir()

            self.ready = True
            factory_count = self.factory_collection.count()
            custom_count = self.custom_collection.count()
            logger.info(
                f"RAG 引擎初始化完成 — "
                f"工廠知識: {factory_count} 筆, 自訂文件: {custom_count} 筆"
            )
        except ImportError:
            logger.warning("chromadb 未安裝，RAG 功能停用。執行: pip install chromadb")
            self.ready = False
        except Exception as e:
            logger.warning(f"RAG 引擎初始化失敗: {e}")
            self.ready = False

    # ─── 工廠知識載入 ───

    def _load_factory_knowledge(self):
        """載入工廠知識到向量資料庫"""
        if not self.factory_collection:
            return

        # 如果已有資料，跳過載入
        if self.factory_collection.count() > 0:
            logger.info(
                f"工廠知識已有 {self.factory_collection.count()} 筆，跳過載入"
            )
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

        # 批量寫入（使用 SentenceTransformer 嵌入）
        if documents:
            self._add_to_collection(
                self.factory_collection, documents, metadatas, ids
            )
            logger.info(f"已載入 {len(documents)} 筆工廠知識")

    # ─── 自訂文件載入 ───

    def _load_documents_dir(self):
        """載入 documents/ 目錄中的 Markdown 文件"""
        if not self.custom_collection:
            return

        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        for md_file in DOCUMENTS_DIR.glob("*.md"):
            # 檢查是否已載入（用檔名前綴查詢）
            file_prefix = f"doc_{md_file.stem}_"
            existing = self.custom_collection.get(
                where={"source_file": md_file.name}
            )
            if existing and existing["ids"]:
                logger.info(f"文件 {md_file.name} 已載入，跳過")
                continue

            self._ingest_markdown(md_file)

    def _chunk_markdown(self, content: str, min_length: int = 5) -> list[str]:
        """將 Markdown 內容分段（段落式分段，源自東海大學 RAG 專案）

        Args:
            content: Markdown 文件全文
            min_length: 最小段落字數，過短的段落會被過濾
        """
        chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > min_length]
        return chunks

    def _ingest_markdown(self, file_path: Path) -> int:
        """匯入單一 Markdown 文件到自訂知識集合

        Args:
            file_path: Markdown 文件路徑

        Returns:
            匯入的段落數量
        """
        if not self.custom_collection:
            return 0

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"讀取文件失敗 {file_path}: {e}")
            return 0

        chunks = self._chunk_markdown(content)
        if not chunks:
            logger.warning(f"文件 {file_path.name} 沒有有效段落")
            return 0

        ids = [f"doc_{file_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "type": "custom_document",
                "source_file": file_path.name,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
            }
            for i in range(len(chunks))
        ]

        self._add_to_collection(self.custom_collection, chunks, metadatas, ids)
        logger.info(f"已匯入文件 {file_path.name} ({len(chunks)} 段落)")
        return len(chunks)

    # ─── 嵌入 & 寫入 ───

    def _encode(self, texts: list[str]) -> list[list[float]] | None:
        """使用 SentenceTransformer 編碼文本（若可用）"""
        if self.embed_model is None:
            return None
        try:
            return self.embed_model.encode(texts).tolist()
        except Exception as e:
            logger.warning(f"嵌入編碼失敗: {e}")
            return None

    def _add_to_collection(
        self, collection, documents: list[str], metadatas: list[dict], ids: list[str]
    ):
        """寫入文件到指定集合（優先使用 SentenceTransformer 嵌入）"""
        embeddings = self._encode(documents)
        kwargs = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        collection.add(**kwargs)

    # ─── 搜尋 ───

    def search(
        self,
        query: str,
        n_results: int = 3,
        doc_type: str | None = None,
        collection_name: str | None = None,
    ) -> list[dict]:
        """搜尋相關知識

        Args:
            query: 搜尋查詢
            n_results: 回傳結果數
            doc_type: 按 type metadata 過濾
            collection_name: 指定集合 ("factory" / "custom" / None=兩者)
        """
        if not self.ready:
            return []

        collections = []
        if collection_name == "factory":
            collections = [self.factory_collection]
        elif collection_name == "custom":
            collections = [self.custom_collection]
        else:
            # 兩個集合都搜尋
            collections = [self.factory_collection, self.custom_collection]

        all_items = []
        for col in collections:
            if not col:
                continue
            if col.count() == 0:
                continue
            items = self._search_collection(col, query, n_results, doc_type)
            all_items.extend(items)

        # 按距離排序，取前 n_results
        all_items.sort(key=lambda x: x.get("distance", 999))
        return all_items[:n_results]

    def _search_collection(
        self, collection, query: str, n_results: int, doc_type: str | None
    ) -> list[dict]:
        """在單一集合中搜尋"""
        where = {"type": doc_type} if doc_type else None

        try:
            # 優先使用 SentenceTransformer 嵌入查詢
            query_embedding = self._encode([query])
            if query_embedding is not None:
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(n_results, collection.count()),
                    where=where,
                )
            else:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(n_results, collection.count()),
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
                        results["distances"][0][i]
                        if results.get("distances")
                        else None
                    ),
                }
                items.append(item)
        return items

    # ─── 文件管理 ───

    def add_document_from_text(
        self,
        doc_id: str,
        content: str,
        metadata: dict | None = None,
        use_chunking: bool = True,
    ) -> int:
        """從文字內容新增文件到自訂知識集合

        Args:
            doc_id: 文件識別碼
            content: 文件內容
            metadata: 額外 metadata
            use_chunking: 是否使用段落分段

        Returns:
            新增的段落/文件數量
        """
        if not self.ready or not self.custom_collection:
            return 0

        base_metadata = {"type": "custom_document"}
        if metadata:
            base_metadata.update(metadata)

        try:
            if use_chunking:
                chunks = self._chunk_markdown(content)
                if not chunks:
                    return 0
                ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
                metadatas = [
                    {**base_metadata, "chunk_index": str(i), "total_chunks": str(len(chunks))}
                    for i in range(len(chunks))
                ]
                self._add_to_collection(
                    self.custom_collection, chunks, metadatas, ids
                )
                return len(chunks)
            else:
                self._add_to_collection(
                    self.custom_collection,
                    [content],
                    [base_metadata],
                    [doc_id],
                )
                return 1
        except Exception as e:
            logger.error(f"新增文件失敗: {e}")
            return 0

    def add_markdown_file(self, file_path: str) -> int:
        """匯入 Markdown 文件

        Args:
            file_path: Markdown 檔案路徑

        Returns:
            匯入段落數
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"檔案不存在: {file_path}")
            return 0
        return self._ingest_markdown(path)

    def remove_document(self, source_file: str) -> bool:
        """移除指定來源檔案的所有段落

        Args:
            source_file: 來源檔案名稱
        """
        if not self.ready or not self.custom_collection:
            return False
        try:
            existing = self.custom_collection.get(
                where={"source_file": source_file}
            )
            if existing and existing["ids"]:
                self.custom_collection.delete(ids=existing["ids"])
                logger.info(f"已移除文件 {source_file} ({len(existing['ids'])} 段落)")
                return True
            return False
        except Exception as e:
            logger.error(f"移除文件失敗: {e}")
            return False

    def list_custom_documents(self) -> list[dict]:
        """列出所有自訂文件"""
        if not self.ready or not self.custom_collection:
            return []

        try:
            all_docs = self.custom_collection.get()
            if not all_docs or not all_docs["ids"]:
                return []

            # 按 source_file 彙總
            file_map: dict[str, dict] = {}
            for i, meta in enumerate(all_docs["metadatas"]):
                source = meta.get("source_file", meta.get("type", "unknown"))
                if source not in file_map:
                    file_map[source] = {
                        "source_file": source,
                        "chunks": 0,
                        "type": meta.get("type", "custom_document"),
                    }
                file_map[source]["chunks"] += 1

            return list(file_map.values())
        except Exception as e:
            logger.error(f"列出文件失敗: {e}")
            return []

    # ─── 統計 ───

    def get_stats(self) -> dict:
        """取得 RAG 引擎統計"""
        if not self.ready:
            return {"ready": False, "factory_documents": 0, "custom_documents": 0}
        return {
            "ready": True,
            "embedding_model": (
                "all-MiniLM-L6-v2" if self.embed_model else "chromadb-default"
            ),
            "factory_documents": (
                self.factory_collection.count() if self.factory_collection else 0
            ),
            "custom_documents": (
                self.custom_collection.count() if self.custom_collection else 0
            ),
            "total_documents": (
                (self.factory_collection.count() if self.factory_collection else 0)
                + (self.custom_collection.count() if self.custom_collection else 0)
            ),
            "custom_files": self.list_custom_documents(),
        }


def get_rag_engine() -> RAGEngine:
    """取得全域 RAG 引擎實例（執行緒安全 Singleton）"""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                _rag_instance = RAGEngine()
    return _rag_instance


def rag_search(query: str, n_results: int = 3, doc_type: str | None = None) -> str:
    """供 LangChain tool 使用的搜尋函數（搜尋工廠知識）"""
    engine = get_rag_engine()
    if not engine.ready:
        return "RAG 引擎未就緒，無法進行知識檢索。"
    results = engine.search(query, n_results, doc_type, collection_name="factory")
    if not results:
        return f"找不到與「{query}」相關的資訊。"

    output = f"找到 {len(results)} 筆相關資料：\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- 結果 {i} ---\n{r['content']}\n\n"
    return output


def rag_search_custom(query: str, n_results: int = 3) -> str:
    """供 LangChain tool 使用：搜尋自訂文件知識"""
    engine = get_rag_engine()
    if not engine.ready:
        return "RAG 引擎未就緒。"
    results = engine.search(query, n_results, collection_name="custom")
    if not results:
        return f"自訂知識庫中找不到與「{query}」相關的資訊。"

    output = f"在自訂知識庫中找到 {len(results)} 筆相關資料：\n\n"
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source_file", "未知來源")
        output += f"--- 結果 {i} (來源: {source}) ---\n{r['content']}\n\n"
    return output


def rag_search_all(query: str, n_results: int = 5) -> str:
    """供 LangChain tool 使用：同時搜尋工廠知識 + 自訂文件"""
    engine = get_rag_engine()
    if not engine.ready:
        return "RAG 引擎未就緒。"
    results = engine.search(query, n_results)
    if not results:
        return f"找不到與「{query}」相關的資訊。"

    output = f"在全部知識庫中找到 {len(results)} 筆相關資料：\n\n"
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source_file", r["metadata"].get("type", "工廠知識"))
        output += f"--- 結果 {i} (來源: {source}) ---\n{r['content']}\n\n"
    return output
