"""
test_rag.py - 測試 RAG 引擎（ChromaDB 可選）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestRAGEngine:
    """RAG 引擎測試（需要 chromadb）"""

    @pytest.fixture(autouse=True)
    def check_chromadb(self):
        try:
            import chromadb  # noqa: F401

            self.has_chromadb = True
        except ImportError:
            self.has_chromadb = False

    def test_rag_engine_init(self):
        if not self.has_chromadb:
            pytest.skip("chromadb not installed")
        from rag_engine import RAGEngine

        engine = RAGEngine()
        assert engine.ready is True

    def test_rag_search(self):
        if not self.has_chromadb:
            pytest.skip("chromadb not installed")
        from rag_engine import RAGEngine

        engine = RAGEngine()
        results = engine.search("組裝線的設備有哪些")
        assert len(results) > 0
        # 應該找到組裝線相關的資料
        found = any("SMT" in r["content"] or "組裝" in r["content"] for r in results)
        assert found

    def test_rag_search_faq(self):
        if not self.has_chromadb:
            pytest.skip("chromadb not installed")
        from rag_engine import RAGEngine

        engine = RAGEngine()
        results = engine.search("停車場在哪裡", doc_type="faq")
        assert len(results) > 0

    def test_rag_search_safety(self):
        if not self.has_chromadb:
            pytest.skip("chromadb not installed")
        from rag_engine import RAGEngine

        engine = RAGEngine()
        results = engine.search("安全規範", doc_type="safety")
        assert len(results) > 0

    def test_rag_stats(self):
        if not self.has_chromadb:
            pytest.skip("chromadb not installed")
        from rag_engine import RAGEngine

        engine = RAGEngine()
        stats = engine.get_stats()
        assert stats["ready"] is True
        assert stats["total_documents"] > 0

    def test_rag_tool_function(self):
        """測試 LangChain tool wrapper"""
        if not self.has_chromadb:
            pytest.skip("chromadb not installed")
        from rag_engine import rag_search

        result = rag_search("品管室檢測流程")
        assert "品管" in result or "檢測" in result or "找到" in result

    def test_rag_graceful_without_chromadb(self):
        """測試無 chromadb 時的優雅降級"""
        from factory_tour_agent import rag_knowledge_search

        # 即使 chromadb 未安裝，tool 也不應 crash
        result = rag_knowledge_search.invoke({"query": "test"})
        assert isinstance(result, str)
