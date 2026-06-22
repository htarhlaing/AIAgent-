from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AI_agent.rag.hybrid_retriever import HybridRetriever, LocalBM25Index
from AI_agent.rag.index_manifest import IndexManifest
from AI_agent.rag.rag_service import RagSummaryService
from AI_agent.rag.vector_store import VectorStoreService
from AI_agent.services.context_service import UserContext
from AI_agent.services.records_service import RecordsService
from AI_agent.utils.config_handler import chroma_config
from AI_agent.workflows.report_workflow import ReportWorkflow


class FakeCollection:
    def count(self) -> int:
        return 1


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        return self.results[:k]


class FakeVectorStoreService:
    def __init__(self, results):
        self.vector_store = FakeVectorStore(results)
        self._collection = FakeCollection()

    def count(self) -> int:
        return 1


class FakeRag:
    def rag_summary(self, query: str) -> str:
        return "定期清理主刷并检查滤网。[来源1]\n\n参考来源：\n- [来源1] 维护保养.txt"


class FakeManagedCollection:
    def __init__(self, owner):
        self.owner = owner

    def count(self) -> int:
        return len(self.owner.documents)


class FakeManagedVectorStore:
    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.deleted_ids: list[str] = []
        self._collection = FakeManagedCollection(self)

    def add_documents(self, documents, ids):
        self.documents.update(dict(zip(ids, documents)))

    def delete(self, ids):
        self.deleted_ids.extend(ids)
        for item_id in ids:
            self.documents.pop(item_id, None)

    def get(self, include):
        return {"ids": list(self.documents)}


class QualityArchitectureTests(unittest.TestCase):
    def test_manifest_supports_upsert_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = IndexManifest(str(Path(directory) / "manifest.sqlite3"))
            manifest.upsert("a.txt", "hash-1", ["chunk-1"])
            self.assertEqual(manifest.all()["a.txt"].content_hash, "hash-1")

            manifest.upsert("a.txt", "hash-2", ["chunk-2", "chunk-3"])
            self.assertEqual(manifest.all()["a.txt"].chunk_ids, ["chunk-2", "chunk-3"])

            manifest.remove("a.txt")
            self.assertEqual(manifest.all(), {})

    def test_vector_sync_handles_add_update_and_source_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            data_dir.mkdir()
            source = data_dir / "guide.txt"
            source.write_text("第一版维护说明", encoding="utf-8")

            old_data_path = chroma_config["data_path"]
            old_types = chroma_config["allow_knowledge_file_type"]
            chroma_config["data_path"] = str(data_dir)
            chroma_config["allow_knowledge_file_type"] = ["txt"]
            try:
                service = VectorStoreService.__new__(VectorStoreService)
                service.vector_store = FakeManagedVectorStore()
                service.spliter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
                service.manifest = IndexManifest(str(Path(directory) / "manifest.sqlite3"))

                added = service.sync_documents()
                self.assertEqual(added.added, 1)
                first_ids = service.manifest.all()[os.path.abspath(source)].chunk_ids

                source.write_text("第二版维护说明，增加滤网清洁要求", encoding="utf-8")
                updated = service.sync_documents()
                self.assertEqual(updated.updated, 1)
                self.assertTrue(set(first_ids).issubset(service.vector_store.deleted_ids))

                source.unlink()
                deleted = service.sync_documents()
                self.assertEqual(deleted.deleted, 1)
                self.assertEqual(service.manifest.all(), {})
            finally:
                chroma_config["data_path"] = old_data_path
                chroma_config["allow_knowledge_file_type"] = old_types

    def test_hybrid_retrieval_reranks_and_filters(self):
        relevant = Document(
            page_content="扫地机器人主刷需要定期清理毛发并检查磨损",
            metadata={"source": "维护保养.txt"},
        )
        irrelevant = Document(
            page_content="选择充电底座的摆放位置",
            metadata={"source": "选购指南.txt"},
        )
        service = FakeVectorStoreService([(irrelevant, 0.2), (relevant, 0.7)])
        retriever = HybridRetriever(service)
        retriever._lexical_index = LocalBM25Index([relevant, irrelevant])

        results = retriever.retrieve("主刷如何清理毛发")
        self.assertTrue(results)
        self.assertEqual(results[0].document.metadata["source"], "维护保养.txt")
        self.assertGreaterEqual(results[0].score, retriever.score_threshold)

    def test_rag_answer_contains_source_citation(self):
        document = Document(
            page_content="滤网应定期清理，清洗后需要完全晾干。",
            metadata={"source": "维护保养.txt", "page": 0},
        )
        vector_service = FakeVectorStoreService([(document, 0.9)])
        rag = RagSummaryService(
            model=RunnableLambda(lambda _: "滤网清洗后应完全晾干。[来源1]"),
            vector_store=vector_service,
        )
        rag.hybrid_retriever._lexical_index = LocalBM25Index([document])

        answer = rag.rag_summary("滤网清洗后怎么处理")
        self.assertIn("[来源1]", answer)
        self.assertIn("维护保养.txt", answer)

    def test_report_graph_runs_explicit_state_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "records.csv"
            csv_path.write_text(
                '"用户ID","特征","清洁效率","耗材","对比","时间"\n'
                '"1001","65㎡公寓","覆盖率85%","滤网剩余40%","优于65%用户","2025-12"\n',
                encoding="utf-8",
            )
            workflow = ReportWorkflow(
                model=RunnableLambda(lambda _: "# 扫地机器人使用情况报告与保养建议\n报告完成。[来源1]"),
                records=RecordsService(str(csv_path)),
                rag=FakeRag(),
            )
            context = UserContext("session-1", "1001", "杭州", "2025-12")

            state = workflow.invoke("生成我的使用报告", context)
            self.assertEqual(
                state["completed_steps"],
                ["resolve_context", "load_record", "retrieve_knowledge", "generate_report"],
            )
            self.assertIn("报告完成", state["report"])


if __name__ == "__main__":
    unittest.main()
