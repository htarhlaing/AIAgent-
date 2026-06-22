from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AI_agent.models.factory import embed_model
from AI_agent.rag.index_manifest import IndexManifest, ManifestEntry
from AI_agent.utils.config_handler import chroma_config
from AI_agent.utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from AI_agent.utils.logger_handler import logger
from AI_agent.utils.path_tool import get_abs_path


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    failed: int = 0


class VectorStoreService:
    def __init__(self):
        persist_directory = get_abs_path(chroma_config["persist_directory"])
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )
        manifest_path = chroma_config.get("manifest_path", "chroma_db/index_manifest.sqlite3")
        self.manifest = IndexManifest(get_abs_path(manifest_path))

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def count(self) -> int:
        return int(self.vector_store._collection.count())

    @staticmethod
    def _load_file(path: str) -> list[Document]:
        if path.lower().endswith(".pdf"):
            return pdf_loader(path)
        if path.lower().endswith(".txt"):
            return txt_loader(path)
        return []

    @staticmethod
    def _chunk_ids(path: str, content_hash: str, count: int) -> list[str]:
        return [
            hashlib.sha256(f"{path}|{content_hash}|{index}".encode("utf-8")).hexdigest()
            for index in range(count)
        ]

    def _index_file(self, path: str, content_hash: str, previous: ManifestEntry | None) -> None:
        documents = self._load_file(path)
        for document in documents:
            document.metadata.update(
                source=os.path.basename(path),
                source_path=os.path.abspath(path),
                content_hash=content_hash,
            )
        chunks = self.spliter.split_documents(documents)
        if not chunks:
            raise ValueError("文档加载或分片结果为空")

        new_chunk_ids = self._chunk_ids(path, content_hash, len(chunks))
        self.vector_store.add_documents(chunks, ids=new_chunk_ids)
        if previous and previous.chunk_ids:
            self.vector_store.delete(ids=previous.chunk_ids)
        self.manifest.upsert(path, content_hash, new_chunk_ids)

    def sync_documents(self) -> SyncResult:
        """Synchronize source files and Chroma, including adds, updates, and deletes."""
        result = SyncResult()
        source_paths = {
            os.path.abspath(path)
            for path in listdir_with_allowed_type(
                get_abs_path(chroma_config["data_path"]),
                tuple(chroma_config["allow_knowledge_file_type"]),
            )
        }
        entries = self.manifest.all()

        # One-time migration from the old MD5-only scheme: untracked vectors cannot
        # participate in reliable updates/deletes, so rebuild them under Manifest control.
        if not entries and self.count() > 0:
            untracked_ids = self.vector_store.get(include=[]).get("ids", [])
            if untracked_ids:
                self.vector_store.delete(ids=untracked_ids)
                logger.info("已清理 %s 个旧版未受Manifest管理的向量分片", len(untracked_ids))

        for removed_path in set(entries) - source_paths:
            entry = entries[removed_path]
            try:
                if entry.chunk_ids:
                    self.vector_store.delete(ids=entry.chunk_ids)
                self.manifest.remove(removed_path)
                result.deleted += 1
                logger.info("已从知识库删除文档：%s", removed_path)
            except Exception as exc:
                result.failed += 1
                logger.error("删除知识库文档失败：%s，原因：%s", removed_path, exc)

        for path in sorted(source_paths):
            content_hash = get_file_md5_hex(path)
            if not content_hash:
                result.failed += 1
                continue
            previous = entries.get(path)
            if previous and previous.content_hash == content_hash:
                result.unchanged += 1
                continue
            try:
                self._index_file(path, content_hash, previous)
                if previous:
                    result.updated += 1
                    logger.info("已更新知识库文档：%s", path)
                else:
                    result.added += 1
                    logger.info("已新增知识库文档：%s", path)
            except Exception as exc:
                result.failed += 1
                logger.error("同步知识库文档失败：%s，原因：%s", path, exc, exc_info=True)
        return result

    def load_document(self) -> SyncResult:
        """Backward-compatible alias for the old indexing command."""
        return self.sync_documents()


if __name__ == "__main__":
    sync_result = VectorStoreService().sync_documents()
    print(sync_result)
