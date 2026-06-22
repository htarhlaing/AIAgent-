from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import RLock

from langchain_core.documents import Document

from AI_agent.utils.config_handler import chroma_config
from AI_agent.utils.file_handler import listdir_with_allowed_type, pdf_loader, txt_loader
from AI_agent.utils.logger_handler import logger
from AI_agent.utils.path_tool import get_abs_path


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese as characters and latin text as words without extra models."""
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text.lower())


@dataclass
class ScoredDocument:
    document: Document
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0
    overlap_score: float = 0.0


class LocalBM25Index:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(doc.page_content) for doc in documents]
        self.avg_length = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = max(len(documents), 1)
        self.idf = {
            token: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        query_tokens = tokenize(query)
        scored: list[tuple[Document, float]] = []
        for document, tokens in zip(self.documents, self.tokens):
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_length, 1)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:k]
        maximum = top[0][1] if top else 1.0
        return [(document, score / maximum) for document, score in top]


class HybridRetriever:
    """Combines Chroma similarity, local BM25, and a lightweight reranker."""

    def __init__(self, vector_store_service) -> None:
        self.vector_store_service = vector_store_service
        self._lexical_index: LocalBM25Index | None = None
        self._lock = RLock()
        self.fetch_k = int(chroma_config.get("fetch_k", 8))
        self.rerank_k = int(chroma_config.get("rerank_k", chroma_config.get("k", 3)))
        self.score_threshold = float(chroma_config.get("score_threshold", 0.15))
        self.vector_weight = float(chroma_config.get("vector_weight", 0.55))
        self.lexical_weight = float(chroma_config.get("lexical_weight", 0.30))
        self.rerank_weight = float(chroma_config.get("rerank_weight", 0.15))

    def _load_source_documents(self) -> list[Document]:
        paths = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )
        chunks: list[Document] = []
        for path in paths:
            try:
                documents = pdf_loader(path) if path.lower().endswith(".pdf") else txt_loader(path)
                for document in documents:
                    document.metadata["source"] = os.path.basename(path)
                chunks.extend(self.vector_store_service.spliter.split_documents(documents))
            except Exception as exc:
                logger.warning("构建BM25索引时跳过%s：%s", path, exc)
        return chunks

    def _get_lexical_index(self) -> LocalBM25Index:
        with self._lock:
            if self._lexical_index is None:
                self._lexical_index = LocalBM25Index(self._load_source_documents())
            return self._lexical_index

    @staticmethod
    def _document_key(document: Document) -> str:
        source = str(document.metadata.get("source", ""))
        page = str(document.metadata.get("page", ""))
        return f"{source}|{page}|{document.page_content}"

    def _vector_search(self, query: str) -> list[tuple[Document, float]]:
        if self.vector_store_service.count() == 0:
            return []
        try:
            results = self.vector_store_service.vector_store.similarity_search_with_relevance_scores(
                query, k=self.fetch_k
            )
            return [(document, max(0.0, min(1.0, float(score)))) for document, score in results]
        except Exception as exc:
            logger.warning("向量检索失败，降级为BM25检索：%s", exc)
            return []

    def retrieve(self, query: str) -> list[ScoredDocument]:
        candidates: dict[str, dict[str, object]] = defaultdict(dict)
        for document, score in self._vector_search(query):
            candidates[self._document_key(document)].update(document=document, vector_score=score)
        for document, score in self._get_lexical_index().search(query, self.fetch_k):
            candidates[self._document_key(document)].update(document=document, lexical_score=score)

        query_tokens = set(tokenize(query))
        ranked: list[ScoredDocument] = []
        for candidate in candidates.values():
            document = candidate["document"]
            document_tokens = set(tokenize(document.page_content))
            overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            if query.strip() and query.strip() in document.page_content:
                overlap = min(1.0, overlap + 0.25)
            vector_score = float(candidate.get("vector_score", 0.0))
            lexical_score = float(candidate.get("lexical_score", 0.0))
            final_score = (
                self.vector_weight * vector_score
                + self.lexical_weight * lexical_score
                + self.rerank_weight * overlap
            )
            if final_score >= self.score_threshold:
                ranked.append(
                    ScoredDocument(
                        document=document,
                        score=final_score,
                        vector_score=vector_score,
                        lexical_score=lexical_score,
                        overlap_score=overlap,
                    )
                )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: self.rerank_k]
