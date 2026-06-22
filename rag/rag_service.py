from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from AI_agent.models.factory import chat_model
from AI_agent.rag.hybrid_retriever import HybridRetriever, ScoredDocument
from AI_agent.rag.vector_store import VectorStoreService
from AI_agent.utils.prompt_loader import load_rag_prompts


@dataclass(frozen=True)
class RagSource:
    index: int
    source: str
    page: int | None
    score: float
    excerpt: str

    @property
    def label(self) -> str:
        page_text = f"，第 {self.page + 1} 页" if self.page is not None else ""
        return f"[来源{self.index}] {self.source}{page_text}"


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: list[RagSource]

    def to_text(self) -> str:
        if not self.sources:
            return self.answer
        source_lines = "\n".join(f"- {source.label}（相关度 {source.score:.2f}）" for source in self.sources)
        return f"{self.answer}\n\n参考来源：\n{source_lines}"


class RagSummaryService:
    def __init__(self, model=chat_model, vector_store: VectorStoreService | None = None) -> None:
        self.vector_store = vector_store or VectorStoreService()
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.chain = self.prompt_template | model | StrOutputParser()

    def retrieve_documents(self, query: str) -> list[ScoredDocument]:
        return self.hybrid_retriever.retrieve(query)

    @staticmethod
    def _source_from_scored(index: int, scored: ScoredDocument) -> RagSource:
        metadata = scored.document.metadata
        source = os.path.basename(str(metadata.get("source", "未知资料")))
        page_value = metadata.get("page")
        page = int(page_value) if isinstance(page_value, int) else None
        excerpt = " ".join(scored.document.page_content.split())[:160]
        return RagSource(index=index, source=source, page=page, score=scored.score, excerpt=excerpt)

    def answer(self, query: str) -> RagResponse:
        scored_documents = self.retrieve_documents(query)
        if not scored_documents:
            return RagResponse(
                answer="当前知识库中没有检索到足够相关的资料。请补充产品型号或更具体的问题。",
                sources=[],
            )

        sources = [
            self._source_from_scored(index, scored)
            for index, scored in enumerate(scored_documents, start=1)
        ]
        context_parts = []
        for source, scored in zip(sources, scored_documents):
            context_parts.append(
                f"{source.label}\n相关度：{source.score:.3f}\n内容：{scored.document.page_content}"
            )
        answer = self.chain.invoke({"input": query, "context": "\n\n".join(context_parts)})
        return RagResponse(answer=answer.strip(), sources=sources)

    def rag_summary(self, query: str) -> str:
        return self.answer(query).to_text()


if __name__ == "__main__":
    print(RagSummaryService().rag_summary("小户型适合哪些扫地机器人"))
