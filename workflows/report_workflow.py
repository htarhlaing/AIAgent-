from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from AI_agent.models.factory import chat_model
from AI_agent.rag.rag_service import RagSummaryService
from AI_agent.services.context_service import UserContext
from AI_agent.services.records_service import RecordsService, records_service
from AI_agent.utils.prompt_loader import load_report_prompts


class ReportState(TypedDict, total=False):
    query: str
    user_id: str
    month: str
    record: dict[str, str]
    knowledge: str
    report: str
    error: str
    completed_steps: list[str]


class ReportWorkflow:
    """Explicit report flow: resolve context → load data → RAG advice → report."""

    def __init__(
        self,
        model=chat_model,
        records: RecordsService = records_service,
        rag: RagSummaryService | None = None,
    ) -> None:
        self.records = records
        self.rag = rag or RagSummaryService(model=model)
        self.report_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", load_report_prompts()),
                    (
                        "human",
                        "用户问题：{query}\n用户ID：{user_id}\n月份：{month}\n"
                        "使用记录：{record}\n检索到的保养知识：{knowledge}\n"
                        "请生成最终报告，并保留知识内容中的来源编号。",
                    ),
                ]
            )
            | model
            | StrOutputParser()
        )
        self.graph = self._build_graph()

    @staticmethod
    def is_report_request(query: str) -> bool:
        return any(keyword in query for keyword in ("报告", "月报", "使用记录", "使用情况"))

    @staticmethod
    def _extract_user_id(query: str, fallback: str) -> str:
        match = re.search(r"(?<!\d)(10\d{2})(?!\d)", query)
        return match.group(1) if match else fallback

    @staticmethod
    def _extract_month(query: str, fallback: str) -> str:
        full_match = re.search(r"(20\d{2})\s*[-/年]\s*(1[0-2]|0?[1-9])\s*月?", query)
        if full_match:
            return f"{full_match.group(1)}-{int(full_match.group(2)):02d}"
        month_match = re.search(r"(?<!\d)(1[0-2]|0?[1-9])\s*月", query)
        if month_match:
            return f"{fallback[:4]}-{int(month_match.group(1)):02d}"
        return fallback

    def _resolve_context_node(self, state: ReportState) -> ReportState:
        return {
            "user_id": state["user_id"],
            "month": state["month"],
            "completed_steps": ["resolve_context"],
        }

    def _load_record_node(self, state: ReportState) -> ReportState:
        record = self.records.get_record(state["user_id"], state["month"])
        steps = [*state.get("completed_steps", []), "load_record"]
        if record is None:
            months = self.records.available_months(state["user_id"])
            available = "、".join(months) if months else "无"
            return {
                "error": f"没有找到用户 {state['user_id']} 在 {state['month']} 的记录。可用月份：{available}。",
                "report": f"暂时无法生成报告：没有找到用户 {state['user_id']} 在 {state['month']} 的使用记录。可用月份：{available}。",
                "completed_steps": steps,
            }
        return {"record": record.to_dict(), "completed_steps": steps}

    @staticmethod
    def _route_after_record(state: ReportState) -> Literal["found", "missing"]:
        return "found" if state.get("record") else "missing"

    def _retrieve_knowledge_node(self, state: ReportState) -> ReportState:
        record = state["record"]
        query = (
            "根据以下扫地机器人使用表现给出维护保养建议："
            f"{record['efficiency']}；{record['consumption']}；{record['comparison']}"
        )
        knowledge = self.rag.rag_summary(query)
        return {
            "knowledge": knowledge,
            "completed_steps": [*state.get("completed_steps", []), "retrieve_knowledge"],
        }

    def _generate_report_node(self, state: ReportState) -> ReportState:
        report = self.report_chain.invoke(
            {
                "query": state["query"],
                "user_id": state["user_id"],
                "month": state["month"],
                "record": json.dumps(state["record"], ensure_ascii=False),
                "knowledge": state["knowledge"],
            }
        )
        return {
            "report": report.strip(),
            "completed_steps": [*state.get("completed_steps", []), "generate_report"],
        }

    def _build_graph(self):
        graph = StateGraph(ReportState)
        graph.add_node("resolve_context", self._resolve_context_node)
        graph.add_node("load_record", self._load_record_node)
        graph.add_node("retrieve_knowledge", self._retrieve_knowledge_node)
        graph.add_node("generate_report", self._generate_report_node)
        graph.add_edge(START, "resolve_context")
        graph.add_edge("resolve_context", "load_record")
        graph.add_conditional_edges(
            "load_record",
            self._route_after_record,
            {"found": "retrieve_knowledge", "missing": END},
        )
        graph.add_edge("retrieve_knowledge", "generate_report")
        graph.add_edge("generate_report", END)
        return graph.compile(name="robot_usage_report")

    def invoke(self, query: str, context: UserContext) -> ReportState:
        user_id = self._extract_user_id(query, context.user_id)
        month = self._extract_month(query, context.current_month)
        return self.graph.invoke(
            {"query": query, "user_id": user_id, "month": month, "completed_steps": []}
        )
