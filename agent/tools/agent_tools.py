from __future__ import annotations

import json

from langchain_core.tools import tool

from AI_agent.rag.rag_service import RagSummaryService
from AI_agent.services.context_service import get_active_user_context
from AI_agent.services.records_service import records_service
from AI_agent.utils.logger_handler import logger


rag = RagSummaryService()


@tool(description="从知识库混合检索资料，返回带相关度与来源引用的回答")
def rag_summarize(query: str) -> str:
    return rag.rag_summary(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取当前会话用户所在城市")
def get_user_location() -> str:
    context = get_active_user_context()
    return context.city if context else "杭州"


@tool(description="获取当前会话用户ID")
def get_user_id() -> str:
    context = get_active_user_context()
    return context.user_id if context else "1001"


@tool(description="获取当前报告月份，格式为YYYY-MM")
def get_current_month() -> str:
    context = get_active_user_context()
    return context.current_month if context else "2025-12"


@tool(description="获取指定用户和月份的扫地机器人使用记录；未检索到时返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    record = records_service.get_record(user_id, month)
    if record is None:
        logger.warning("未检索到用户%s在月份%s的记录", user_id, month)
        return ""
    return json.dumps(record.to_dict(), ensure_ascii=False)


@tool(description="标记报告生成场景，以兼容原有Agent提示词切换")
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"
