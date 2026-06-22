from langchain.agents import create_agent
from AI_agent.models.factory import chat_model
from AI_agent.utils.prompt_loader import load_system_prompts
from AI_agent.agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from AI_agent.agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from AI_agent.services.context_service import session_context_store, use_user_context
from AI_agent.workflows.report_workflow import ReportWorkflow
from uuid import uuid4


def normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "".join(text_parts).strip()

    return str(content).strip()


class ReactAgent:
    def __init__(self, session_id: str | None = None):
      self.session_id = session_id or str(uuid4())
      self.report_workflow = ReportWorkflow()
      self.agent=create_agent(
        model=chat_model,
        system_prompt=load_system_prompts(),
        tools=[rag_summarize, get_weather, get_user_location, get_user_id,
              get_current_month, fetch_external_data, fill_context_for_report],
        middleware=[monitor_tool, log_before_model, report_prompt_switch]
      )


    def execute_stream(self, query: str):
        user_context = session_context_store.get_or_create(self.session_id)

        if self.report_workflow.is_report_request(query):
            with use_user_context(user_context):
                state = self.report_workflow.invoke(query, user_context)
            yield state["report"]
            return

        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        with use_user_context(user_context):
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                latest_message = chunk["messages"][-1]
                text = normalize_message_content(latest_message.content)
                if text:
                    yield text + "\n"

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的城市的气温下如何保养？"):
        print(chunk, end="", flush=True)
