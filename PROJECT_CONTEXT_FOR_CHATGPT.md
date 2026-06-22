# AI Agent 项目说明（供 ChatGPT 阅读）

> 请基于本文档理解、分析并协助改进这个项目。本文档未包含任何 API Key 或其他敏感凭据。

## 1. 项目简介

这是一个面向扫地机器人和扫拖一体机器人的智能客服项目，主要使用：

- Streamlit：聊天界面
- LangChain Agent：意图判断和工具调用
- LangGraph/LangChain Middleware：工具监控和动态 Prompt 切换
- Chroma：本地向量数据库
- Gemini：聊天模型和 Embedding 模型
- RAG：从本地 PDF/TXT 知识库检索资料后生成回答

项目目前支持三类主要场景：

1. 扫地机器人知识问答、选购、维护和故障排查。
2. 根据用户所在城市及天气给出使用或保养建议。
3. 查询用户月度使用数据并生成个人使用报告。

## 2. 项目目录

```text
AI_agent/
├── app.py                         # Streamlit 页面入口
├── react_agent.py                 # Agent 创建和流式执行
├── agent/
│   └── tools/
│       ├── agent_tools.py         # RAG、天气、用户及报告工具
│       └── middleware.py          # 日志、工具监控、动态 Prompt
├── rag/
│   ├── rag_service.py             # 检索并调用模型总结
│   └── vector_store.py            # 文档加载、切片和 Chroma 索引
├── models/
│   └── factory.py                 # Chat/Embedding 模型工厂
├── utils/
│   ├── config_handler.py          # YAML 配置加载
│   ├── file_handler.py            # PDF/TXT 加载和 MD5 计算
│   ├── logger_handler.py          # 日志配置
│   ├── path_tool.py               # 项目绝对路径工具
│   └── prompt_loader.py           # Prompt 文件加载
├── config/
│   ├── agent.yml
│   ├── chroma.yml
│   ├── prompt.yml
│   └── rag.yml
├── prompts/
│   ├── main_prompt.txt            # 主客服 Prompt
│   ├── rag_summarize.txt          # RAG 总结 Prompt
│   └── report_prompt.txt          # 使用报告 Prompt
├── data/
│   ├── *.txt / *.pdf              # 扫地机器人知识资料
│   └── external/records.csv       # 模拟用户月度使用记录
├── logs/                          # 运行日志
└── md5.txt                        # 已索引文件的 MD5
```

## 3. 主要运行链路

```text
用户在 Streamlit 输入问题
            ↓
ReactAgent.execute_stream()
            ↓
LangChain Agent 分析问题并选择工具
            ├── rag_summarize
            │       ↓
            │   Chroma Retriever
            │       ↓
            │   Gemini 总结检索内容
            │
            ├── get_user_location → get_weather
            │
            └── get_user_id/get_current_month
                    ↓
              fill_context_for_report
                    ↓
              动态切换报告 Prompt
                    ↓
              fetch_external_data
            ↓
Agent 组织最终回答
            ↓
Streamlit 显示结果
```

## 4. 核心模块说明

### app.py

- 创建 Streamlit 聊天页面。
- 将 Agent 对象保存在 `st.session_state`。
- 保存前端聊天记录。
- 调用 `ReactAgent.execute_stream()` 获取回答。
- 当前前端虽然保存历史消息，但历史没有传给 Agent，因此模型实际上没有多轮记忆。

### react_agent.py

- 使用 `create_agent()` 创建 Agent。
- 注册 RAG、天气、位置、用户数据和报告工具。
- 注册工具监控、模型调用日志和报告 Prompt 切换中间件。
- 每次执行只向 Agent 传入当前用户问题。

### agent/tools/agent_tools.py

提供以下工具：

- `rag_summarize(query)`：从知识库检索资料。
- `get_weather(city)`：返回模拟天气。
- `get_user_location()`：随机返回模拟城市。
- `get_user_id()`：随机返回模拟用户 ID。
- `get_current_month()`：随机返回模拟月份。
- `fetch_external_data(user_id, month)`：从 CSV 查询用户记录。
- `fill_context_for_report()`：触发报告场景的 Prompt 切换。

### agent/tools/middleware.py

- `monitor_tool`：记录工具名称、参数、成功或失败状态。
- `log_before_model`：记录模型调用前的消息状态。
- `report_prompt_switch`：根据运行时 `report` 标记选择客服 Prompt 或报告 Prompt。

### rag/rag_service.py

- 从 Chroma Retriever 获取相关文档。
- 将问题与文档内容填充到 RAG Prompt。
- 调用 Gemini 生成基于资料的中文答案。

### rag/vector_store.py

- 加载 `data` 目录下的 TXT 和 PDF。
- 使用 `RecursiveCharacterTextSplitter` 切片。
- 使用 Gemini Embedding 生成向量。
- 写入 Chroma。
- 使用 MD5 避免重复处理文件。

## 5. 当前配置

Chroma 的主要参数：

```yaml
collection_name: agent
persist_directory: chroma_db
k: 3
chunk_size: 200
chunk_overlap: 20
allow_knowledge_file_type: [txt, pdf]
```

实际模型目前在 `models/factory.py` 中直接指定：

- Chat model：`gemini-2.5-flash`
- Embedding model：`gemini-embedding-001`

## 6. 当前已知问题

### 高优先级

1. `app.py` 导入 `AI_agent.agent.react_agent`，但 `react_agent.py` 实际位于项目根目录，当前入口会产生 `ModuleNotFoundError`。
2. `models/factory.py` 曾将 Google API Key 硬编码在源码中。必须撤销旧 Key，并改用环境变量或 Secret 管理。
3. `rag_service.py` 的 `return` 位于遍历检索结果的循环内部，导致只使用第一条文档；没有检索结果时可能返回 `None`。
4. Chroma 的持久化目录使用相对路径，启动目录不同会访问不同的向量库。
5. `md5.txt` 和 Chroma 数据没有事务一致性，可能出现文件被标记为已处理，但向量库不存在或写入失败的情况。

### 中优先级

1. 前端消息历史没有传入 Agent，不支持真正多轮对话。
2. 用户 ID、月份和位置是随机值，同一会话结果不稳定。
3. `fetch_external_data()` 使用字符串 `split(',')` 解析 CSV，应该改用 Python `csv` 模块或数据模型。
4. 工具注解声明返回字符串，但部分工具实际返回字典。
5. `rag.yml` 中的模型配置没有真正被模型工厂使用。
6. 模型和 RAG 服务在模块导入阶段全局初始化，启动耦合较重，也不利于测试。
7. 当前“流式输出”主要是将完整状态结果逐字显示，不是真正的模型 Token Streaming。
8. 主 Prompt 要求输出模型的真实思考过程，不适合生产环境；应该只展示简洁的工具执行状态和最终答案。

### 工程化缺失

- 没有 README。
- 没有 `requirements.txt` 或 `pyproject.toml`。
- 没有自动化测试。
- 没有 `.env.example`。
- 没有 Dockerfile 和 CI/CD。
- 没有 RAG 质量评估数据集。

## 7. 建议的目标架构

建议将“所有问题都交给自由 Agent”升级为“确定性工作流优先、Agent 负责复杂场景”：

```text
Streamlit/Web Frontend
          ↓
FastAPI / Conversation Service
          ↓
Intent Router
   ├── 知识问题 → Hybrid Retrieval → Reranker → 带引用回答
   ├── 用户报告 → LangGraph 固定状态工作流
   ├── 天气问题 → 真实天气 API Adapter
   └── 复杂任务 → Agent 工具规划
          ↓
Session Store / Cache / Observability / Evaluation
```

推荐改进：

1. 改成标准 `src/ai_agent/` 包结构。
2. 使用 Pydantic Settings 管理模型参数、路径和环境变量。
3. 将 Streamlit UI 与后端服务分离。
4. 使用 LangGraph 明确定义用户报告流程。
5. RAG 加入 BM25 + Vector 混合检索、Reranker、相关度阈值和来源引用。
6. 给知识库建立 Manifest，支持文件新增、修改、删除和索引版本管理。
7. 使用 Pydantic 定义工具输入输出。
8. 保存会话历史，并实现可控的短期记忆。
9. 增加重试、超时、限流、缓存和友好的错误降级。
10. 建立 pytest 测试和 RAG 离线评估集。

## 8. 推荐实施顺序

### P0：恢复可运行和安全状态

- 修复入口导入路径。
- 撤销源码中暴露过的 API Key。
- 改用环境变量。
- 修复 RAG 循环提前返回。
- 将 Chroma 路径转换为项目绝对路径。
- 增加依赖清单和启动说明。

### P1：改善功能质量

- 接入聊天历史。
- 固定当前会话的用户上下文。
- 改用标准 CSV 解析。
- 增加检索来源、相关度阈值和无结果处理。
- 为工具增加结构化输入输出。

### P2：升级架构

- 分离 FastAPI 后端和 Streamlit 前端。
- 使用 LangGraph 实现报告工作流。
- 增加混合检索和 Reranker。
- 建立独立、可重复执行的知识库索引任务。

### P3：生产化

- 自动化测试和评估。
- 日志、Tracing 和指标监控。
- Docker 与 CI/CD。
- 鉴权、限流、缓存和数据隐私控制。

## 9. 希望 ChatGPT 协助的任务

请先基于以上信息回答：

1. 你对当前架构和问题判断是否认同？
2. 请给出适合该项目规模的目标目录结构。
3. 请提供分阶段重构方案，避免一次性重写。
4. 优先解决会阻止项目运行、安全性或 RAG 正确性的问题。
5. 修改代码时，请说明涉及的文件、原因和验证方式。

