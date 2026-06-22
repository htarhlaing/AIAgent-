# 智扫通 AI Agent

面向扫地机器人和扫拖一体机器人的智能客服项目，支持产品问答、故障排查、维护保养和
个人使用报告。

## 技术栈

- Python、Streamlit
- LangChain Agent、LangGraph
- Gemini Chat / Embedding
- Chroma Vector Database
- BM25 + 向量混合检索、轻量 Reranker
- SQLite Index Manifest

## 主要能力

- 使用 LangChain Agent 调用知识检索、天气、用户上下文和使用记录工具。
- 使用 LangGraph 固定执行报告生成流程：上下文解析、数据查询、知识检索、报告生成。
- 使用 BM25 与 Chroma 向量混合召回，并加入重排、相关度阈值和来源引用。
- 使用 SQLite Manifest 管理知识文档新增、修改和删除后的向量同步。

## 安装

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env` 并填写：

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
```

## 同步知识库

在 `AI_agent` 的父目录运行：

```bash
python -m AI_agent.rag.vector_store
```

## 启动

```bash
streamlit run app.py
```

## 测试

在 `AI_agent` 的父目录运行：

```bash
python -m unittest AI_agent.tests.test_quality_architecture -v
```


