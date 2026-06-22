# LangGraph 报告流、增强 RAG 与索引 Manifest 实施说明

本文档说明本次重构完成了什么、各文件代表什么，以及如何运行和调整。

## 1. 本次实施范围

本次按最终确认的范围完成三项改造：

1. 使用 LangGraph 明确定义报告生成状态流。
2. RAG 加入关键词与向量混合检索、轻量重排、相关度阈值和来源引用。
3. 使用 SQLite Manifest 管理文档索引，支持新增、内容修改和文件删除。

API 拆分等其他改造不在本次范围内，现有 Streamlit 启动方式保持不变。

## 2. 新的整体调用关系

```text
用户问题
   ↓
ReactAgent
   ├── 报告意图
   │      ↓
   │   LangGraph ReportWorkflow
   │      ├── resolve_context
   │      ├── load_record
   │      ├── retrieve_knowledge
   │      └── generate_report
   │
   └── 普通知识问题
          ↓
       Agent 调用 rag_summarize
          ↓
       HybridRetriever
          ├── Chroma 向量检索
          ├── 本地 BM25 关键词检索
          └── 轻量 Reranker + 阈值过滤
          ↓
       带 [来源N] 的回答
```

## 3. LangGraph 报告生成状态流

核心文件：[workflows/report_workflow.py](workflows/report_workflow.py)

报告不再完全依赖 Agent 自由决定调用顺序，而是使用以下固定状态：

```text
START
  ↓
resolve_context
  ↓
load_record
  ├── 找不到数据 → END（返回可用月份）
  └── 找到数据
          ↓
retrieve_knowledge
          ↓
generate_report
          ↓
         END
```

### ReportState 字段

- `query`：用户原始问题。
- `user_id`：从问题或当前会话上下文得到的用户 ID。
- `month`：从问题或默认报告月份得到的月份。
- `record`：CSV 中的结构化使用记录。
- `knowledge`：增强 RAG 返回的保养知识和来源。
- `report`：最终报告。
- `error`：没有记录等业务错误。
- `completed_steps`：本次实际执行过的节点，便于测试和排查。

### 节点职责

#### resolve_context

确认用户 ID 和月份。用户问题中存在明确 ID 或月份时优先使用，否则采用当前会话上下文。

#### load_record

通过 `RecordsService` 查询 CSV。没有数据时直接结束，不再让模型根据空数据编造报告。

#### retrieve_knowledge

根据清洁效率、耗材状态和用户对比信息生成检索问题，调用增强后的 RAG 获取维护建议。

#### generate_report

将用户记录与 RAG 资料一起交给模型，生成 Markdown 报告，并要求保留 `[来源N]` 标记。

### 如何接入现有 Agent

[react_agent.py](react_agent.py) 会先调用 `ReportWorkflow.is_report_request()`。含有“报告”“月报”“使用记录”“使用情况”等意图时走 LangGraph；其他问题仍使用原有 Agent。

## 4. 混合检索与 Reranker

核心文件：[rag/hybrid_retriever.py](rag/hybrid_retriever.py)

### 关键词检索

使用本地 BM25 实现，不需要下载额外模型。中文按字、英文和数字按词切分，因此可以覆盖设备型号、故障码和中文关键词。

### 向量检索

从 Chroma 取回语义相关文档。向量库为空时不会发起无意义的 Embedding 查询；向量服务异常时会自动降级到 BM25。

### 轻量重排

每个候选文档最终得分由三部分组成：

```text
最终得分 = 向量得分 × vector_weight
         + BM25 得分 × lexical_weight
         + 查询词覆盖率 × rerank_weight
```

这是一种本地轻量 Reranker，不需要额外部署 Cross-Encoder。以后如需更高精度，可以保持调用接口不变，将覆盖率部分替换成专用重排模型。

### 阈值过滤

低于 `score_threshold` 的文档会被丢弃。如果所有文档均低于阈值，系统会明确表示没有足够相关资料，不会强行生成答案。

### 当前配置

配置文件：[config/chroma.yml](config/chroma.yml)

```yaml
fetch_k: 8             # 向量和 BM25 各自召回的候选数量
rerank_k: 3            # 重排后最终提供给模型的文档数
score_threshold: 0.15  # 最低相关度
vector_weight: 0.55    # 向量检索权重
lexical_weight: 0.30   # BM25 权重
rerank_weight: 0.15    # 查询词覆盖率权重
```

如果系统经常回答“没有相关资料”，可逐步将阈值调低到 `0.10`；如果召回内容太宽泛，可提高到 `0.20` 左右。应通过测试问题集调整，不建议只凭单个问题修改。

## 5. 来源引用

核心文件：[rag/rag_service.py](rag/rag_service.py)

每个进入模型的文档都会带有编号：

```text
[来源1] 维护保养.txt
[来源2] 扫地机器人100问.pdf，第 3 页
```

模型被要求在事实后保留 `[来源N]`，系统还会在回答末尾统一追加文件名、PDF 页码和相关度。

示例：

```text
滤网清洗后应完全晾干再装回，以免影响电机和过滤效果。[来源1]

参考来源：
- [来源1] 维护保养.txt（相关度 0.73）
```

同步更新的 Prompt：

- [prompts/rag_summarize.txt](prompts/rag_summarize.txt)
- [prompts/main_prompt.txt](prompts/main_prompt.txt)
- [prompts/report_prompt.txt](prompts/report_prompt.txt)

## 6. SQLite Manifest 文档索引

核心文件：

- [rag/index_manifest.py](rag/index_manifest.py)
- [rag/vector_store.py](rag/vector_store.py)

旧方案只把 MD5 写入 `md5.txt`，不知道某个文件对应 Chroma 中的哪些分片，因此无法可靠删除或更新。

新 Manifest 使用 SQLite 保存：

```text
path            文件绝对路径
content_hash    当前内容哈希
chunk_ids       写入 Chroma 的所有分片 ID
indexed_at      最近成功索引时间
```

默认数据库位置：

```text
chroma_db/index_manifest.sqlite3
```

### 新增文件

Manifest 中没有该路径时，文档会被加载、切片、向量化并写入 Chroma，然后登记所有分片 ID。

### 修改文件

路径相同但内容哈希变化时：

1. 写入新内容产生的新分片。
2. 删除旧 Manifest 中记录的分片 ID。
3. 更新 Manifest。

先写新内容再删除旧内容，可以降低 Embedding 失败导致原索引丢失的风险。

### 删除文件

Manifest 中存在、但 `data` 目录已经不存在的文件，会根据保存的分片 ID 从 Chroma 删除，同时删除 Manifest 记录。

### 从旧 MD5 方案迁移

第一次运行时，如果 Manifest 为空但 Chroma 中已有无法追踪的旧分片，系统会清理旧分片并重新建立受 Manifest 管理的索引。旧的 `md5.txt` 不再参与判断，可以留作历史记录，也可以手动归档。

### 执行索引同步

在 `AI_agent` 的父目录运行：

```bash
python -m AI_agent.rag.vector_store
```

输出示例：

```text
SyncResult(added=6, updated=0, deleted=0, unchanged=0, failed=0)
```

再次运行且文件没有变化时应显示为 `unchanged`，不会重复生成向量。

## 7. 数据与上下文辅助服务

### services/records_service.py

使用标准 `csv.DictReader` 读取带引号和多行字段的真实 CSV，并返回 `UsageRecord`，供工具和 LangGraph 共用。

### services/context_service.py

为 Streamlit 会话生成稳定的演示用户、城市和默认报告月份。它用于保证同一个 Agent 会话生成报告时不会随机切换用户。

当前默认月份在 [config/agent.yml](config/agent.yml) 中设置为数据集存在的 `2025-12`。

## 8. 测试

测试文件：[tests/test_quality_architecture.py](tests/test_quality_architecture.py)

覆盖内容：

1. Manifest 新增、更新和删除。
2. 混合检索与重排顺序。
3. 相关度阈值和来源引用。
4. LangGraph 是否严格执行四个报告节点。

测试使用离线 Fake Model，不消耗 Gemini 配额。

运行命令：

```bash
cd ..
python -m unittest AI_agent.tests.test_quality_architecture -v
```

## 9. 日常运行

先同步知识库：

```bash
cd ..
python -m AI_agent.rag.vector_store
```

再启动页面：

```bash
cd AI_agent
streamlit run app.py
```

## 10. 后续可继续升级的地方

- 将轻量重排替换为 BGE Reranker 或其他 Cross-Encoder。
- 为 Manifest 增加索引版本和 Embedding 模型版本，模型变化时自动重建。
- 将报告 LangGraph 的执行步骤展示在前端。
- 建立固定问题集，按召回率、引用准确率和答案正确性调整阈值及权重。
