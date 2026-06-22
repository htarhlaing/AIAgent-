# 项目修改清单与代码导览

这份文档用于快速理解本次项目升级改了什么，以及主要代码是如何工作的。

## 一、项目现在能做什么

项目现在包含两条主要处理路线：

```text
用户问题
   ↓
ReactAgent
   ├── 普通知识问题 → Agent → Hybrid RAG → 带来源回答
   └── 使用报告问题 → LangGraph 固定工作流 → 使用报告
```

知识库索引则是另一条独立流程：

```text
扫描 data 目录
   ↓
与 SQLite Manifest 比较
   ├── 新文件 → 新增向量
   ├── 内容改变 → 更新向量
   ├── 文件删除 → 删除对应向量
   └── 没有变化 → 跳过
```

## 二、新增和修改了哪些文件

### app.py

作用：Streamlit 聊天页面入口。

修改内容：

- 增加项目父目录到 `sys.path`。
- 修复从当前目录执行 `streamlit run app.py` 时找不到 `AI_agent` 包的问题。

关键代码：

```python
PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from AI_agent.react_agent import ReactAgent
```

### react_agent.py

作用：决定问题走普通 Agent 还是报告工作流。

核心逻辑：

```python
if self.report_workflow.is_report_request(query):
    state = self.report_workflow.invoke(query, user_context)
    yield state["report"]
    return
```

如果问题包含“报告”“月报”“使用记录”等关键词，就交给 LangGraph；否则继续使用 LangChain Agent。

每个 `ReactAgent` 会获得一个固定的 `session_id`，同一聊天会话不会不断随机切换用户。

### workflows/report_workflow.py

作用：使用 LangGraph 明确定义报告生成流程。

状态结构：

```python
class ReportState(TypedDict, total=False):
    query: str
    user_id: str
    month: str
    record: dict[str, str]
    knowledge: str
    report: str
    error: str
    completed_steps: list[str]
```

每个字段代表工作流执行过程中保存的数据：

- `query`：用户问题。
- `user_id`：报告所属用户。
- `month`：报告月份。
- `record`：从 CSV 查询到的设备使用记录。
- `knowledge`：RAG 检索得到的保养知识。
- `report`：模型生成的最终报告。
- `error`：没有数据时的错误信息。
- `completed_steps`：已经执行的节点，方便测试和排查。

图结构：

```python
graph.add_edge(START, "resolve_context")
graph.add_edge("resolve_context", "load_record")
graph.add_conditional_edges(
    "load_record",
    self._route_after_record,
    {"found": "retrieve_knowledge", "missing": END},
)
graph.add_edge("retrieve_knowledge", "generate_report")
graph.add_edge("generate_report", END)
```

实际流程：

```text
resolve_context
    ↓
load_record
    ├── 没有记录 → 直接返回提示
    └── 有记录
          ↓
retrieve_knowledge
          ↓
generate_report
```

这样做的优势是报告不会漏掉数据查询步骤，也不会在没有记录时让模型编造内容。

### services/context_service.py

作用：维护稳定的会话用户信息。

```python
@dataclass(frozen=True)
class UserContext:
    session_id: str
    user_id: str
    city: str
    current_month: str
```

系统根据 `session_id` 的哈希稳定选择演示用户和城市。因此同一会话重复调用 `get_user_id()` 时会得到相同结果。

`ContextVar` 用于让 Agent 工具取得当前请求的上下文，同时避免把用户信息写成一个所有请求共享的全局变量。

### services/records_service.py

作用：读取和查询用户使用记录 CSV。

原来通过 `split(",")` 手工拆分 CSV，在字段包含引号、逗号或换行时容易出错。现在改为：

```python
with open(self.csv_path, "r", encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        ...
```

每条数据转换为：

```python
@dataclass(frozen=True)
class UsageRecord:
    user_id: str
    month: str
    feature: str
    efficiency: str
    consumption: str
    comparison: str
```

LangGraph 和 Agent 工具使用同一个 `RecordsService`，避免出现两套 CSV 解析逻辑。

### rag/hybrid_retriever.py

作用：实现关键词和向量混合检索，以及候选文档重排。

#### 1. BM25 关键词检索

中文按字切分，英文和数字按词切分：

```python
def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text.lower())
```

这部分适合查找：

- 产品型号。
- 故障码。
- “主刷”“滤网”“回充失败”等明确关键词。

#### 2. Chroma 向量检索

```python
results = self.vector_store_service.vector_store.similarity_search_with_relevance_scores(
    query,
    k=self.fetch_k,
)
```

向量检索用于找到表达不同但语义相似的资料。

如果向量服务异常，系统会记录警告并降级到 BM25，不会让整个知识问答直接失败。

#### 3. 轻量 Reranker

最终分数由三部分组成：

```python
final_score = (
    self.vector_weight * vector_score
    + self.lexical_weight * lexical_score
    + self.rerank_weight * overlap
)
```

- `vector_score`：语义相似度。
- `lexical_score`：BM25 关键词相关度。
- `overlap`：查询词在文档中的覆盖率。

低于阈值的候选会被删除：

```python
if final_score >= self.score_threshold:
    ranked.append(...)
```

### rag/rag_service.py

作用：将重排后的文档交给模型，并生成带引用的回答。

每个来源会转换成：

```python
RagSource(
    index=1,
    source="维护保养.txt",
    page=0,
    score=0.73,
    excerpt="...",
)
```

模型看到的上下文带有 `[来源1]`、`[来源2]` 等编号。回答生成后，系统统一追加来源列表：

```text
参考来源：
- [来源1] 维护保养.txt（相关度 0.73）
- [来源2] 扫地机器人100问.pdf，第 3 页（相关度 0.61）
```

如果所有候选都低于阈值，系统返回“没有检索到足够相关资料”，而不是使用不相关内容回答。

### rag/index_manifest.py

作用：使用 SQLite 保存知识库文件和 Chroma 分片之间的对应关系。

数据表结构：

```sql
CREATE TABLE indexed_documents (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_ids TEXT NOT NULL,
    indexed_at TEXT NOT NULL
)
```

字段含义：

- `path`：源文件路径。
- `content_hash`：文件当前内容哈希。
- `chunk_ids`：该文件写入 Chroma 的全部分片 ID。
- `indexed_at`：最近成功索引时间。

旧 `md5.txt` 只能判断“见过这个哈希”，无法知道需要删除哪些向量。Manifest 保存了分片 ID，因此可以精确更新和删除。

### rag/vector_store.py

作用：根据 Manifest 同步 `data` 目录与 Chroma。

入口方法：

```python
result = VectorStoreService().sync_documents()
```

新增文件：

```text
读取文件 → 切片 → 生成确定性分片 ID → 写入 Chroma → 写入 Manifest
```

修改文件：

```text
发现哈希变化 → 写入新分片 → 删除旧分片 → 更新 Manifest
```

删除文件：

```text
Manifest 有记录但文件不存在 → 删除对应 Chroma 分片 → 删除 Manifest 记录
```

同步结果使用 `SyncResult` 表示：

```python
SyncResult(
    added=1,
    updated=2,
    deleted=1,
    unchanged=3,
    failed=0,
)
```

### agent/tools/agent_tools.py

作用：给普通 Agent 提供工具。

主要变化：

- `rag_summarize` 改用增强后的混合 RAG。
- 用户 ID、位置和月份改从稳定会话上下文读取。
- `fetch_external_data` 改用统一的 `RecordsService`。
- 删除随机用户和手写 CSV 解析代码。

### config/chroma.yml

新增配置：

```yaml
fetch_k: 8
rerank_k: 3
score_threshold: 0.15
vector_weight: 0.55
lexical_weight: 0.30
rerank_weight: 0.15
manifest_path: chroma_db/index_manifest.sqlite3
```

这些参数控制召回数量、最终文档数量、最低相关度和混合检索权重。

### config/agent.yml

新增稳定演示上下文配置：

```yaml
default_report_month: "2025-12"
demo_user_ids: ["1001", "1002", ...]
demo_cities: ["深圳", "合肥", "杭州"]
```

默认月份选择 CSV 中确实存在数据的月份，避免演示时每次报告都查不到记录。

### prompts 目录

修改内容：

- RAG Prompt 要求关键事实使用 `[来源N]`。
- 主 Prompt 要求保留 RAG 来源，不编造引用。
- 报告 Prompt 要求保留保养知识引用。
- 删除要求模型公开内部思考过程的规则。

### requirements.txt

新增完整依赖清单，包括：

- LangChain
- LangGraph
- Chroma
- Gemini 适配器
- Streamlit
- PDF 解析器
- YAML 解析器

### tests/test_quality_architecture.py

增加 5 项离线测试：

1. Manifest 基本新增、更新和删除。
2. 文档同步的新增、内容修改和源文件删除。
3. 混合检索和重排顺序。
4. RAG 来源引用。
5. LangGraph 报告节点执行顺序。

测试使用 Fake Model 和 Fake Vector Store，不消耗 Gemini 配额。

## 三、如何运行

### 1. 同步知识库

在 `AI_agent` 的父目录运行：

```bash
python -m AI_agent.rag.vector_store
```

首次迁移会清理旧版无法由 Manifest 追踪的向量，并重新建立索引。

### 2. 启动项目

```bash
cd AI_agent
streamlit run app.py
```

### 3. 运行测试

```bash
cd ..
python -m unittest AI_agent.tests.test_quality_architecture -v
```

## 四、可以怎样向别人介绍这次升级

可以用下面这段话概括：

> 这个项目原本是一个基于 LangChain Agent 和向量检索的扫地机器人客服。本次升级将个人报告改造成确定性的 LangGraph 状态流，避免 Agent 漏步骤或在无数据时编造报告；知识检索升级为 BM25 与 Chroma 向量的混合召回，并通过轻量重排、相关度阈值和来源引用提高回答可信度；知识库索引则从单纯保存 MD5 升级为 SQLite Manifest，可以精确管理每个文件对应的向量分片，并支持文件新增、修改和删除。

## 五、目前仍需注意

- 当前 Reranker 是轻量本地算法，不是大型 Cross-Encoder 模型。
- 第一次建立真实向量索引会调用 Gemini Embedding，可能受到 API 配额限制。
- 会话上下文目前保存在进程内，服务重启后会重新生成。
- `md5.txt` 已不再参与索引判断，可以作为历史文件保留。
