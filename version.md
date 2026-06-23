# MIRROR-v2 — Version Roadmap

---

## V2 改动 — 去除链式写法，改为直接 LLM 调用

### 改动概述

参考 `langchain1.0.py` 中 `create_agent(model=..., tools=..., system_prompt=...)` 的设计思想，将 prompt 作为 system_prompt 直接传入、通过消息列表驱动，去除了 `PromptTemplate | LLM` 的 LCEL 链式写法，改为在 `BaseExpert` 中直接构建 `SystemMessage` + `HumanMessage` 并调用 `llm.invoke(messages)`。

核心变化：不再使用 `self.forward_chain` / `self.revision_chain`，统一使用 `self.forward(**kwargs)` / `self.backward(**kwargs)`。

### 具体改动清单

#### `agent_teams/base_expert.py` — 核心改动

**删除**：
- `from langchain_core.prompts import PromptTemplate`
- `self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK`
- `self.forward_chain = self.forward_prompt | self.llm`
- revision chain 相关逻辑

**新增**：
- `from langchain_core.messages import HumanMessage, SystemMessage`
- `_build_messages(task_template, **kwargs)` 方法：将 `ROLE_DESCRIPTION` 作为 `SystemMessage`，将 `FORWARD_TASK.format(**kwargs)` 作为 `HumanMessage`
- `forward(**kwargs)`：直接调用 `self.llm.invoke(messages).content`
- `backward(**kwargs)`：同上，使用 `_revision_task`

#### `utils/rag.py`

**删除**：
- `from langchain_core.runnables import RunnablePassthrough`
- `from langchain_core.prompts import PromptTemplate`
- `{"context": retriever, "question": RunnablePassthrough()} | prompt | llm` 链

**改为**：
- `self._vector_store.similarity_search(query, k=3)` 手动获取文档
- 拼接 context 字符串
- 构建 `[SystemMessage, HumanMessage(content=prompt_template.format(context=..., question=...))]`
- 调用 `self._llm.invoke(messages)`

#### 专家子类 — 调用方式变更

所有子类仅将 `self.forward_chain.invoke(kwargs)` 改为 `super().forward(**kwargs)`，`self.revision_chain.invoke(kwargs)` 改为 `super().backward(**kwargs)`，传入的 kwargs 名称和值完全不变：

| 文件 | v1 调用 | v2 调用 |
|------|---------|---------|
| `model_expert.py` forward | `self.forward_chain.invoke({"problem_description": ..., "comments_text": ..., "knowledge_context": ...})` | `super().forward(problem_description=..., comments_text=..., knowledge_context=...)` |
| `model_expert.py` revision | `self.revision_chain.invoke({"problem_description": ..., "original_model": ..., "error_message": ..., "last_tip": ..., "last_code": ...})` | `super().backward(problem_description=..., original_model=..., error_message=..., last_tip=..., last_code=...)` |
| `code_expert.py` forward | 同 model_expert | `super().forward(...)` |
| `code_expert.py` revision | 同 model_expert | `super().backward(...)` |
| `parameter_extractor.py` | `self.forward_chain.invoke({"problem_description": ..., "comment_text": ...})` | `super().forward(...)` |
| `modeling_advisor.py` | `self.forward_chain.invoke({"problem_description": ..., "comment_text": ...})` | `super().forward(...)` |
| `solver.py` | `self.forward_chain.invoke({"comment_text": ..., "attention": ...})` | `super().forward(...)` |

#### 不变文件

以下文件零改动：`utils/json_parser.py`、`utils/comment_pool.py`、`utils/comment.py`、`utils/test_generated_code.py`、`utils/utils.py`、`utils/result.py`、所有 shim 文件、`main.py`、`run_rag_exp.py`、`requirements.txt`、`.env`。

### v1 → v2 上下文传输对比

| 维度 | v1（链式） | v2（直接调用） | 是否一致 |
|------|-----------|---------------|----------|
| ROLE_DESCRIPTION 位置 | 拼接在 prompt 文本开头 | 作为 SystemMessage | 语义等价 |
| FORWARD_TASK 占位符 | `PromptTemplate.format(**kwargs)` | `str.format(**kwargs)` | **完全一致** |
| 所有 kwargs 名称 | dict key 传入 | 直接传 kwargs | **完全一致** |
| RAG 的 context | `RunnablePassthrough` 透传 | `similarity_search` 手动拼接 | **内容相同** |
| 模型创建 | `ChatTongyi(model_name=..., ...)` | 同上 | **不变** |

### 设计动机

- `langchain1.0.py` 展示的新范式是 `create_agent(model=..., system_prompt=..., tools=...)` —— prompt 直接作为字符串传入，不经过 `PromptTemplate` 编译成 chain
- 移除了 `RunnablePassthrough`、`PromptTemplate` 等 LCEL 组件，代码更直白
- 模型创建方式保持不变（`ChatTongyi`）

---

*本文档随版本迭代持续更新。*
