"""Unified RAG module for both modeling and code retrieval."""
import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
import json

_EMBEDDING_MODEL = "text-embedding-v4"
_RAG_LLM_MODEL = "qwen-plus-2025-09-11"

_SYSTEM_MODELING = """You are a mathematical optimization modeling expert. You must **NOT create, summarize, or rewrite** any content."""

_SYSTEM_CODE = """You are a mathematical optimization modeling expert. You must **NOT create, summarize, or rewrite** any content."""

_PROMPT_MODELING = """Given the following context retrieved from a knowledge base of modeling examples:
{context}

And given the current user question:
"{question}"

Your task is to identify the two most relevant modeling cases from the context by applying the following priority criteria in order:

**Priority 1**: Same problem_type (e.g., "Linear Programming (LP)", "Mixed-Integer Linear Programming (MILP)")
**Priority 2**: Same problem_subtype (e.g., "Blending & Mixing", "Discrete Scheduling & Assignment")
**Priority 3**: Similar problem description semantics
Focus on examples that provide complete Gurobipy implementations to help understand how to translate mathematical models into executable code.

Each example should follow this JSON-like structure:
{{
  "Problem description": "the original complete problem description",
  "Mathematical Model": {{
    "VARIABLES": "A concise description about variables and its shape or type",
    "CONSTRAINTS": "A mathematical Formula about constraints",
    "OBJECTIVE": "A mathematical Formula about objective"
    }}
}}

Now, return your answer strictly in the following format:
example1: <first relevant modeling case summary>
example2: <second relevant modeling case summary>

If only one relevant case is found, set example2 to "example2: N/A".
If no relevant case is found, return exactly:
"No relevant information found."
"""

_PROMPT_CODE = """Given the following context retrieved from a knowledge base of modeling examples:
{context}

And given the current user question:
"{question}"

Your task is to identify the two most relevant modeling cases with complete code implementations from the context by applying the following priority criteria in order:

**Priority 1**: Same problem_type (e.g., "Linear Programming (LP)", "Mixed-Integer Linear Programming (MILP)")
**Priority 2**: Same problem_subtype (e.g., "Blending & Mixing", "Discrete Scheduling & Assignment")
**Priority 3**: Similar problem description semantics
Focus on examples that provide complete Gurobipy implementations to help understand how to translate mathematical models into executable code.

Each example should follow this JSON-like structure:
{{
  "Problem description": "the original complete problem",
  "Mathematical Model": {{
    "VARIABLES": "A concise description about variables and its shape or type",
    "CONSTRAINTS": "A mathematical Formula about constraints",
    "OBJECTIVE": "A mathematical Formula about objective"
    }},
  "Code": "Complete Gurobipy implementation code from the knowledge base"
}}
Pay special attention to the Gurobipy implementation patterns for variable definition, constraint formulation, and objective function setup.

Now, return your answer strictly in the following format:
example1: <first relevant modeling case with complete code>
example2: <second relevant modeling case with complete code>

If only one relevant case is found, set example2 to "example2: N/A".
If no relevant case is found, return exactly:
"No relevant information found."
"""


def _load_json_lines_from_md(file_path: str) -> list[Document]:
    """Load JSON objects line-by-line from a .md file."""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(('#', '<!--', '```')):
                continue
            try:
                data = json.loads(stripped)
                documents.append(Document(
                    page_content=stripped,
                    metadata={
                        "source": file_path,
                        "line": line_num,
                        "problem_type": data.get("problem_type", "general"),
                        "problem_subtype": data.get("problem_subtype", "general"),
                    }
                ))
            except json.JSONDecodeError:
                continue
    return documents


class KnowledgeRetriever:
    """Thread-safe singleton for RAG knowledge retrieval."""

    _instances = {}

    def __new__(cls, mode="modeling", data_path=None, persist_dir=None, api_key=None):
        key = (mode, data_path, persist_dir)
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            instance._mode = mode
            instance._data_path = data_path or str(_project_root / "datasets" / "rag_data" / "few_shot_subtype.md")
            instance._persist_dir = persist_dir or str(_project_root / "chroma_db" / "few_shot_subtype")
            instance._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            instance._vector_store = None
            instance._llm = None
            cls._instances[key] = instance
        return cls._instances[key]

    def _ensure_initialized(self):
        if self._initialized:
            return
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment or init args.")

        embeddings = DashScopeEmbeddings(
            model=_EMBEDDING_MODEL,
            dashscope_api_key=self._api_key,
        )

        if not os.path.exists(self._persist_dir):
            print(f"[RAG] Building vector DB from {self._data_path} ...")
            docs = _load_json_lines_from_md(self._data_path)
            if not docs:
                raise RuntimeError("No valid JSON lines found in the data file.")
            Chroma.from_documents(
                docs, embeddings, persist_directory=self._persist_dir,
                collection_name="knowledge_base"
            )
            print(f"[RAG] Vector DB built with {len(docs)} examples.")

        self._vector_store = Chroma(
            persist_directory=self._persist_dir,
            embedding_function=embeddings,
            collection_name="knowledge_base"
        )

        self._llm = ChatTongyi(
            model_name=_RAG_LLM_MODEL,
            temperature=0,
            dashscope_api_key=self._api_key,
        )

        self._system_prompt = SystemMessage(
            content=_SYSTEM_MODELING if self._mode == "modeling" else _SYSTEM_CODE
        )
        self._prompt_template = _PROMPT_MODELING if self._mode == "modeling" else _PROMPT_CODE
        self._initialized = True

    def retrieve(self, query: str) -> str:
        self._ensure_initialized()
        try:
            docs = self._vector_store.similarity_search(query, k=3)
            context = "\n\n".join(doc.page_content for doc in docs)
            user_text = self._prompt_template.format(context=context, question=query)
            messages = [self._system_prompt, HumanMessage(content=user_text)]
            response = self._llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return f"Retrieval failed: {str(e)}"
