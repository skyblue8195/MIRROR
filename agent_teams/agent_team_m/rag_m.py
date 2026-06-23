"""Backward-compatible shim — use utils.rag.KnowledgeRetriever directly."""
from utils.rag import KnowledgeRetriever

_retriever = None


def retrieve_knowledge(query: str, data_path=None, persist_dir=None, api_key=None) -> str:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever(
            mode="modeling", data_path=data_path, persist_dir=persist_dir, api_key=api_key,
        )
    return _retriever.retrieve(query)
