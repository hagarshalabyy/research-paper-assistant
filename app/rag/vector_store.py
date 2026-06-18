"""Vector store and embedding utilities."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, TOP_K


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store(collection_name: str = "papers") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def add_documents(documents: list[Document], collection_name: str = "papers") -> int:
    """Index documents into Chroma. Returns number of chunks added."""
    store = get_vector_store(collection_name)
    ids = store.add_documents(documents)
    return len(ids)


def delete_by_source(source_filename: str, collection_name: str = "papers") -> None:
    """Remove all chunks belonging to a given PDF filename."""
    store = get_vector_store(collection_name)
    store.delete(where={"source": source_filename})


def list_indexed_papers(collection_name: str = "papers") -> list[str]:
    """Return unique paper filenames currently in the index."""
    store = get_vector_store(collection_name)
    result = store.get(include=["metadatas"])
    sources = {m.get("source") for m in result.get("metadatas", []) if m}
    return sorted(s for s in sources if s)


def get_retriever(
    collection_name: str = "papers",
    top_k: int = TOP_K,
    filter_sources: list[str] | None = None,
):
    """Build a retriever, optionally filtered to specific paper filenames."""
    store = get_vector_store(collection_name)
    search_kwargs: dict = {"k": top_k}
    if filter_sources:
        if len(filter_sources) == 1:
            search_kwargs["filter"] = {"source": filter_sources[0]}
        else:
            search_kwargs["filter"] = {"source": {"$in": filter_sources}}
    return store.as_retriever(search_kwargs=search_kwargs)
