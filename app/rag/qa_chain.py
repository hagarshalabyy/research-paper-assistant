"""RAG query engine with citation formatting."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from app.prompts.templates import (
    COMPARE_PROMPT,
    LIMITATIONS_PROMPT,
    METHODOLOGY_PROMPT,
    QA_PROMPT,
    SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from app.rag.vector_store import get_retriever


def format_context(docs: list[Document]) -> str:
    """Format retrieved chunks with citation labels for the LLM."""
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("paper_title", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[Excerpt {i}] Paper: {title} | Page: {page}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def retrieve(
    question: str,
    filter_sources: list[str] | None = None,
    top_k: int = TOP_K,
) -> list[Document]:
    retriever = get_retriever(filter_sources=filter_sources, top_k=top_k)
    return retriever.invoke(question)


def _build_chain(template: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", template),
        ]
    )
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )
    return prompt | llm | StrOutputParser()


def ask(
    question: str,
    filter_sources: list[str] | None = None,
    mode: str = "qa",
    paper_title: str | None = None,
) -> tuple[str, list[Document]]:
    """Run a RAG query and return (answer, source_documents)."""
    docs = retrieve(question, filter_sources=filter_sources)
    context = format_context(docs)

    if mode == "summary" and paper_title:
        template = SUMMARY_PROMPT
        chain = _build_chain(template)
        answer = chain.invoke({"context": context, "paper_title": paper_title})
    elif mode == "limitations":
        chain = _build_chain(LIMITATIONS_PROMPT)
        answer = chain.invoke({"context": context})
    elif mode == "compare":
        chain = _build_chain(COMPARE_PROMPT)
        answer = chain.invoke({"context": context})
    elif mode == "methodology":
        chain = _build_chain(METHODOLOGY_PROMPT)
        answer = chain.invoke({"context": context})
    else:
        chain = _build_chain(QA_PROMPT)
        answer = chain.invoke({"context": context, "question": question})

    return answer, docs


def format_source_cards(docs: list[Document]) -> list[dict]:
    """Prepare source excerpts for UI display."""
    seen: set[tuple] = set()
    cards: list[dict] = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            {
                "title": doc.metadata.get("paper_title", doc.metadata.get("source")),
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "excerpt": doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
            }
        )
    return cards
