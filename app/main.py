"""Research Paper Assistant — Streamlit UI."""

from pathlib import Path

import streamlit as st

from app.config import DATA_DIR, LLM_MODEL, EMBEDDING_MODEL
from app.ingestion.pdf_processor import process_pdf
from app.prompts.templates import PRESET_QUERIES
from app.rag.ollama_client import missing_models
from app.rag.qa_chain import ask, format_source_cards
from app.rag.vector_store import add_documents, delete_by_source, list_indexed_papers

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; }
    .source-card {
        background: #f8f9fb;
        border-left: 3px solid #4f46e5;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .source-card strong { color: #4f46e5; }
    div[data-testid="stSidebar"] { background: #fafafa; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "selected_papers": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_pdf(uploaded_file) -> Path:
    dest = DATA_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def ingest_pdf(pdf_path: Path) -> int:
    chunks = process_pdf(pdf_path)
    return add_documents(chunks)


def render_source_cards(cards: list[dict]) -> None:
    st.markdown("**Sources**")
    for card in cards:
        st.markdown(
            f'<div class="source-card">'
            f"<strong>{card['title']}</strong> — p. {card['page']}<br>"
            f"<em>{card['excerpt']}</em>"
            f"</div>",
            unsafe_allow_html=True,
        )


def run_query(question: str, mode: str = "qa", paper_title: str | None = None) -> None:
    filter_sources = st.session_state.selected_papers or None
    with st.spinner("Searching papers and generating answer..."):
        answer, docs = ask(
            question,
            filter_sources=filter_sources,
            mode=mode,
            paper_title=paper_title,
        )
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": format_source_cards(docs),
        }
    )


def main() -> None:
    init_session_state()

    st.title("📄 Research Paper Assistant")
    st.caption("Upload PDFs, ask questions, and get cited answers — 100% local & free via Ollama.")

    online, not_pulled = missing_models()
    if not online:
        st.error(
            "Ollama is not running. Install it from [ollama.com](https://ollama.com), "
            "then start it and pull the required models."
        )
        st.code(f"ollama pull {LLM_MODEL}\nollama pull {EMBEDDING_MODEL}", language="bash")
        st.stop()
    if not_pulled:
        st.error("Missing Ollama models. Pull them in a terminal:")
        st.code("\n".join(f"ollama pull {m}" for m in not_pulled), language="bash")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Papers")

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more research papers to index.",
        )

        if uploaded_files:
            if st.button("Index uploaded papers", type="primary", use_container_width=True):
                progress = st.progress(0, text="Processing...")
                for i, uf in enumerate(uploaded_files):
                    try:
                        path = save_uploaded_pdf(uf)
                        n = ingest_pdf(path)
                        st.success(f"Indexed **{uf.name}** ({n} chunks)")
                    except Exception as exc:
                        st.error(f"Failed to index {uf.name}: {exc}")
                    progress.progress((i + 1) / len(uploaded_files))
                progress.empty()
                st.rerun()

        indexed = list_indexed_papers()
        st.divider()

        if indexed:
            st.subheader(f"Indexed ({len(indexed)})")
            selected = st.multiselect(
                "Filter to specific papers (optional)",
                options=indexed,
                default=st.session_state.selected_papers,
                help="Leave empty to search all papers. Select 2+ for comparison.",
            )
            st.session_state.selected_papers = selected

            for name in indexed:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📎 {name}")
                if col2.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                    delete_by_source(name)
                    pdf_path = DATA_DIR / name
                    if pdf_path.exists():
                        pdf_path.unlink()
                    if name in st.session_state.selected_papers:
                        st.session_state.selected_papers.remove(name)
                    st.rerun()
        else:
            st.info("No papers indexed yet. Upload PDFs above.")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Main area ---
    if not indexed:
        st.markdown(
            """
            ### Get started
            1. Upload one or more research paper PDFs in the sidebar
            2. Click **Index uploaded papers**
            3. Ask a question or use a quick action below
            """
        )
        st.stop()

    # Quick actions
    st.subheader("Quick actions")
    cols = st.columns(4)
    actions = [
        ("📝 Summarize", "Summarize", "summary"),
        ("🔬 Methodology", "Methodology", "methodology"),
        ("⚠️ Limitations", "Limitations", "limitations"),
        ("⚖️ Compare", "Compare", "compare"),
    ]
    for col, (label, key, mode) in zip(cols, actions):
        with col:
            if st.button(label, use_container_width=True, key=f"preset_{key}"):
                question = PRESET_QUERIES[key]
                paper_title = None
                if mode == "summary" and len(st.session_state.selected_papers) == 1:
                    paper_title = (
                        st.session_state.selected_papers[0]
                        .replace(".pdf", "")
                        .replace("_", " ")
                    )
                if mode == "compare" and len(st.session_state.selected_papers) < 2:
                    st.warning("Select at least 2 papers in the sidebar to compare.")
                else:
                    run_query(question, mode=mode, paper_title=paper_title)
                    st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("View source excerpts"):
                    render_source_cards(msg["sources"])

    # Chat input
    if prompt := st.chat_input("Ask about your papers..."):
        run_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
