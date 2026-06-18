# Research Paper Assistant

A RAG-powered web app for querying research papers. Upload PDFs, ask natural-language questions, and get answers backed by inline citations using [Ollama](https://ollama.com).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![Ollama](https://img.shields.io/badge/Ollama-Local-black)

## Features

- **Multi-PDF upload** — index several papers into a shared vector store
- **Question answering** — free-form chat with retrieved context
- **Inline citations** — answers reference `[Paper Title, p. N]`
- **Quick actions** — summarize, methodology, limitations, compare
- **Paper filtering** — scope queries to one or more selected papers
- **Source excerpts** — expandable citations showing retrieved text
- **Zero API cost** — runs entirely on your machine

## Example questions

```
Summarize this paper.
What methodology did the authors use?
Compare paper A and paper B.
What are the limitations?
Which papers discuss eventual consistency?
```

## Architecture

```
PDF Upload → PyMuPDF extraction → Chunking → Ollama Embeddings → ChromaDB
                                                                      ↓
User Question → Retriever (top-k) → Context + Prompt → Ollama LLM → Cited Answer
```

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| PDF parsing | PyMuPDF |
| Embeddings | Ollama `nomic-embed-text` |
| Vector store | ChromaDB (local persistence) |
| LLM | Ollama `llama3.2:3b` |
| Orchestration | LangChain |

## Setup

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

> **Hardware:** `llama3.2:3b` runs on most laptops (8 GB RAM minimum). For better answers on a stronger machine, set `LLM_MODEL=llama3.2` in `.env`.

### 2. Install Python dependencies

```bash
cd "Research Assistant"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser. 

Optional: copy `.env.example` to `.env` to customize models or chunk settings.

## Usage

1. Upload one or more PDF research papers in the sidebar
2. Click **Index uploaded papers**
3. Optionally select specific papers to filter queries
4. Use quick actions or type a question in the chat box
5. Expand **View source excerpts** to see retrieved passages

**Compare mode:** select at least 2 papers in the sidebar, then click **Compare**.

## Project structure

```
Research Assistant/
├── app/
│   ├── main.py              # Streamlit UI
│   ├── config.py            # Environment config
│   ├── ingestion/
│   │   └── pdf_processor.py # PDF extract + chunk
│   ├── rag/
│   │   ├── vector_store.py  # ChromaDB operations
│   │   ├── qa_chain.py      # Retrieval + LLM chain
│   │   └── ollama_client.py # Ollama health checks
│   └── prompts/
│       └── templates.py     # Prompt templates
├── data/                    # Uploaded PDFs (gitignored)
├── chroma_db/               # Vector index (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2:3b` | Chat model (must be pulled in Ollama) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `6` | Retrieved chunks per query |


