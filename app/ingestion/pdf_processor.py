"""PDF text extraction and document chunking."""

from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text_from_pdf(pdf_path: Path) -> list[Document]:
    """Extract page-level text from a PDF with metadata."""
    docs: list[Document] = []
    paper_title = pdf_path.stem.replace("_", " ").replace("-", " ")

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "paper_title": paper_title,
                        "page": page_num,
                        "file_path": str(pdf_path),
                    },
                )
            )
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split page documents into smaller chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_pdf(pdf_path: Path) -> list[Document]:
    """Full pipeline: extract pages then chunk."""
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path.name}")
    return chunk_documents(pages)
