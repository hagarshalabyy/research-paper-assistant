"""Prompt templates for research-paper Q&A tasks."""

SYSTEM_PROMPT = """You are a research assistant that answers questions about academic papers.
You must:
- Base answers ONLY on the provided context excerpts.
- Cite sources inline using [Paper Title, p. N] format for every factual claim.
- Say clearly when the context does not contain enough information.
- Use precise academic language while staying readable.
- For comparisons, structure the answer with clear sections per paper."""

QA_PROMPT = """Context excerpts from research papers:
{context}

Question: {question}

Answer with inline citations. If comparing papers, use headings for each paper."""

SUMMARY_PROMPT = """Context excerpts from the paper "{paper_title}":
{context}

Provide a structured summary covering:
1. **Problem & Motivation**
2. **Methodology**
3. **Key Results**
4. **Contributions**

Cite specific sections with [Paper Title, p. N]."""

LIMITATIONS_PROMPT = """Context excerpts from research papers:
{context}

Extract and summarize:
1. **Limitations** explicitly stated by the authors
2. **Future Work** or open problems mentioned

Quote or paraphrase closely and cite with [Paper Title, p. N].
If a section is not found in the context, say so."""

COMPARE_PROMPT = """Context excerpts from multiple research papers:
{context}

Compare the papers on:
1. **Research Problem**
2. **Methodology**
3. **Key Findings**
4. **Strengths & Weaknesses**

Use headings per paper and cite with [Paper Title, p. N]."""

METHODOLOGY_PROMPT = """Context excerpts from research papers:
{context}

Describe the methodology used in detail:
- Experimental setup, datasets, models, or theoretical approach
- Evaluation metrics
- Any ablation or validation steps

Cite with [Paper Title, p. N]."""

PRESET_QUERIES = {
    "Summarize": "Summarize this paper with key contributions and results.",
    "Methodology": "What methodology did the authors use?",
    "Limitations": "What are the limitations and future work mentioned?",
    "Compare": "Compare the uploaded papers across problem, method, and findings.",
}
