# LLM Eval Framework

[![LLM Eval CI](https://github.com/dharanimurugaraj/llm-eval-framework/actions/workflows/eval_ci.yml/badge.svg)](https://github.com/dharanimurugaraj/llm-eval-framework/actions/workflows/eval_ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://llm-eval-frameworkgit-rugym9w2eulg68fnb4q9ct.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A production-grade RAG evaluation framework that systematically benchmarks chunking strategies, embedding models, and retrieval quality using RAGAS metrics and Weights & Biases experiment tracking.
>
> **Key finding:** Semantic chunking achieves an overall score of **0.84**, a **29% improvement in context recall** (0.57 → 0.85) over fixed-size chunking, with Gemini embedding-001 delivering perfect context precision (1.00) across all strategies.

---

## 🚀 Live Demo
**[View the live dashboard →](https://llm-eval-frameworkgit-rugym9w2eulg68fnb4q9ct.streamlit.app/)**

Explore 3 chunking strategy experiments with real RAGAS scores,
W&B experiment tracking, and A/B comparison charts.

---

## The Problem

RAG systems fail silently. A pipeline that scores well on toy examples can hallucinate on production documents — and without systematic evaluation, you won't know until a user catches it. The three most common failure modes are: answers that sound correct but contradict the source documents (faithfulness failure), retrieved chunks that miss the information actually needed to answer the question (low recall), and no regression testing, meaning every configuration change is a blind bet.

This framework makes those failure modes measurable, comparable, and trackable across experiments.

---

## What This Measures

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Catches hallucination — the model inventing facts not in the source |
| **Answer Relevancy** | Does the answer actually address the question asked? | Catches evasive or off-topic answers |
| **Context Recall** | Did retrieval surface all the information needed to answer? | Catches gaps — the right answer existed but wasn't retrieved |
| **Context Precision** | Were the retrieved chunks relevant (no noise)? | Catches retrieval pollution — irrelevant chunks confusing the LLM |
| **Overall Score** | Mean of the four metrics above | Single comparable number per experiment run |

---

## Experimental Results

Three chunking strategies evaluated on the same 8-question benchmark over the same document corpus:

| Strategy | Overall | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|---|---|---|---|---|---|
| `recursive` | 0.73 | 0.51 | 0.75 | 0.66 | 1.00 |
| `fixed_size` | 0.75 | 0.67 | 0.75 | 0.57 | 1.00 |
| **`semantic`** | **0.84** | **0.75** | **0.75** | **0.85** | **1.00** |

**Key findings:**

- **Semantic chunking wins overall (0.84 vs 0.73–0.75):** Grouping sentences by meaning keeps related facts together, driving context recall from 0.57 (fixed-size) to 0.85 — a 49% relative improvement. The LLM receives complete context rather than fragmented paragraphs.
- **Context Precision = 1.00 across all strategies:** Every retrieved chunk was relevant. This is a strong signal that Gemini `embedding-001` produces high-quality semantic representations — zero retrieval noise regardless of chunking strategy.
- **Recursive chunking trades recall for faithfulness risk:** Recursive scored the lowest faithfulness (0.51) despite reasonable recall (0.66). Paragraph boundary cuts fragment supporting evidence, causing the LLM to partially ground answers in incomplete chunks.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LLM EVAL FRAMEWORK                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  PDF Documents                                       │
│       │                                             │
│       ▼                                             │
│  ┌─────────────┐    ┌──────────────┐               │
│  │  Document   │    │   Chunking   │               │
│  │   Loader    │───▶│  Strategies  │               │
│  │  (PyPDF)    │    │  fixed_size  │               │
│  └─────────────┘    │  recursive   │               │
│                     │  semantic    │               │
│                     └──────┬───────┘               │
│                            │                        │
│                            ▼                        │
│                     ┌──────────────┐               │
│                     │  Embeddings  │               │
│                     │   Gemini     │               │
│                     │ embedding-001│               │
│                     │  (3072-dim)  │               │
│                     └──────┬───────┘               │
│                            │                        │
│                            ▼                        │
│                     ┌──────────────┐               │
│                     │ Qdrant Cloud │               │
│                     │ Vector Store │               │
│                     └──────┬───────┘               │
│                            │                        │
│          Query ───────────▶│                        │
│                            ▼                        │
│                     ┌──────────────┐               │
│                     │  RAG Pipeline│               │
│                     │    Groq      │               │
│                     │ llama-3.1-8b │               │
│                     └──────┬───────┘               │
│                            │                        │
│                            ▼                        │
│  ┌─────────────┐    ┌──────────────┐               │
│  │  Weights &  │◀───│    RAGAS     │               │
│  │   Biases    │    │  Evaluation  │               │
│  │  Tracking   │    │  4 Metrics   │               │
│  └─────────────┘    └──────┬───────┘               │
│                            │                        │
│                            ▼                        │
│                     ┌──────────────┐               │
│                     │  Streamlit   │               │
│                     │  Dashboard   │               │
│                     │  4 Pages     │               │
│                     └──────────────┘               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| **Embeddings** | Google Gemini `embedding-001` (3072-dim) | Document and query vectorisation |
| **Vector Store** | Qdrant Cloud | High-performance semantic similarity search |
| **LLM** | Groq `llama-3.1-8b-instant` | Fast, cost-effective RAG answer generation |
| **Evaluation** | RAGAS | Automated RAG quality scoring (4 metrics) |
| **Experiment Tracking** | Weights & Biases | Run comparison, metric logging, visualisation |
| **Dashboard** | Streamlit | Interactive UI for running and comparing experiments |
| **Backend** | FastAPI | API layer for evaluation orchestration |
| **Document Loading** | PyPDF | PDF parsing and text extraction |
| **CI/CD** | GitHub Actions | Automated lint + test on every push |
| **Language** | Python 3.12 | Type-hinted throughout |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier)
- [Groq API key](https://console.groq.com/) (free tier)
- [Google AI Studio API key](https://aistudio.google.com/) (free)
- [Weights & Biases](https://wandb.ai/) account (free)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/dharanimurugaraj/llm-eval-framework.git
cd llm-eval-framework

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install all dependencies
pip install -e ".[dev]"

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Add your PDF documents
mkdir -p data/raw
# Copy your PDFs into data/raw/

# 6. Run ingestion (chunks + embeds + stores in Qdrant)
python ingestion/pipeline.py

# 7. Run evaluation
py -3.12 eval/run_evaluation.py

# 8. Launch the dashboard
py -3.12 -m streamlit run dashboard/app.py
```

### Environment Variables

```bash
# .env
GOOGLE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=your_groq_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION_NAME=llm_eval_docs
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=llm-eval-framework
```

### Running Experiments

Compare chunking strategies programmatically:

```python
py -3.12 -c "
from eval.run_evaluation import run_full_evaluation

run_full_evaluation(
    experiment_name='my_experiment',
    chunking_strategy='semantic',   # 'recursive', 'fixed_size', 'semantic'
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5,
)
"
```

Or trigger from the Streamlit dashboard → **🔬 Run Evaluation** page.

---

## Project Structure

```
llm-eval-framework/
├── backend/              # FastAPI app + RAG pipeline
│   ├── api.py            # REST endpoints
│   └── rag_pipeline.py   # Retrieval + generation orchestration
├── dashboard/            # Streamlit 4-page UI
│   └── app.py            # Overview, Run Eval, History, A/B Compare
├── data/
│   ├── raw/              # Input PDF documents (gitignored)
│   └── processed/        # eval_results.json output
├── docs/                 # Architecture documentation
├── eval/                 # Evaluation engine
│   ├── eval_dataset.py   # 8-question ground-truth benchmark
│   ├── ragas_eval.py     # RAGAS metric computation
│   └── run_evaluation.py # End-to-end evaluation runner
├── experiments/          # W&B experiment tracking
│   └── tracker.py        # Run logging + metric sync
├── ingestion/            # Document processing pipeline
│   ├── chunker.py        # 3 chunking strategies
│   ├── embedder.py       # Gemini embedding wrapper
│   ├── pipeline.py       # End-to-end ingestion orchestrator
│   └── vector_store.py   # Qdrant client wrapper
├── tests/                # 13 unit tests (all green)
│   ├── test_eval.py
│   ├── test_ingestion.py
│   └── test_vector_store.py
├── .github/workflows/    # CI/CD pipeline
│   └── eval_ci.yml
├── pyproject.toml        # Dependencies + tool config
└── .env.example          # Environment variable template
```

---

## CI/CD

Every push to `main` triggers:

1. **Ruff lint** — enforces style and catches unused imports across the entire codebase
2. **Unit tests (13 tests)** — validates chunkers, RAGAS dataset builder, vector store search, RAG pipeline output keys, and experiment tracker
3. **Coverage report** — generated with `pytest-cov` (non-blocking)

All runs are visible at:  
[https://github.com/dharanimurugaraj/llm-eval-framework/actions](https://github.com/dharanimurugaraj/llm-eval-framework/actions)

---

## Experiment Tracking

Every evaluation run is automatically logged to Weights & Biases with the full metric set (faithfulness, answer relevancy, context recall, context precision, overall score), configuration (chunking strategy, chunk size, overlap, top-k), and number of samples evaluated. This makes it trivial to compare experiments across sessions and identify regressions.

**Live W&B dashboard:**  
[https://wandb.ai/dharani-vyrenzo-vyrenzo-in/llm-eval-framework](https://wandb.ai/dharani-vyrenzo-vyrenzo-in/llm-eval-framework)

---

## Key Findings & Next Steps

### What We Found

1. **Semantic chunking is the clear winner (overall: 0.84)** — grouping sentences by semantic similarity rather than character count produces dramatically better context recall (0.85 vs 0.57 for fixed-size), giving the LLM complete, coherent context to generate faithful answers.
2. **Gemini `embedding-001` retrieval is noise-free** — context precision of 1.00 across all three strategies means every retrieved chunk was relevant. The embedding model is not the bottleneck; chunking strategy is.
3. **Recursive chunking has a faithfulness problem (0.51)** — the lowest faithfulness score despite reasonable recall. Paragraph-boundary cuts fragment multi-sentence evidence, causing the LLM to partially hallucinate when connecting incomplete chunks.

### How To Improve Scores

1. **Increase `top_k` from 5 to 10** — higher recall at the cost of slightly more noise; worthwhile for complex multi-fact questions where relevant information is spread across multiple chunks.
2. **Add a reranker (e.g. Cohere Rerank)** between retrieval and generation — rerank the top-10 chunks back to 5 using a cross-encoder to eliminate the few irrelevant results and push precision even higher.
3. **Increase chunk overlap to 300 characters** — reduces boundary artifacts in recursive chunking where a sentence split across two chunks causes both halves to be retrieved without their supporting context.

---

## Author

**Dharani Murugaraj**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/dharani-murugaraj)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/dharanimurugaraj)
