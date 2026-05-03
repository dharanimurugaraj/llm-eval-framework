# Architecture Deep Dive

## System Overview

The LLM Eval Framework is a modular, pipeline-oriented system designed to make RAG quality measurable and reproducible. Each component has a single responsibility, communicates through well-defined interfaces, and is independently testable. The architecture is intentionally free-tier compatible: every external service (Qdrant Cloud, Groq, Gemini, W&B) has a free tier sufficient for development and experimentation.

The pipeline flows in two phases:

1. **Ingestion Phase** — Documents are loaded, chunked by strategy, embedded, and stored in Qdrant. This is run once per experiment configuration change.
2. **Evaluation Phase** — A fixed set of benchmark questions is passed through the RAG pipeline, answers and retrieved contexts are collected, RAGAS scores all outputs against ground-truth answers, and results are logged to W&B and saved locally.

---

## Component Details

### Document Ingestion Pipeline (`ingestion/pipeline.py`)

The `IngestionPipeline` orchestrates the full ingestion flow. It accepts `chunking_strategy`, `chunk_size`, `chunk_overlap`, and `embedding_model` as parameters, enabling fully parameterised experiment configurations. On each run it:

1. Loads all PDFs from `data/raw/` using PyPDF
2. Instantiates the selected chunker
3. Splits documents into chunks with metadata attached (`source`, `page`, `chunk_strategy`, `chunk_index`)
4. Embeds all chunks via `EmbeddingManager`
5. Upserts into Qdrant, optionally recreating the collection to ensure a clean experiment slate

The `recreate_collection=True` flag ensures no cross-contamination between experiments using different chunking strategies.

---

### Chunking Strategies (`ingestion/chunker.py`)

Three strategies are implemented, each with different tradeoffs:

| Strategy | Mechanism | Strengths | Weaknesses |
|---|---|---|---|
| `fixed_size` | Split every N characters regardless of content | Fast, predictable, no API calls | Cuts mid-sentence; fragments evidence |
| `recursive` | Try `\n\n` → `\n` → `. ` → ` ` → `""` separator hierarchy | Respects paragraph/sentence boundaries | Can still fragment multi-paragraph evidence |
| `semantic` | Embed each sentence; split where cosine distance exceeds threshold | Groups related facts together | Slowest; requires OpenAI embeddings for splitting |

**Why semantic chunking wins:** It aligns chunk boundaries with meaning shifts rather than character positions. A multi-sentence explanation of a financial metric stays in a single chunk; with fixed-size splitting it may span two chunks with neither containing the complete evidence.

**The `get_chunker()` factory** provides a unified interface so `IngestionPipeline` and the evaluation runner can select any strategy by name without conditional logic in the calling code.

---

### Embedding Layer (`ingestion/embedder.py`)

`EmbeddingManager` wraps two embedding backends:

- **Gemini `embedding-001`** (default): 3072-dimensional vectors via `langchain-google-genai`. Free tier, high quality, used for all experiments in this project.
- **OpenAI `text-embedding-3-small`**: 1536-dimensional vectors as an alternative backend.

The dimensionality is read from `get_model_info()` and passed to Qdrant collection creation automatically, ensuring the vector store is always correctly sized for the active embedding model.

---

### Vector Store (`ingestion/vector_store.py`)

`VectorStoreManager` wraps the Qdrant Python client. Key design decisions:

- **Cosine distance** is used for all collections — standard for text similarity tasks
- **Batched upsert** (100 points per batch) prevents timeout errors on large document sets
- **`timeout=60.0`** on the Qdrant client prevents write timeouts during embedding-heavy ingestion
- The `search()` method returns plain Python dicts (not Qdrant types) so the rest of the codebase is decoupled from Qdrant's SDK

The collection name is read from `QDRANT_COLLECTION_NAME` in `.env`, making it trivial to test against a separate collection without touching code.

---

### RAG Pipeline (`backend/rag_pipeline.py`)

`RAGPipeline` takes a natural language query and returns a structured dict containing:

```python
{
    "question": str,       # original query
    "answer": str,         # LLM-generated answer
    "contexts": list[str], # retrieved chunk texts (for RAGAS)
    "source_chunks": list, # full chunk metadata
    "model": str,          # which LLM was used
    "top_k": int,          # how many chunks were retrieved
}
```

The `contexts` key is the RAGAS contract — RAGAS expects a list of retrieved text strings alongside the question, answer, and ground truth for metric computation. Every key `RAGPipeline.run()` returns maps directly to what `RAGASEvaluator.build_dataset()` consumes.

**LLM:** Groq `llama-3.1-8b-instant` via `langchain-groq`. Temperature 0 for deterministic evaluation runs.

---

### Evaluation Pipeline (`eval/run_evaluation.py`, `eval/ragas_eval.py`)

`run_full_evaluation()` is the top-level entry point, wiring together:

1. `IngestionPipeline` — fresh vector store for the experiment config
2. `RAGPipeline` — answers all 8 benchmark questions
3. `RAGASEvaluator` — scores all answers against ground truth
4. `ExperimentTracker` — logs to W&B
5. JSON persistence — saves to `data/processed/eval_results.json`

**Rate limit handling:** Groq free tier enforces a 6,000 TPM limit. The `run_with_retry()` utility wraps all Groq API calls with exponential backoff (base delay 30s for RAG, 45s for RAGAS evaluation). A 3-second inter-question delay prevents burst exhaustion across the 8-question benchmark.

**`RAGASEvaluator`** uses:
- Judge LLM: Groq `llama-3.1-8b-instant` (same model, temperature 0)
- Judge embeddings: Gemini `embedding-001` (for answer relevancy cosine similarity)
- `answer_relevancy.strictness = 1` — prevents RAGAS from requesting parallel LLM completions, which Groq blocks
- `RunConfig(max_workers=1)` — forces sequential metric computation to avoid burst rate limits

---

### Experiment Tracking (`experiments/tracker.py`)

`ExperimentTracker` wraps the W&B Python SDK. Each evaluation run logs:
- All four RAGAS metric scores + overall
- Full experiment configuration (strategy, chunk size, overlap, top-k, model names)
- Number of samples evaluated

The W&B run name matches `experiment_name`, making it easy to filter and compare runs in the W&B UI.

---

### Dashboard (`dashboard/app.py`)

A 4-page Streamlit app:

| Page | Purpose |
|---|---|
| **🏠 Overview** | Latest run metrics, radar chart, per-question breakdown table |
| **🔬 Run Evaluation** | Form to configure and trigger a new experiment; displays results inline |
| **📈 Experiment History** | All W&B runs fetched via API; trends chart; best-run callout |
| **⚖️ A/B Comparison** | Side-by-side metric comparison of any two runs with verdict |

W&B data is cached for 5 minutes (`@st.cache_data(ttl=300)`) to avoid slow API calls on every page navigation.

---

## Data Flow

```
1. RAW PDF
   └─ PyPDF extracts text pages → list[Document]

2. CHUNKS
   └─ Chunker splits each page → list[Document]
      Metadata attached: source, page, chunk_strategy, chunk_index, chunk_size

3. VECTORS
   └─ EmbeddingManager.embed_documents(texts) → list[list[float]]  (3072-dim)
      PointStruct(id=i, vector=vec, payload={text, source, page, ...})
      Upserted to Qdrant collection in batches of 100

4. RETRIEVAL
   └─ EmbeddingManager.embed_query(question) → query_vector
      QdrantClient.query_points(query=query_vector, limit=top_k)
      Returns top-k hits sorted by cosine similarity

5. GENERATION
   └─ format_context(chunks) → context string (numbered, with source tags)
      ChatGroq.invoke([system_prompt, human_message]) → answer string
      Returns: {question, answer, contexts, source_chunks, model, top_k}

6. RAGAS SCORING
   └─ Dataset.from_dict({question, answer, contexts, ground_truth})
      ragas.evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                        context_recall, context_precision])
      Returns: EvaluationResult → DataFrame → per-metric means

7. PERSISTENCE
   └─ wandb.log({faithfulness, answer_relevancy, ...})
      json.dump(results, eval_results.json)
      Streamlit dashboard reads eval_results.json on next load
```

---

## Design Decisions

### 1. Why Qdrant over ChromaDB

Qdrant is production-ready with a managed cloud offering (free tier), gRPC + REST API, and native support for payload filtering. ChromaDB is excellent for local prototyping but has no managed cloud tier comparable to Qdrant's. For a framework designed to demonstrate production engineering, Qdrant signals the right intent.

### 2. Why Gemini `embedding-001` over OpenAI `text-embedding-3-small`

Two reasons: cost and dimensions. Gemini `embedding-001` is free via Google AI Studio API (generous rate limits), while OpenAI charges per token. The 3072-dimension output is richer than OpenAI's 1536-dim `text-embedding-3-small`, and the experimental results validate the quality — context precision of 1.00 across all experiments means zero retrieval noise. The only tradeoff is that semantic chunking still uses OpenAI embeddings (LangChain's `SemanticChunker` depends on `OpenAIEmbeddings`), which is a known gap for future work.

### 3. Why Groq over OpenAI for LLM

Groq provides free-tier inference at very high speeds (hundreds of tokens/second). For an evaluation framework where the LLM is used both for answer generation and as a judge (via RAGAS), keeping inference costs at zero is critical for accessibility. `llama-3.1-8b-instant` is a strong open-weight model that produces reliable JSON outputs for RAGAS's structured prompts.

### 4. Why RAGAS over Manual Evaluation

Manual evaluation doesn't scale and isn't reproducible. RAGAS provides four complementary automated metrics that together catch the distinct ways a RAG system can fail: hallucination (faithfulness), off-topic answers (answer relevancy), missing information (context recall), and noisy retrieval (context precision). It uses an LLM-as-judge pattern that, while imperfect, is consistent across runs, making it suitable for relative comparison between experiments.

### 5. Why Sequential Evaluation (`max_workers=1`) over Parallel

Groq's free tier enforces a 6,000 tokens-per-minute (TPM) limit. RAGAS with `max_workers=8` (the default) sends 8 concurrent LLM requests, exhausting the TPM limit within seconds and triggering 429 errors mid-evaluation. Setting `max_workers=1` makes evaluation sequential, completing ~32 metric computations (4 metrics × 8 questions) in 3–5 minutes instead of 30 seconds, but with zero rate limit failures. Retry logic with exponential backoff (`run_with_retry()`) provides a safety net for the rare transient error.

---

## Experiment Design

### Controlled Variables (keep constant between experiments)

- Document corpus (`data/raw/` — same PDFs)
- Benchmark questions (`eval/eval_dataset.py` — same 8 QA pairs)
- Embedding model (`gemini-embedding-001`)
- LLM (`llama-3.1-8b-instant`, temperature 0)
- `top_k` (5, unless specifically testing retrieval depth)

### Independent Variables (change one per experiment)

- `chunking_strategy`: `fixed_size`, `recursive`, `semantic`
- `chunk_size`: 500, 1000, 1500, 2000 characters
- `chunk_overlap`: 0, 100, 200, 300 characters
- `top_k`: 3, 5, 10

### Running a Controlled Experiment

```python
from eval.run_evaluation import run_full_evaluation

# Baseline
run_full_evaluation(
    experiment_name="baseline_recursive_k5",
    chunking_strategy="recursive",
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5,
)

# Test: larger overlap
run_full_evaluation(
    experiment_name="recursive_overlap300_k5",
    chunking_strategy="recursive",
    chunk_size=1000,
    chunk_overlap=300,   # only this changes
    top_k=5,
)
```

Compare runs directly in the W&B dashboard at:  
[https://wandb.ai/dharani-vyrenzo-vyrenzo-in/llm-eval-framework](https://wandb.ai/dharani-vyrenzo-vyrenzo-in/llm-eval-framework)
