# LLM Eval Framework

> A production-grade evaluation framework for RAG pipelines — benchmark
> chunking strategies, embedding models, and LLM outputs with RAGAS,
> DeepEval, and Weights & Biases.

## The Problem
Most RAG applications are shipped without any systematic evaluation.
Developers change chunking strategies, swap embedding models, or update
prompts with no reliable way to measure if quality improved or regressed.
This framework makes RAG evaluation a first-class concern — measurable,
repeatable, and automated.

## Architecture
> Architecture diagram coming soon.

## What This Evaluates
| Metric | What It Measures |
|---|---|
| Faithfulness | Does the answer stick to retrieved context? |
| Answer Relevance | Does the answer address the question? |
| Context Recall | Did retrieval fetch all necessary information? |
| Context Precision | Did retrieval avoid irrelevant chunks? |
| Answer Correctness | Is the answer factually accurate? |

## Tech Stack
| Category | Tool | Purpose |
|---|---|---|
| RAG Framework | LangChain | Pipeline orchestration |
| Evaluation | RAGAS | Core RAG metrics |
| Evaluation | DeepEval | Extended metrics + Pytest integration |
| Vector DB | Qdrant | Document storage and retrieval |
| Experiment Tracking | Weights & Biases | Metric logging and comparison |
| API | FastAPI | Backend REST API |
| Dashboard | Streamlit | Visualization |
| CI/CD | GitHub Actions | Automated eval pipeline |

## Quick Start
1. Clone the repo: git clone <repo-url>
2. Copy env file: cp .env.example .env
3. Fill in API keys in .env
4. Start Qdrant: docker-compose up -d
5. Install deps: pip install -e ".[dev]"
6. Run dashboard: streamlit run dashboard/app.py

## Project Structure
(show folder tree with one-line description per folder)

## Experiments
All evaluation runs are logged to Weights & Biases automatically.
Each run tracks: chunking strategy, embedding model, LLM used,
and all 5 evaluation metric scores. Compare runs visually in W&B.

## Evaluation CI Pipeline
GitHub Actions runs the eval test suite on every push to main.
Future: will auto-run eval regression check and fail the build
if faithfulness drops below threshold.

## Results
> Benchmark results coming soon.

## Author
Dharani Murugaraj
[LinkedIn](https://linkedin.com/in/dharani-murugaraj)
