"""
Main evaluation runner.

This script:
1. Loads the RAG pipeline (already has data in Qdrant)
2. Runs all eval questions through the RAG pipeline
3. Evaluates outputs with RAGAS
4. Logs results to Weights & Biases
5. Saves results to data/processed/eval_results.json

Run with:

    python eval/run_evaluation.py

Plain English: this ties retrieval + Groq answering + RAGAS judging into one batch
report you can archive and plot in W&B.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow ``python eval/run_evaluation.py`` without an editable package install:
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.rag_pipeline import RAGPipeline  # noqa: E402
from eval.eval_dataset import EVAL_QUESTIONS  # noqa: E402
from eval.ragas_eval import RAGASEvaluator  # noqa: E402
from experiments.tracker import ExperimentTracker  # noqa: E402
from ingestion.vector_store import VectorStoreManager  # noqa: E402

load_dotenv()


def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy-ish scalars to native Python types for ``json.dump``."""

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar types
        return obj.item()
    return obj


def run_full_evaluation(
    experiment_name: str = "baseline_recursive_gemini_groq",
    chunking_strategy: str = "recursive",
    embedding_model: str = "gemini",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Runs complete evaluation pipeline.

    Parameters allow naming different experiments:

    .. note::
       ``chunking_strategy`` reflects how ingestion was originally run —
       retrieval here uses whatever vectors already live in Qdrant; we only
       pass this forward for reproducible labeling in configs and dashboards.

    Returns:
        Full results dict suitable for persistence and logging.
    """

    print(f"Starting evaluation: {experiment_name}")
    print(
        f"Config: chunking={chunking_strategy}, embeddings={embedding_model}, "
        f"top_k={top_k}"
    )
    print("=" * 60)

    # Step 1: Load RAG pipeline
    print("Step 1/4: Initializing RAG pipeline...")
    vector_store: VectorStoreManager = VectorStoreManager(
        embedding_model=embedding_model
    )

    rag_pipeline: RAGPipeline = RAGPipeline(
        vector_store=vector_store,
        model="groq",
        top_k=top_k,
    )

    # --- Step 2: Run eval questions ---
    print("Step 2/4: Running questions through RAG pipeline...")
    rag_outputs: list[dict[str, Any]] = []
    total: int = len(EVAL_QUESTIONS)

    for i, row in enumerate(EVAL_QUESTIONS):
        qtxt: str = row["question"]
        preview = qtxt if len(qtxt) <= 50 else qtxt[:50] + "..."
        print(f"  Q{i + 1}/{total}: {preview}")
        rag_out = rag_pipeline.run(qtxt)
        rag_out["ground_truth"] = row["ground_truth"]
        rag_outputs.append(rag_out)

    # --- Step 3: Run RAGAS evaluation ---
    print("Step 3/4: Running RAGAS evaluation...")
    evaluator: RAGASEvaluator = RAGASEvaluator()
    results: dict[str, Any] = evaluator.evaluate(
        rag_outputs,
        experiment_name=experiment_name,
    )

    # --- Step 4: Save JSON + optional W&B ---
    print("Step 4/4: Saving results...")
    tracker_config: dict[str, Any] = {
        "experiment_name": experiment_name,
        "chunking_strategy": chunking_strategy,
        "embedding_model": embedding_model,
        "top_k": top_k,
    }

    wandb_ok: bool = bool(os.getenv("WANDB_API_KEY"))
    if wandb_ok:
        try:
            tracker: ExperimentTracker = ExperimentTracker()
            tracker.log_evaluation(results, config=tracker_config)
        except Exception as exc:  # noqa: BLE001 — user prefers visible telemetry failures
            print(f"WARN: Weights & Biases logging skipped due to error: {exc}")
    else:
        print("WANDB_API_KEY not set — skipping W&B run.")

    out_path: Path = _ROOT / "data" / "processed" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = _sanitize_for_json(results)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    RESULTS_MAIN: dict[str, Any] = run_full_evaluation()
    print("\nEvaluation complete!")
    print(f"Overall Score: {RESULTS_MAIN['overall_score']:.3f}")
