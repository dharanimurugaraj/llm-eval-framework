"""RAGAS evaluation pipeline for RAG systems.

Uses RAGAS with a Groq LLM judge (via LangChain) and Gemini embeddings so
evaluation matches Phase 2 (same retrieval embedding family, separate judge).

Plain English: this module wraps RAGAS to turn your pipeline's question /
answer / chunks / gold answers into numerical quality scores you can compare
across experiments.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Use RAGAS's LangChain wrappers so RAGAS can call the judge LLM consistently.
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

load_dotenv()

# Column names aligned with REQUIRED_COLS_v1 in ragas.utils — RAGAS renames these
# to user_input / response / retrieved_contexts / reference internally.
_REQUIRED_ITEM_KEYS: frozenset[str] = frozenset(
    {"question", "answer", "contexts", "ground_truth"}
)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS metrics.

    WHY RAGAS: It is the industry standard framework for RAG evaluation.
    Using it signals that you understand production AI evaluation,
    not just model building.

    Metrics computed:
    - Faithfulness: answer grounded in retrieved context?
    - Answer Relevance: answer addresses the question?
    - Context Recall: all relevant info retrieved?
    - Context Precision: only relevant info retrieved?

    LLM Judge: Groq llama-3.1-8b-instant (fast, deterministic at temperature 0).
    Embeddings: Gemini embedding-001 (aligned with Phase 2 vector store).
    """

    def __init__(self) -> None:
        # Keys only from env so credentials never leak into source control.
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY is not set — required for RAGAS LLM-as-judge (Groq)."
            )
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY is not set — required for Gemini judge embeddings "
                "(answer relevance and similar)."
            )

        chat: ChatGroq = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
        )

        embedded: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        # Wrappers expose the API RAGAS expects for batch metric scoring.
        self.evaluator_llm: LangchainLLMWrapper = LangchainLLMWrapper(chat)
        self.evaluator_embeddings: LangchainEmbeddingsWrapper = LangchainEmbeddingsWrapper(
            embedded
        )

        # Module-level singletons — we assign judge + embeddings on each metric
        # once per evaluator instance so all four metrics share the same models.
        self.metrics: list[Any] = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ]

        for metric in self.metrics:
            metric.llm = self.evaluator_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.evaluator_embeddings
            # Groq chat completions forbid ``n`` > 1; RAGAS answer relevancy defaults
            # ``strictness=3`` parallel generations. Dialing down keeps the judge runnable.
            if getattr(metric, "name", "") == "answer_relevancy" and hasattr(
                metric, "strictness"
            ):
                metric.strictness = 1

        # Gentle concurrency avoids Groq burst rate-limit + long tail timeouts during eval.
        self.run_config = RunConfig(
            timeout=240,
            max_workers=8,
            max_retries=12,
            seed=42,
        )

        print("RAGAS Evaluator ready — 4 metrics configured")
        print("Judge LLM: llama-3.1-8b-instant (Groq)")
        print("Embeddings: gemini-embedding-001 (Google)")

    def build_dataset(self, rag_outputs: list[dict[str, Any]]) -> Dataset:
        """
        Converts RAG pipeline outputs to RAGAS Dataset format.

        WHY THIS CONVERSION: RAGAS expects a HuggingFace ``Dataset`` with v1-style
        column names (`question`, `answer`, `contexts`, `ground_truth`). Keeping
        this layer thin lets the rest of the app stay dict-based.

        Args:
            rag_outputs: Rows from ``RAGPipeline.run()`` plus ``ground_truth``.

        Returns:
            A ``datasets.Dataset`` suitable for ``ragas.evaluate``.
        """

        rows: dict[str, list[Any]] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for i, item in enumerate(rag_outputs):
            missing: set[str] = _REQUIRED_ITEM_KEYS - set(item.keys())
            if missing:
                raise ValueError(f"rag_outputs[{i}] missing keys: {sorted(missing)}")

            q: Any = item["question"]
            a: Any = item["answer"]
            cx: Any = item["contexts"]
            gt: Any = item["ground_truth"]

            if not isinstance(q, str) or not q.strip():
                raise ValueError(f"rag_outputs[{i}] has invalid question")
            if not isinstance(a, str):
                raise ValueError(f"rag_outputs[{i}] answer must be a string")
            if not isinstance(gt, str) or not gt.strip():
                raise ValueError(f"rag_outputs[{i}] ground_truth must be non-empty str")
            if not isinstance(cx, list) or not all(isinstance(t, str) for t in cx):
                raise TypeError(f"rag_outputs[{i}] contexts must be list[str]")

            rows["question"].append(q)
            rows["answer"].append(a)
            rows["contexts"].append(cx)
            rows["ground_truth"].append(gt)

        # HuggingFace ``Dataset.from_dict`` is the standard entry point into RAGAS.
        return Dataset.from_dict(rows)

    def evaluate(
        self,
        rag_outputs: list[dict[str, Any]],
        experiment_name: str = "eval_run",
    ) -> dict[str, Any]:
        """
        Runs full RAGAS evaluation and returns scores.

        WHY experiment_name: Every eval run should be named so comparisons in
        W&B / dashboards distinguish chunking strategies and retrieval settings.

        Args:
            rag_outputs: Rows with keys ``question``, ``answer``, ``contexts``, ``ground_truth``.
            experiment_name: Human-readable run label.

        Returns:
            Dict with aggregates, counts, serializable detailed column dict, and naming.
        """

        print(f"Running RAGAS evaluation: {experiment_name}")
        print(f"Evaluating {len(rag_outputs)} question-answer pairs...")

        dataset: Dataset = self.build_dataset(rag_outputs)

        # ``evaluate`` returns ``EvaluationResult`` with per-row scores + means.
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            run_config=self.run_config,
        )
        result_df: pd.DataFrame = result.to_pandas()

        def _numeric_mean(series: pd.Series) -> float:
            s = pd.to_numeric(series, errors="coerce")
            return float(s.mean(skipna=True)) if len(s) else float("nan")

        faithfulness_f: float = _numeric_mean(result_df["faithfulness"])
        answer_relevancy_f: float = _numeric_mean(result_df["answer_relevancy"])
        context_recall_f: float = _numeric_mean(result_df["context_recall"])
        context_precision_f: float = _numeric_mean(result_df["context_precision"])

        aggregates: list[float] = []
        for v in (
            faithfulness_f,
            answer_relevancy_f,
            context_recall_f,
            context_precision_f,
        ):
            if not math.isnan(v):
                aggregates.append(v)

        overall: float = float(sum(aggregates) / len(aggregates)) if aggregates else (
            float("nan")
        )

        # ``object`` + ``where`` avoids JSON-breaking ``NaN`` in saved eval dumps.
        detailed_results = (
            result_df.astype(object).where(pd.notnull(result_df), None).to_dict(
                orient="list"
            )
        )

        out: dict[str, Any] = {
            "experiment_name": experiment_name,
            "faithfulness": faithfulness_f,
            "answer_relevancy": answer_relevancy_f,
            "context_recall": context_recall_f,
            "context_precision": context_precision_f,
            "overall_score": overall,
            "num_samples": len(rag_outputs),
            "detailed_results": detailed_results,
        }

        print()
        print("============================================")
        print(f"RAGAS Evaluation Results: {experiment_name}")
        print("============================================")
        print(f"Faithfulness:       {faithfulness_f:.2f}")
        print(f"Answer Relevancy:   {answer_relevancy_f:.2f}")
        print(f"Context Recall:     {context_recall_f:.2f}")
        print(f"Context Precision:  {context_precision_f:.2f}")
        print("--------------------------------------------")
        print(f"Overall Score:      {overall:.2f}")
        print(f"Samples evaluated:  {len(rag_outputs)}")
        print("============================================")

        return out

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, Any]:
        """
        Evaluates a single QA pair for debugging low scores on one question.

        Args:
            question: User query.
            answer: Model answer.
            contexts: Retrieved chunk texts.
            ground_truth: Reference answer for recall-style metrics.

        Returns:
            Same structure as :meth:`evaluate` for one sample.
        """

        single: list[dict[str, Any]] = [
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        ]
        return self.evaluate(single, experiment_name="single_debug")
