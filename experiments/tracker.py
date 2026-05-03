"""Weights & Biases experiment tracking.

Plain English: this module sends one evaluation summary (four RAGAS metrics plus
overall) to W&B so you can visually compare retrieval / chunking experiments.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()


def _sanitize_wandb_entity(raw: str | None) -> str:
    """Reject placeholder `.env.example` entities so wandb defaults to logged-in team."""

    if raw is None or raw.strip() == "":
        return ""
    cleaned: str = raw.strip().lower()
    if cleaned.startswith("your_") or "username_here" in cleaned:
        return ""
    return raw.strip()


class ExperimentTracker:
    """
    Logs evaluation results to Weights & Biases.

    WHY W&B: Dashboards compare runs side by side — e.g., fixed vs recursive
    chunking — which is compelling for portfolios and stakeholder reviews.

    Credential handling: Reads ``WANDB_API_KEY``, ``WANDB_PROJECT``, ``WANDB_ENTITY``
    from ``.env`` so tokens never ship in repository code.

    WHY no ``wandb.init`` in ``__init__``: One tracker instance logs many sequential
    runs; each logged evaluation opens and closes its own run to avoid orphaned
    global session state between scripts.
    """

    def __init__(self, project: str | None = None) -> None:
        # ``WANDB_API_KEY`` auto-picked up by wandb SDK from env once present.
        _ = os.getenv("WANDB_API_KEY")  # touch so tools / readers know we expect it
        env_default: str = os.getenv(
            "WANDB_PROJECT",
            "llm-eval-framework",
        ).strip()
        if env_default == "":
            env_default = "llm-eval-framework"

        resolved_project = (
            project.strip()
            if project is not None and project.strip()
            else env_default
        )

        self.project: str = resolved_project.strip() or env_default
        self.entity: str = _sanitize_wandb_entity(os.getenv("WANDB_ENTITY"))

    def log_evaluation(
        self,
        results: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        """
        Logs a complete evaluation run to W&B.

        Args:
            results: Output dict from ``RAGASEvaluator.evaluate`` including detailed rows.
            config: Experiment hyperparameters echoed into W&B.

        Returns:
            Public URL pointing at the logged run page.
        """

        cfg_payload: dict[str, Any] = dict(config or {})
        run = None

        try:
            run = wandb.init(
                entity=self.entity or None,
                project=self.project,
                name=results["experiment_name"],
                config={
                    **cfg_payload,
                    "experiment_name": results["experiment_name"],
                },
            )

            wandb.log(
                {
                    "faithfulness": float(results["faithfulness"]),
                    "answer_relevancy": float(results["answer_relevancy"]),
                    "context_recall": float(results["context_recall"]),
                    "context_precision": float(results["context_precision"]),
                    "overall_score": float(results["overall_score"]),
                    "num_samples": int(results["num_samples"]),
                },
            )

            detail: dict[str, Any] = results.get("detailed_results", {})

            cols: list[str] = [
                "question",
                "answer",
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "context_precision",
            ]

            tbl = wandb.Table(columns=cols)
            df_detail = pd.DataFrame(detail)

            if not df_detail.empty:
                metric_cols_tail: list[str] = cols[2:]
                for _, row_series in df_detail.iterrows():
                    q_cell = (
                        row_series["question"] if "question" in df_detail.columns else ""
                    )
                    a_cell = (
                        row_series["answer"] if "answer" in df_detail.columns else ""
                    )

                    metrics_row: list[float] = []
                    for m_col in metric_cols_tail:
                        if m_col in df_detail.columns:
                            mv = row_series[m_col]
                            metrics_row.append(
                                float(mv)
                                if pd.notna(mv) and mv is not None
                                else 0.0
                            )
                        else:
                            metrics_row.append(0.0)

                    tbl.add_data(
                        "" if pd.isna(q_cell) else str(q_cell),
                        "" if pd.isna(a_cell) else str(a_cell),
                        *metrics_row,
                    )

            wandb.log({"detailed_results": tbl})

            run_url = str(run.url if hasattr(run, "url") else run.get_url())  # type: ignore[union-attr]
            print(f"Results logged to W&B: {run_url}")
            return run_url
        finally:
            if wandb.run is not None:
                wandb.finish()

    def compare_runs(self, run_ids: list[str]) -> None:
        """Prints a W&B URL to compare runs side by side in the UI."""

        if not run_ids:
            print("No run IDs supplied.")
            return

        ent: str = self.entity or "<your_entity>"
        joined: str = ",".join(run_ids)
        compare_url = (
            f"https://wandb.ai/{ent}/{self.project}/runs?"
            f"runIds={joined}"
        )
        print(f"Compare runs at:\n{compare_url}")
