"""Tests for RAGAS evaluation, curated eval datasets, and W&B tracking stubs."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.eval_dataset import EVAL_QUESTIONS, get_eval_questions, get_eval_questions_only
from eval.ragas_eval import RAGASEvaluator


def test_ragas_evaluator_builds_dataset() -> None:
    """Ensures rag_outputs reshape into HF columns RAGAS will rename downstream."""

    sample: list[dict[str, object]] = [
        {
            "question": "Q1?",
            "answer": "A1",
            "contexts": ["c1", "c2"],
            "ground_truth": "G1",
        },
        {
            "question": "Q2?",
            "answer": "A2",
            "contexts": ["cx"],
            "ground_truth": "G2",
        },
        {
            "question": "Q3?",
            "answer": "A3",
            "contexts": [],
            "ground_truth": "G3",
        },
    ]

    with patch.object(RAGASEvaluator, "__init__", lambda self: None):
        evaluator_under_test: RAGASEvaluator = RAGASEvaluator()
        dataset = evaluator_under_test.build_dataset(sample)

    assert dataset.num_rows == 3
    feature_names = set(dataset.features.keys())
    assert feature_names == {"question", "answer", "contexts", "ground_truth"}


def test_eval_dataset_has_required_keys() -> None:
    """Sanity-check ground-truth questions before paying for API calls."""

    assert len(EVAL_QUESTIONS) >= 5

    for i, bundle in enumerate(EVAL_QUESTIONS):
        assert isinstance(bundle, dict)
        assert "question" in bundle
        assert "ground_truth" in bundle
        assert isinstance(bundle["question"], str)
        assert isinstance(bundle["ground_truth"], str)
        assert bundle["question"].strip() != ""
        assert bundle["ground_truth"].strip() != ""


def test_get_eval_questions_returns_list() -> None:
    qs = get_eval_questions()
    assert isinstance(qs, list)
    assert qs
    assert all(isinstance(x, dict) for x in qs)

    qs_only = get_eval_questions_only()
    assert isinstance(qs_only, list)
    assert len(qs_only) == len(EVAL_QUESTIONS)


@patch("experiments.tracker.wandb.init", MagicMock())
def test_experiment_tracker_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracker should hydrate project from env or our safe default."""

    from experiments.tracker import ExperimentTracker

    monkeypatch.setenv("WANDB_PROJECT", "")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    tr_default = ExperimentTracker()
    assert tr_default.project == "llm-eval-framework"

    monkeypatch.setenv("WANDB_PROJECT", "custom-proj-from-env")

    tr_env = ExperimentTracker()
    assert tr_env.project == "custom-proj-from-env"

    tr_override = ExperimentTracker(project="override-name")
    assert tr_override.project == "override-name"
