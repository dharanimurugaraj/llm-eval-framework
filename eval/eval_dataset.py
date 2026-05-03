"""Ground-truth Q&A pairs for Phase 3 RAGAS scoring.

Plain English: this file lists “exam questions” and their correct answers about
your financial corpus. Context recall compares retrieved chunks against these
answers, so RAGAS can tell if retrieval missed important evidence.
"""

from __future__ import annotations

EVAL_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "What was the revenue growth percentage year-over-year?",
        "ground_truth": "Revenue increased 23% year-over-year to 4.2 billion dollars.",
    },
    {
        "question": "What is the operating margin?",
        "ground_truth": "Operating margin expanded 150 basis points to 18.3 percent.",
    },
    {
        "question": "What was the free cash flow?",
        "ground_truth": "Free cash flow generation was 890 million dollars.",
    },
    {
        "question": "How much did cloud services revenue grow?",
        "ground_truth": "Cloud Services revenue grew 45% to 1.8 billion dollars.",
    },
    {
        "question": "What is the full-year revenue guidance?",
        "ground_truth": "Management raised full-year guidance to 16.5 billion dollars.",
    },
    {
        "question": "What was the customer retention rate?",
        "ground_truth": "Customer retention rate improved to 94% from 91% last year.",
    },
    {
        "question": "What is the total number of employees?",
        "ground_truth": "Headcount grew to 42000 employees globally.",
    },
    {
        "question": "How much cash does the company have?",
        "ground_truth": "Cash and equivalents increased to 6.2 billion dollars.",
    },
]


def get_eval_questions() -> list[dict[str, str]]:
    """Returns the full evaluation question set."""
    return list(EVAL_QUESTIONS)


def get_eval_questions_only() -> list[str]:
    """Returns just the questions without ground truth."""
    return [q["question"] for q in EVAL_QUESTIONS]
