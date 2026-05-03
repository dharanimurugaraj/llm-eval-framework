"""Streamlit Dashboard for LLM Evaluation Framework."""

import json
import os
import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import wandb
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

WANDB_ENTITY = "dharani-vyrenzo-vyrenzo-in"
WANDB_PROJECT = "llm-eval-framework"
RESULTS_PATH = "data/processed/eval_results.json"

METRIC_COLORS = {
    "faithfulness": "#2ecc71",
    "answer_relevancy": "#3498db",
    "context_recall": "#e67e22",
    "context_precision": "#9b59b6",
    "overall_score": "#1abc9c",
}

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
    "overall_score": "Overall Score",
}

# ===================================================
# SECTION 2: HELPER FUNCTIONS
# ===================================================

def load_latest_results() -> dict | None:
    """
    Loads most recent eval results from eval_results.json.
    Returns None if file missing or JSON invalid.
    """
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=300)
def load_wandb_runs() -> list[dict]:
    """
    Fetches all runs from W&B project.
    Returns list of dicts with metric scores.
    Cached for 5 minutes to avoid slow API calls.
    Returns empty list on any failure.
    """
    try:
        api = wandb.Api()
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
        result = []
        for run in runs:
            summary = run.summary
            result.append({
                "name": run.name,
                "faithfulness": summary.get("faithfulness", 0),
                "answer_relevancy": summary.get("answer_relevancy", 0),
                "context_recall": summary.get("context_recall", 0),
                "context_precision": summary.get("context_precision", 0),
                "overall_score": summary.get("overall_score", 0),
                "num_samples": summary.get("num_samples", 0),
                "created_at": run.created_at,
            })
        return sorted(result, key=lambda x: x["overall_score"], reverse=True)
    except Exception:
        return []

def score_to_label(score: float) -> str:
    """Converts 0-1 score to human readable quality label."""
    if score >= 0.85: return "🟢 Excellent"
    if score >= 0.70: return "🟡 Good"
    if score >= 0.50: return "🟠 Needs Improvement"
    return "🔴 Poor"

def score_to_delta_color(score: float) -> str:
    """Returns normal/inverse for st.metric delta_color."""
    return "normal" if score >= 0.70 else "inverse"

def make_radar_chart(metrics: dict, title: str) -> go.Figure:
    """
    Creates a plotly radar chart for 4 RAGAS metrics.
    
    Args:
        metrics: dict with faithfulness, answer_relevancy,
                 context_recall, context_precision keys
        title: chart title string
    """
    categories = [
        "Faithfulness", "Answer Relevancy",
        "Context Recall", "Context Precision"
    ]
    values = [
        metrics.get("faithfulness", 0),
        metrics.get("answer_relevancy", 0),
        metrics.get("context_recall", 0),
        metrics.get("context_precision", 0),
    ]
    # Close the polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(26, 188, 156, 0.2)",
        line=dict(color="#1abc9c", width=2),
        name="Scores"
    ))
    # Add threshold ring at 0.7
    threshold = [0.7] * len(categories_closed)
    fig.add_trace(go.Scatterpolar(
        r=threshold,
        theta=categories_closed,
        line=dict(color="#e74c3c", width=1, dash="dash"),
        name="Quality Threshold (0.7)",
        fill=None,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title,
        height=400,
    )
    return fig

def make_comparison_bar_chart(
    runs: list[dict], 
    run_a_name: str, 
    run_b_name: str
) -> go.Figure:
    """
    Creates grouped bar chart comparing two experiment runs.
    """
    metrics = ["faithfulness", "answer_relevancy", 
               "context_recall", "context_precision"]
    metric_display = ["Faithfulness", "Answer Relevancy",
                      "Context Recall", "Context Precision"]
    
    run_a = next((r for r in runs if r["name"] == run_a_name), {})
    run_b = next((r for r in runs if r["name"] == run_b_name), {})
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=run_a_name,
        x=metric_display,
        y=[run_a.get(m, 0) for m in metrics],
        marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        name=run_b_name,
        x=metric_display,
        y=[run_b.get(m, 0) for m in metrics],
        marker_color="#e67e22",
    ))
    # Quality threshold line
    fig.add_hline(
        y=0.7, 
        line_dash="dash", 
        line_color="#e74c3c",
        annotation_text="Quality Threshold",
        annotation_position="right",
    )
    fig.update_layout(
        barmode="group",
        title=f"A/B Comparison: {run_a_name} vs {run_b_name}",
        yaxis=dict(range=[0, 1.1], title="Score"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

def make_trends_chart(runs: list[dict]) -> go.Figure:
    """
    Creates line chart showing metric trends across all runs.
    X axis: experiment name
    Y axis: metric scores
    One colored line per metric.
    """
    metrics = ["faithfulness", "answer_relevancy",
               "context_recall", "context_precision", "overall_score"]
    
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=[r["name"] for r in runs],
            y=[r.get(metric, 0) for r in runs],
            mode="lines+markers",
            name=METRIC_LABELS[metric],
            line=dict(color=METRIC_COLORS[metric], width=2),
            marker=dict(size=8),
        ))
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Quality Threshold (0.7)",
        annotation_position="right",
    )
    fig.update_layout(
        title="Metric Trends Across Experiments",
        xaxis_title="Experiment",
        yaxis=dict(range=[0, 1.1], title="Score"),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

# ===================================================
# SECTION 3: PAGE FUNCTIONS
# ===================================================

def page_overview():
    """Renders the Overview page with latest eval results."""

    st.title("📊 LLM Eval Framework")
    st.caption(
        "A production-grade RAG evaluation framework — "
        "benchmark chunking strategies, embedding models, "
        "and LLM outputs with RAGAS and Weights & Biases."
    )
    st.divider()

    results = load_latest_results()

    if results is None:
        st.info(
            "No evaluation results found yet. "
            "Go to **🔬 Run Evaluation** to run your first experiment."
        )
        return

    # Experiment header
    st.markdown(f"### Latest Run: `{results['experiment_name']}`")
    st.caption(f"Samples evaluated: {results.get('num_samples', 0)}")

    # Metric cards row
    st.markdown("#### Metric Scores")
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_display = [
        (col1, "faithfulness", "Faithfulness"),
        (col2, "answer_relevancy", "Answer Relevancy"),
        (col3, "context_recall", "Context Recall"),
        (col4, "context_precision", "Context Precision"),
        (col5, "overall_score", "Overall Score"),
    ]
    for col, key, label in metrics_display:
        score = results.get(key, 0)
        with col:
            st.metric(
                label=label,
                value=f"{score:.3f}",
                delta=score_to_label(score),
                delta_color=score_to_delta_color(score),
            )

    st.divider()

    # Two column layout: radar + findings
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### Metric Radar")
        fig = make_radar_chart(results, 
              f"RAGAS Scores — {results['experiment_name']}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Red dashed line = quality threshold (0.7). "
            "Scores above this are considered production-ready."
        )

    with col_right:
        st.markdown("#### 🔍 Key Findings")
        st.markdown("""
**Context Precision = 1.0 across all runs**
Every retrieved chunk was relevant — zero retrieval noise.
Strong signal that Gemini embedding-001 quality is excellent.

---

**Semantic chunking wins overall (0.84)**
Context recall jumped from 0.57 → 0.85 vs fixed-size.
Grouping sentences by meaning keeps related facts together.

---

**Faithfulness bottleneck in recursive chunking**
Recursive scored 0.51 faithfulness vs 0.75 for semantic.
Paragraph boundary cuts may fragment supporting evidence.

---

**Answer Relevancy stable at 0.75**
Consistent across all 3 strategies — a property of the
LLM judge, not the chunking. Groq llama-3.1-8b-instant
tends toward terse answers which reduces this score.
        """)

    st.divider()

    # Per-question breakdown
    st.markdown("#### Per-Question Results")
    st.caption("Click column headers to sort by any metric.")

    detailed = results.get("detailed_results", {})
    if detailed and "question" in detailed:
        df = pd.DataFrame({
            "Question": [
                q[:60] + "..." if len(q) > 60 else q
                for q in detailed.get("question", [])
            ],
            "Answer": [
                a[:80] + "..." if len(a) > 80 else a
                for a in detailed.get("answer", [])
            ],
            "Faithfulness": [
                round(v, 3) 
                for v in detailed.get("faithfulness", [])
            ],
            "Answer Relevancy": [
                round(v, 3) 
                for v in detailed.get("answer_relevancy", [])
            ],
            "Context Recall": [
                round(v, 3) 
                for v in detailed.get("context_recall", [])
            ],
            "Context Precision": [
                round(v, 3) 
                for v in detailed.get("context_precision", [])
            ],
        })
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Detailed per-question results not available.")


def page_run_evaluation():
    """Renders the Run Evaluation page with config form."""

    st.title("🔬 Run New Evaluation")
    st.caption("Configure and trigger a new RAG evaluation experiment.")
    st.warning(
        "⏱️ Running evaluation takes 2-4 minutes and makes "
        "API calls to Groq (LLM) and Google (embeddings)."
    )
    st.divider()

    # Config form
    with st.form("eval_form"):
        st.markdown("#### Experiment Configuration")

        experiment_name = st.text_input(
            "Experiment Name",
            value="my_experiment_001",
            help="Unique name for this run. Will appear in W&B dashboard."
        )

        col1, col2 = st.columns(2)
        with col1:
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ["recursive", "fixed_size", "semantic"],
                help=(
                    "recursive: respects sentence/paragraph boundaries\n"
                    "fixed_size: splits by character count\n"
                    "semantic: splits by meaning (slowest, best quality)"
                )
            )
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=200, max_value=2000,
                value=1000, step=100,
            )
        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0, max_value=400,
                value=200, step=50,
            )
            top_k = st.slider(
                "Top K Retrieval",
                min_value=1, max_value=10,
                value=5, step=1,
                help="Number of chunks retrieved per query."
            )

        log_to_wandb = st.checkbox(
            "Log results to Weights & Biases",
            value=True
        )

        submitted = st.form_submit_button(
            "▶ Run Evaluation", 
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not experiment_name.strip():
            st.error("Experiment name cannot be empty.")
            return
        try:
            with st.spinner(
                f"Running evaluation: {experiment_name}... "
                "this takes 2-4 minutes"
            ):
                import sys
                sys.path.insert(0, ".")
                from eval.run_evaluation import run_full_evaluation
                results = run_full_evaluation(
                    experiment_name=experiment_name.strip(),
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
            st.success(
                f"✅ Evaluation complete! "
                f"Overall Score: {results['overall_score']:.3f}"
            )
            st.balloons()

            # Show results
            st.markdown("#### Results")
            col1, col2, col3, col4, col5 = st.columns(5)
            for col, key, label in [
                (col1, "faithfulness", "Faithfulness"),
                (col2, "answer_relevancy", "Answer Relevancy"),
                (col3, "context_recall", "Context Recall"),
                (col4, "context_precision", "Context Precision"),
                (col5, "overall_score", "Overall"),
            ]:
                with col:
                    score = results.get(key, 0)
                    st.metric(label, f"{score:.3f}",
                             score_to_label(score))

            if log_to_wandb:
                st.info(
                    "Results logged to W&B. "
                    "[View Dashboard]"
                    "(https://wandb.ai/dharani-vyrenzo-vyrenzo-in"
                    "/llm-eval-framework)"
                )
            # Clear cache so history page shows new run
            load_wandb_runs.clear()

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.exception(e)


def page_experiment_history():
    """Renders Experiment History page with all W&B runs."""

    st.title("📈 Experiment History")
    st.caption(
        "All evaluation runs logged to Weights & Biases. "
        "Sorted by overall score (best first)."
    )
    st.markdown(
        "[🔗 Open W&B Dashboard](https://wandb.ai/"
        "dharani-vyrenzo-vyrenzo-in/llm-eval-framework)"
    )
    st.divider()

    with st.spinner("Loading runs from Weights & Biases..."):
        runs = load_wandb_runs()

    if not runs:
        st.info(
            "No runs found in W&B. "
            "Run an evaluation first on the **🔬 Run Evaluation** page."
        )
        return

    # Summary table
    st.markdown("#### All Runs")
    df = pd.DataFrame(runs)
    df_display = df[[
        "name", "overall_score", "faithfulness",
        "answer_relevancy", "context_recall",
        "context_precision", "num_samples"
    ]].copy()
    df_display.columns = [
        "Experiment", "Overall", "Faithfulness",
        "Answer Relevancy", "Context Recall",
        "Context Precision", "Samples"
    ]
    # Round scores
    score_cols = [
        "Overall", "Faithfulness", "Answer Relevancy",
        "Context Recall", "Context Precision"
    ]
    for col in score_cols:
        df_display[col] = df_display[col].round(3)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Trends chart
    st.markdown("#### Metric Trends")
    # Reverse for chronological order in chart
    runs_chrono = list(reversed(runs))
    fig = make_trends_chart(runs_chrono)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Red dashed line = quality threshold (0.7). "
        "Each point is one evaluation run."
    )

    st.divider()

    # Best run callout
    best = runs[0]  # already sorted by overall_score desc
    st.success(
        f"🏆 Best run: **{best['name']}** "
        f"with overall score **{best['overall_score']:.3f}**"
    )
    col1, col2, col3, col4 = st.columns(4)
    for col, key, label in [
        (col1, "faithfulness", "Faithfulness"),
        (col2, "answer_relevancy", "Answer Relevancy"),
        (col3, "context_recall", "Context Recall"),
        (col4, "context_precision", "Context Precision"),
    ]:
        with col:
            score = best.get(key, 0)
            st.metric(label, f"{score:.3f}", score_to_label(score))


def page_ab_comparison():
    """Renders A/B Comparison page for two experiment runs."""

    st.title("⚖️ A/B Comparison")
    st.caption("Compare any two experiments side by side.")
    st.divider()

    with st.spinner("Loading runs..."):
        runs = load_wandb_runs()

    if len(runs) < 2:
        st.info(
            "Need at least 2 experiment runs to compare. "
            "Go to **🔬 Run Evaluation** to run another experiment."
        )
        return

    run_names = [r["name"] for r in runs]

    col1, col2 = st.columns(2)
    with col1:
        run_a_name = st.selectbox(
            "Experiment A", run_names, index=0
        )
    with col2:
        run_b_name = st.selectbox(
            "Experiment B", run_names, 
            index=min(1, len(run_names)-1)
        )

    if run_a_name == run_b_name:
        st.warning("Please select two different experiments.")
        return

    run_a = next((r for r in runs if r["name"] == run_a_name), {})
    run_b = next((r for r in runs if r["name"] == run_b_name), {})

    st.divider()

    # Side by side metric comparison
    st.markdown("#### Metric Comparison")
    metrics = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
    ]

    for key, label in metrics:
        score_a = run_a.get(key, 0)
        score_b = run_b.get(key, 0)
        delta = score_a - score_b
        winner = run_a_name if score_a > score_b else run_b_name
        tied = abs(delta) < 0.001

        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 0.5, 1.5, 2])
        with col1:
            color_a = "#2ecc71" if score_a >= score_b else "#e74c3c"
            st.markdown(
                f"<p style='text-align:right; color:{color_a}; "
                f"font-size:1.2em; font-weight:bold'>{score_a:.3f}</p>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<p style='text-align:right'>{run_a_name}</p>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"<p style='text-align:center'><b>{label}</b></p>",
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(run_b_name)
        with col5:
            color_b = "#2ecc71" if score_b >= score_a else "#e74c3c"
            st.markdown(
                f"<p style='color:{color_b}; "
                f"font-size:1.2em; font-weight:bold'>{score_b:.3f}</p>",
                unsafe_allow_html=True
            )

    st.divider()

    # Grouped bar chart
    st.markdown("#### Visual Comparison")
    fig = make_comparison_bar_chart(runs, run_a_name, run_b_name)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Verdict
    st.markdown("#### 🏆 Verdict")
    metrics_keys = [
        "faithfulness", "answer_relevancy",
        "context_recall", "context_precision"
    ]
    a_wins = sum(
        1 for m in metrics_keys 
        if run_a.get(m, 0) > run_b.get(m, 0)
    )
    b_wins = sum(
        1 for m in metrics_keys 
        if run_b.get(m, 0) > run_a.get(m, 0)
    )

    if a_wins > b_wins:
        st.success(
            f"**{run_a_name}** wins {a_wins}/4 metrics overall."
        )
    elif b_wins > a_wins:
        st.success(
            f"**{run_b_name}** wins {b_wins}/4 metrics overall."
        )
    else:
        st.info("Both experiments tied — 2 wins each.")

    # Specific insights for large differences
    st.markdown("**Metric Insights:**")
    for key, label in metrics:
        score_a = run_a.get(key, 0)
        score_b = run_b.get(key, 0)
        delta = abs(score_a - score_b)
        if delta >= 0.05:
            winner = run_a_name if score_a > score_b else run_b_name
            loser = run_b_name if score_a > score_b else run_a_name
            st.markdown(
                f"- **{label}**: `{winner}` scores "
                f"{delta:.2f} higher than `{loser}`"
            )

# ===================================================
# SECTION 4: MAIN APP + SIDEBAR + ROUTING
# ===================================================

def main():
    """Main app entry point — sidebar + page routing."""

    # Sidebar
    with st.sidebar:
        st.title("📊 LLM Eval Framework")
        st.caption("RAG Pipeline Evaluation")
        st.divider()

        page = st.selectbox(
            "Navigate",
            [
                "🏠 Overview",
                "🔬 Run Evaluation",
                "📈 Experiment History",
                "⚖️ A/B Comparison",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Current Stack**")
        st.markdown("🔢 Embeddings: Gemini embedding-001")
        st.markdown("🤖 LLM: Groq llama-3.1-8b-instant")
        st.markdown("🗄️ Vector DB: Qdrant Cloud")
        st.markdown("📏 Eval Framework: RAGAS")
        st.divider()
        st.markdown(
            "[🔗 W&B Dashboard](https://wandb.ai/"
            "dharani-vyrenzo-vyrenzo-in/llm-eval-framework)"
        )
        st.markdown(
            "[💻 GitHub](https://github.com/dharanimurugaraj"
            "/llm-eval-framework)"
        )

    # Page routing
    if page == "🏠 Overview":
        page_overview()
    elif page == "🔬 Run Evaluation":
        page_run_evaluation()
    elif page == "📈 Experiment History":
        page_experiment_history()
    elif page == "⚖️ A/B Comparison":
        page_ab_comparison()

if __name__ == "__main__":
    main()
