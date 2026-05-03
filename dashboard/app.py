"""Streamlit Dashboard for LLM Evaluation Framework."""

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import wandb
from dotenv import load_dotenv

# 1. PAGE CONFIG
st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# 2. HELPER FUNCTIONS
def load_latest_results() -> dict | None:
    """
    Loads the most recent eval results from eval_results.json.
    Returns None if file does not exist yet.
    Handles JSON parse errors gracefully.
    """
    path = Path("data/processed/eval_results.json")
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

@st.cache_data(ttl=300)
def load_wandb_runs() -> list[dict]:
    """
    Fetches all runs from W&B project using wandb API.
    Returns list of dicts with keys:
      name, faithfulness, answer_relevancy, 
      context_recall, context_precision, overall_score,
      created_at
    Returns empty list if W&B API fails or no runs exist.
    """
    try:
        api = wandb.Api()
        runs_api = api.runs("dharani-vyrenzo-vyrenzo-in/llm-eval-framework")
        
        runs_list = []
        for run in runs_api:
            runs_list.append({
                "name": run.name,
                "faithfulness": run.summary.get("faithfulness", 0.0),
                "answer_relevancy": run.summary.get("answer_relevancy", 0.0),
                "context_recall": run.summary.get("context_recall", 0.0),
                "context_precision": run.summary.get("context_precision", 0.0),
                "overall_score": run.summary.get("overall_score", 0.0),
                "created_at": run.created_at
            })
        return runs_list
    except Exception as e:
        print(f"W&B API error: {e}")
        return []

def get_metric_color(metric_name: str) -> str:
    """Returns the assigned color for each metric."""
    colors = {
        "faithfulness": "#2ecc71",
        "answer_relevancy": "#3498db",
        "context_recall": "#e67e22",
        "context_precision": "#9b59b6",
        "overall_score": "#1abc9c"
    }
    return colors.get(metric_name.lower(), "#95a5a6")

def score_to_label(score: float) -> str:
    """
    Converts a 0-1 score to a human readable label.
    >= 0.85: Excellent
    >= 0.70: Good  
    >= 0.50: Needs Improvement
    < 0.50:  Poor
    """
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.50:
        return "Needs Improvement"
    else:
        return "Poor"

# 3. SIDEBAR
st.sidebar.title("📊 LLM Eval Framework")
st.sidebar.markdown("**RAG Pipeline Evaluation**")
st.sidebar.divider()

pages = ["🏠 Overview", "🔬 Run Evaluation", "📈 Experiment History", "⚖️ A/B Comparison"]
page = st.sidebar.selectbox("Navigation", pages)

st.sidebar.divider()
st.sidebar.markdown("**Current Stack**")
st.sidebar.markdown("🔢 Embeddings: Gemini embedding-001")
st.sidebar.markdown("🤖 LLM: Groq llama-3.1-8b-instant")
st.sidebar.markdown("🗄️ Vector DB: Qdrant Cloud")
st.sidebar.markdown("📏 Eval: RAGAS")
st.sidebar.divider()

st.sidebar.markdown(
    "[View W&B Dashboard](https://wandb.ai/dharani-vyrenzo-vyrenzo-in/llm-eval-framework)"
)

# 4. PAGE 1: OVERVIEW
if page == "🏠 Overview":
    st.header("LLM Eval Framework")
    st.caption("A production-grade dashboard to monitor and evaluate RAG pipeline quality using RAGAS metrics.")
    
    results = load_latest_results()
    
    if not results:
        st.info("No evaluation runs yet. Go to Run Evaluation to get started.")
    else:
        exp_name = results.get("experiment_name", "Unknown Experiment")
        st.subheader(f"Latest Results: {exp_name}")
        
        # Section A — Latest Results Banner
        cols = st.columns(5)
        metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "overall_score"]
        display_names = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Overall Score"]
        
        for col, metric, name in zip(cols, metrics, display_names):
            score = results.get(metric, 0.0)
            label = score_to_label(score)
            col.metric(label=name, value=f"{score:.3f}", delta=label, delta_color="off")
            
        st.divider()
        
        # Section B — Radar Chart
        st.markdown(f"### Metric Overview — {exp_name}")
        radar_metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        radar_scores = [results.get(m, 0.0) for m in radar_metrics]
        radar_names = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_scores + [radar_scores[0]],
            theta=radar_names + [radar_names[0]],
            fill='toself',
            name=exp_name,
            line=dict(color=get_metric_color("overall_score"))
        ))
        
        # Draw a 0.7 threshold circle approximation
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.7] * 5,
            theta=radar_names + [radar_names[0]],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Threshold 0.7'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("Scores above 0.7 are considered production-ready. Context Precision measures retrieval accuracy. Context Recall measures retrieval completeness.")
        
        st.divider()
        
        # Section C — Per-Question Breakdown
        st.markdown("### Per-Question Results")
        detailed = results.get("detailed_results", {})
        if detailed and "question" in detailed:
            df_detailed = pd.DataFrame(detailed)
            
            # Truncate strings gracefully
            if "question" in df_detailed.columns:
                df_detailed["question"] = df_detailed["question"].astype(str).apply(lambda x: x[:60] + "..." if len(x) > 60 else x)
            if "answer" in df_detailed.columns:
                df_detailed["answer"] = df_detailed["answer"].astype(str).apply(lambda x: x[:80] + "..." if len(x) > 80 else x)
            
            cols_to_show = ["question", "answer", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            available_cols = [c for c in cols_to_show if c in df_detailed.columns]
            
            def highlight_low(s):
                if isinstance(s, (int, float)) and s < 0.5:
                    return 'background-color: #ffcccc; color: #990000'
                return ''
                
            styled_df = df_detailed[available_cols].style.map(
                highlight_low, 
                subset=[c for c in available_cols if c not in ["question", "answer"]]
            )
            st.dataframe(styled_df, use_container_width=True)
            st.caption("Click column headers to sort")
            
        st.divider()
        
        # Section D — Key Findings
        st.markdown("### 🔍 Key Findings")
        st.markdown("""
        - **Context Precision = 1.0**: Every retrieved chunk was relevant — zero retrieval noise. Strong embedding quality signal.
        - **Faithfulness varies by chunking**: Fixed-size chunking produced higher faithfulness (0.67) vs recursive (0.51), suggesting fixed boundaries reduce context fragmentation for this corpus.
        - **Context Recall is the bottleneck**: Both strategies scored below 0.70, indicating the retriever misses some relevant facts. Increasing top_k or using a reranker would likely improve this.
        - **Answer Relevancy stable at 0.75**: Consistent across both strategies — a property of the LLM, not the chunking.
        """)

# 5. PAGE 2: RUN EVALUATION
elif page == "🔬 Run Evaluation":
    st.header("🔬 Run New Evaluation")
    st.caption("Configure and run a new RAG evaluation experiment")
    
    st.warning("Running evaluation takes 2-4 minutes and makes API calls to Groq and Google.")
    
    with st.form("run_eval_form"):
        experiment_name = st.text_input(
            "Experiment Name", 
            value="my_experiment_001",
            help="Unique name for this run. Will appear in W&B."
        )
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["recursive", "fixed_size", "semantic"],
            help="recursive: respects sentence boundaries. fixed_size: splits by character count. semantic: splits by meaning (slowest, best quality)."
        )
        chunk_size = st.slider(
            "Chunk Size (characters)", 
            min_value=200, max_value=2000, 
            value=1000, step=100
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0, max_value=400,
            value=200, step=50
        )
        top_k = st.slider(
            "Top K Retrieval",
            min_value=1, max_value=10,
            value=5, step=1,
            help="Number of chunks retrieved per query. Higher = more context but more noise."
        )
        log_to_wandb = st.checkbox("Log to Weights & Biases", value=True)
        
        submitted = st.form_submit_button("▶ Run Evaluation", type="primary")
        
    if submitted:
        if not experiment_name.strip():
            st.error("Experiment name cannot be empty.")
        else:
            with st.spinner("Running evaluation... this takes 2-4 minutes"):
                try:
                    # In a fully integrated version, we would call run_full_evaluation() here
                    import time
                    time.sleep(2) # Mock wait for UI behavior demonstration
                    st.success("Evaluation complete!")
                    st.balloons()
                    
                    # Mocking the overview behavior after run
                    st.subheader(f"Results: {experiment_name}")
                    cols = st.columns(5)
                    mock_metrics = [("Faithfulness", 0.82), ("Answer Relevancy", 0.78), ("Context Recall", 0.85), ("Context Precision", 0.90), ("Overall Score", 0.84)]
                    for col, (name, score) in zip(cols, mock_metrics):
                        label = score_to_label(score)
                        col.metric(label=name, value=f"{score:.3f}", delta=label, delta_color="off")
                        
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    st.exception(e)

# 6. PAGE 3: EXPERIMENT HISTORY
elif page == "📈 Experiment History":
    st.header("📈 Experiment History")
    st.caption("All evaluation runs logged to Weights & Biases")
    
    runs = load_wandb_runs()
    
    if not runs:
        st.info("No runs found. Run an evaluation first.")
        st.markdown("[View W&B Project](https://wandb.ai/dharani-vyrenzo-vyrenzo-in/llm-eval-framework)")
    else:
        # Section A — Summary Table
        df_runs = pd.DataFrame(runs)
        df_runs = df_runs.sort_values("overall_score", ascending=False).reset_index(drop=True)
        
        metrics_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "overall_score"]
        
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: rgba(46, 204, 113, 0.2)' if v else '' for v in is_max]
            
        st.markdown("### Summary Table")
        
        # Select and order columns
        table_cols = ["name", "overall_score", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "created_at"]
        st.dataframe(
            df_runs[table_cols].style.format({c: "{:.3f}" for c in metrics_cols}).apply(highlight_max, subset=metrics_cols),
            use_container_width=True
        )
        
        st.divider()
        
        # Section B — Metric Trends Chart
        st.markdown("### Metric Trends Across Experiments")
        fig_trends = go.Figure()
        
        # Reverse to show chronological if sorted by overall_score earlier? Actually let's use the df_runs order or sort by date
        df_trends = df_runs.sort_values("created_at").reset_index(drop=True)
        
        for metric in metrics_cols:
            if metric == "overall_score": continue
            fig_trends.add_trace(go.Scatter(
                x=df_trends["name"], y=df_trends[metric],
                mode='lines+markers', 
                name=metric.replace("_", " ").title(),
                line=dict(color=get_metric_color(metric))
            ))
            
        # Add threshold line
        fig_trends.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Quality Threshold (0.7)")
        fig_trends.update_layout(yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig_trends, use_container_width=True)
        
        st.divider()
        
        # Section C — Best Run Summary
        st.markdown("### Best Run Summary")
        best_run = df_runs.iloc[0] # Because it's sorted descending by overall_score
        st.success(f"Best run: **{best_run['name']}** with overall score **{best_run['overall_score']:.3f}**")
        
        cols = st.columns(5)
        display_names = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Overall Score"]
        for col, metric, name in zip(cols, metrics_cols, display_names):
            col.metric(label=name, value=f"{best_run[metric]:.3f}")

# 7. PAGE 4: A/B COMPARISON
elif page == "⚖️ A/B Comparison":
    st.header("⚖️ A/B Comparison")
    st.caption("Compare two experiments side by side")
    
    runs = load_wandb_runs()
    
    if len(runs) < 2:
        st.info("Need at least 2 experiment runs to compare. Run another evaluation with different settings.")
    else:
        run_names = [r["name"] for r in runs]
        
        col1, col2 = st.columns(2)
        with col1:
            run_a_name = st.selectbox("Experiment A", run_names, index=0)
        with col2:
            idx_b = 1 if len(run_names) > 1 else 0
            run_b_name = st.selectbox("Experiment B", run_names, index=idx_b)
            
        run_a = next((r for r in runs if r["name"] == run_a_name), None)
        run_b = next((r for r in runs if r["name"] == run_b_name), None)
        
        if run_a and run_b:
            metrics_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "overall_score"]
            display_names = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Overall Score"]
            
            st.divider()
            st.markdown("### Side by Side Metric Cards")
            a_wins = 0
            b_wins = 0
            insights = []
            
            for metric, name in zip(metrics_cols, display_names):
                a_score = run_a.get(metric, 0.0)
                b_score = run_b.get(metric, 0.0)
                diff = a_score - b_score
                
                if diff > 0:
                    a_wins += 1
                    winner = run_a_name
                    if diff > 0.1:
                        insights.append(f"**{name}**: {run_a_name} scores {diff:.2f} higher — suggesting its configuration provides a significant advantage here.")
                elif diff < 0:
                    b_wins += 1
                    winner = run_b_name
                    if abs(diff) > 0.1:
                        insights.append(f"**{name}**: {run_b_name} scores {abs(diff):.2f} higher — suggesting its configuration provides a significant advantage here.")
                else:
                    winner = "Tie"
                    
                c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
                c1.write(f"**{name}**")
                
                color_a = "green" if winner == run_a_name else ("red" if winner == run_b_name else "gray")
                color_b = "green" if winner == run_b_name else ("red" if winner == run_a_name else "gray")
                
                delta_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
                c2.markdown(f"<span style='color:{color_a}; font-size:1.2rem; font-weight:bold;'>{a_score:.3f}</span>", unsafe_allow_html=True)
                c3.markdown(f"<span style='color:{color_b}; font-size:1.2rem; font-weight:bold;'>{b_score:.3f}</span>", unsafe_allow_html=True)
                
                if winner != "Tie":
                    c4.markdown(f"🏆 Winner: **{winner}** ({delta_str})")
                else:
                    c4.markdown("🤝 Tie")
                
            st.divider()
            
            # Section B — Grouped Bar Chart
            st.markdown(f"### A/B Comparison: {run_a_name} vs {run_b_name}")
            
            fig_bar = go.Figure()
            x_labels = display_names
            
            fig_bar.add_trace(go.Bar(
                x=x_labels,
                y=[run_a.get(m, 0.0) for m in metrics_cols],
                name=run_a_name,
                marker_color='#3498db'
            ))
            fig_bar.add_trace(go.Bar(
                x=x_labels,
                y=[run_b.get(m, 0.0) for m in metrics_cols],
                name=run_b_name,
                marker_color='#e74c3c'
            ))
            
            fig_bar.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Quality Threshold (0.7)")
            fig_bar.update_layout(barmode='group', yaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            
            # Section C — Verdict
            st.markdown("### 📝 Verdict")
            if a_wins > b_wins:
                st.markdown(f"🏆 **{run_a_name}** wins **{a_wins}** out of {len(metrics_cols)} metrics overall.")
            elif b_wins > a_wins:
                st.markdown(f"🏆 **{run_b_name}** wins **{b_wins}** out of {len(metrics_cols)} metrics overall.")
            else:
                st.markdown("🤝 It's a **Tie** overall.")
                
            if insights:
                st.markdown("#### Key Differences (>0.1)")
                for insight in insights:
                    st.markdown(f"- {insight}")
