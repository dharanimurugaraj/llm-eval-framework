"""Streamlit dashboard for LLM Eval Framework result visualization.

This app will show evaluation summaries, execution workflows, historical
experiments, and side-by-side comparisons as the framework matures.
"""

import streamlit as st

st.set_page_config(page_title="LLM Eval Framework", page_icon="📊", layout="wide")

st.title("LLM Eval Framework")
st.caption("Production-grade evaluation framework for RAG quality benchmarking.")

page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Run Evaluation", "Experiment History", "A/B Comparison"],
)

st.header(page)
st.info("Coming soon")
