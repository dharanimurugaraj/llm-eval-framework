# Architecture

## System Overview
LLM Eval Framework evaluates RAG pipeline quality across multiple
dimensions: retrieval quality, generation faithfulness, and answer
relevance.

## Components
- **Ingestion Pipeline**: Loads documents, applies chunking strategy,
  embeds and stores in Qdrant
- **RAG Pipeline**: Retrieves context, generates answer via LLM
- **Eval Pipeline**: Scores output using RAGAS + DeepEval metrics
- **Experiment Tracker**: Logs all runs to Weights & Biases
- **Dashboard**: Visualizes results and enables A/B comparison

## Data Flow
Document -> Chunker -> Embedder -> Qdrant
Query -> Qdrant Retrieval -> LLM -> Answer
Answer + Context -> RAGAS/DeepEval -> Scores -> W&B -> Dashboard

## Architecture Diagram
Coming soon — will add Mermaid diagram here.
