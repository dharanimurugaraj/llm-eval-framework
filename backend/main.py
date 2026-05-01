"""FastAPI application for the LLM Eval Framework backend API.

This module defines the API app instance, lifecycle handling, CORS policy,
health endpoints, and placeholder sections where feature routers will be added.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown events."""
    # Startup initialization hooks will be added here.
    yield
    # Shutdown cleanup hooks will be added here.


app = FastAPI(
    title="LLM Eval Framework API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return API health status and version."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
async def root() -> dict[str, str]:
    """Return root API message."""
    return {"message": "LLM Eval Framework API"}


# Evaluation router section:
# Will contain endpoints to run RAGAS/DeepEval and fetch score summaries.

# Ingestion router section:
# Will contain endpoints for document upload, chunking, and embedding jobs.

# Experiments router section:
# Will contain endpoints for W&B run history, comparisons, and regressions.
