"""

FastAPI application entry point for Modules A + B + C.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analyze_router
from app.api.result import router as result_router
from app.api.llm_eval import router as llm_eval_router
from app.api.pipeline_router import router as pipeline_router
from app.utils.config_loader import validate_configs_on_startup
from app import config
from app.services.llm_eval import qwen_client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----

    # Attach RotatingFileHandler AFTER uvicorn's dictConfig runs.
    # If attached during import, uvicorn wipes it. Here it survives.
    config.configure_logging()

    logger.info("Starting up — Modules A + B + C")

    # Print hardware-detected config to deployment logs
    config.print_config()

    # Warn about missing production settings
    config.validate()

    # Force-load and validate all YAML configs at startup.
    # Fails immediately if any file is missing, empty, has bad syntax,
    # or scoring weights don't sum to 1.0.
    validate_configs_on_startup()

    
    ollama_healthy = qwen_client.check_health()
    if not ollama_healthy:
        logger.warning(
            "Ollama is not available at startup — Module B (LLM evaluation) "
            "will fail until Ollama is running. "
            "Module A (static analysis) is unaffected. "
            "To fix: ollama serve && ollama pull %s",
            qwen_client.OLLAMA_MODEL,
        )
    else:
        logger.info(
            "Ollama health check passed — model=%s ready",
            qwen_client.OLLAMA_MODEL,
        )

    logger.info("All services ready.")
    yield

    # ---- Shutdown ----
    logger.info("Shutting down.")


app = FastAPI(
    title="Code Evaluation — Modules A + B + C",
    description=(
        "Static Code Analysis (Module A) + LLM Evaluation (Module B) "
        "+ Score Aggregation (Module C). "
        "Module A: AST metrics via Tree-sitter and static analysis via Semgrep. "
        "Module B: Code quality evaluation via Qwen-2.5 7B (Ollama). "
        "Module C: Pipeline orchestration, score aggregation, and explainability."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — configured by backend team for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(analyze_router,  prefix="/api/v1", tags=["Static Analysis"])
app.include_router(result_router,   prefix="/api/v1", tags=["Static Analysis"])
app.include_router(llm_eval_router, prefix="/api/v1", tags=["LLM Evaluation"])
app.include_router(pipeline_router, prefix="/api/v1", tags=["Pipeline Evaluation"])


@app.get("/health", tags=["Health"], summary="Service health check")
async def health() -> dict:
    """Returns service status including Ollama availability."""
    ollama_ok = qwen_client.check_health()
    return {
        "status":        "ok",
        "version":       app.version,
        "env":           config.ENV,
        "mode":          config.MODE,
        "workers":       config.STATIC_ANALYSIS_CONCURRENCY,
        "ollama_model":  qwen_client.OLLAMA_MODEL,
        "ollama_status": "available" if ollama_ok else "unavailable",
    }