"""

Single auto-detecting configuration for the entire system.

DevOps sets these environment variables:
    CODE_EVAL_ENV               = production | development (default: development)
    REDIS_URL                   = redis://your-elasticache-endpoint:6379/0
    REDIS_RESULT_URL            = redis://your-elasticache-endpoint:6379/1
    API_HOST                    = 0.0.0.0
    API_PORT                    = 8000
    API_KEY                     = your-secret-api-key
    STATIC_ANALYSIS_CONCURRENCY = 8  (optional override — auto-detected otherwise)

Module B (LLM evaluation):
    LLM_MODEL_PATH  = /models/llm
    LLM_CONCURRENCY = 2  (optional override)
"""

from __future__ import annotations

import logging
import logging.config
import multiprocessing
import os
import sys
from logging.handlers import RotatingFileHandler



os.makedirs("logs", exist_ok=True)

_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Minimal bootstrap so hardware-detection lines below can log immediately.
# This will be overwritten by uvicorn's dictConfig — that's fine.
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """
    Attach the RotatingFileHandler to the root logger.

    Call this from main.py's lifespan AFTER the FastAPI/uvicorn app is
    created — uvicorn runs its own logging.config.dictConfig() during
    startup which would wipe any handlers set before that point.

    Calling this after ensures the RotatingFileHandler survives and all
    log output (from uvicorn, FastAPI, our code) goes to the file.

    Safe to call multiple times — checks for existing handler first.
    """
    root = logging.getLogger()

    # Don't add a second RotatingFileHandler if already present
    # (e.g. called twice in tests).
    already_has_file_handler = any(
        isinstance(h, RotatingFileHandler) for h in root.handlers
    )
    if already_has_file_handler:
        return

    fmt = logging.Formatter(_LOG_FMT)

    file_handler = RotatingFileHandler(
        filename="logs/code_eval.log",
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,                # keep last 5 = max 50 MB total
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    # Also re-apply our format to the existing console handlers that
    # uvicorn set up — so console and file output look consistent.
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    logger.info(
        "RotatingFileHandler attached — logs/code_eval.log "
        "(10MB × 5 files, max 50MB)"
    )


# Environment

ENV           = os.getenv("CODE_EVAL_ENV", "development")
IS_PRODUCTION = ENV == "production"
IS_DEV        = not IS_PRODUCTION


# External services — set by DevOps as environment variables

REDIS_URL        = os.getenv("REDIS_URL",        "redis://localhost:6379/0")
REDIS_RESULT_URL = os.getenv("REDIS_RESULT_URL", REDIS_URL.replace("/0", "/1"))
API_HOST         = os.getenv("API_HOST",         "0.0.0.0")
API_PORT         = int(os.getenv("API_PORT",     "8000"))
API_KEY          = os.getenv("API_KEY",          "")   # empty = unprotected (dev only)


# Hardware auto-detection
# For Module A (CPU-bound: Tree-sitter + Semgrep) we only need CPU count.
# GPU detection is included here ready for Module B (LLM inference) and
# Module C (embedding models) — they will read IS_GPU, VRAM_GB, NUM_GPUS.

def _detect_hardware() -> dict:
    """
    Detect CPU cores and GPU resources.
    GPU detection is non-fatal — if torch is not installed, falls back to CPU.
    """
    cpu_cores = multiprocessing.cpu_count()
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            num_gpus = torch.cuda.device_count()
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_name = torch.cuda.get_device_properties(0).name
            return {
                "mode":      "gpu",
                "num_gpus":  num_gpus,
                "vram_gb":   round(vram_gb, 1),
                "gpu_name":  gpu_name,
                "cpu_cores": cpu_cores,
            }
    except Exception:
        pass
    return {
        "mode":      "cpu",
        "num_gpus":  0,
        "vram_gb":   0.0,
        "gpu_name":  "CPU only",
        "cpu_cores": cpu_cores,
    }


_HW       = _detect_hardware()
MODE      = _HW["mode"]
IS_GPU    = MODE == "gpu"
NUM_GPUS  = _HW["num_gpus"]
VRAM_GB   = _HW["vram_gb"]
GPU_NAME  = _HW["gpu_name"]
CPU_CORES = _HW["cpu_cores"]


# Worker concurrency — per module, auto-detected from hardware

def _get_concurrency(env_var: str, default: int) -> int:
    """Read concurrency from env var, fallback to default. Always >= 1."""
    val = os.getenv(env_var)
    if val:
        try:
            return max(1, int(val))
        except ValueError:
            logger.warning(
                "Invalid value for %s=%r, using default %d", env_var, val, default
            )
    return default


# Module A — CPU-bound (Tree-sitter + Semgrep)
# Use all cores minus 1 — leave one for OS + uvicorn
STATIC_ANALYSIS_CONCURRENCY = _get_concurrency(
    "STATIC_ANALYSIS_CONCURRENCY",
    default=max(1, CPU_CORES - 1),
)

# Module B — I/O-bound (LLM evaluation)
LLM_CONCURRENCY = _get_concurrency(
    "LLM_CONCURRENCY",
    default=NUM_GPUS if IS_GPU else 2,
)


# Celery pool — auto-detected from OS
# prefork: multi-process, correct for CPU-bound on Linux/macOS
# solo:    single-process, only option on Windows (prefork broken there)

CELERY_POOL           = "solo" if sys.platform == "win32" else "prefork"
CELERY_BROKER_URL     = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_RESULT_URL

TASK_ACKS_LATE             = True
TASK_REJECT_ON_WORKER_LOST = True
WORKER_PREFETCH_MULTIPLIER = 1
RESULT_EXPIRES_SECONDS     = 3600   # 1 hour


# Model caching paths — for Module B (LLM)

LLM_MODEL_PATH   = os.getenv("LLM_MODEL_PATH",   "models/llm")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "qwen2.5:7b")
OLLAMA_TIMEOUT   = float(os.getenv("OLLAMA_TIMEOUT", "300"))


# Idempotency

IDEMPOTENCY_TTL_SECONDS = RESULT_EXPIRES_SECONDS


# print_config — call at startup to confirm what was detected

def print_config() -> None:
    """Print a human-readable summary of the active configuration."""
    print(f"""
CODE EVALUATION — PIPELINE CONFIG
-----------------------------------
Environment  : {ENV}
Mode         : {MODE.upper()}
GPU          : {GPU_NAME}
Num GPUs     : {NUM_GPUS}
VRAM         : {VRAM_GB:.1f} GB
CPU Cores    : {CPU_CORES}

Workers
  static_analysis   : {STATIC_ANALYSIS_CONCURRENCY} ({CELERY_POOL} pool)
  llm_evaluation    : {LLM_CONCURRENCY} (gevent pool)

Redis Broker : {'elasticache' if 'localhost' not in REDIS_URL else 'localhost (dev)'}
Ollama URL   : {OLLAMA_BASE_URL}
Ollama Model : {OLLAMA_MODEL}
API Key      : {'set' if API_KEY else 'NOT SET (unprotected — dev mode)'}
Log File     : logs/code_eval.log (10MB × 5 files)
-----------------------------------
    """, flush=True)


# validate — warn about missing production settings

def validate() -> bool:
    """
    Warn if critical settings are missing or wrong in production.
    Call from main.py lifespan alongside validate_configs_on_startup().
    Returns True if all checks pass.
    """
    warnings = []
    if IS_PRODUCTION:
        if "localhost" in REDIS_URL:
            warnings.append("REDIS_URL is localhost — use ElastiCache in production")
        if not API_KEY:
            warnings.append("API_KEY is not set — all endpoints are unprotected")
        if ENV == "development":
            warnings.append(
                "CODE_EVAL_ENV is 'development' but IS_PRODUCTION evaluated True"
            )
    else:
        if not API_KEY:
            logger.info("API_KEY not set — running in open dev mode (expected)")

    for w in warnings:
        logger.warning("CONFIG WARNING: %s", w)

    return len(warnings) == 0