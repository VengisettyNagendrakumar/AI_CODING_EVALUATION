"""
config.py

Single auto-detecting configuration for the entire system.

DevOps sets these environment variables:
    CODE_EVAL_ENV               = production | development (default: development)
    REDIS_URL                   = redis://your-elasticache-endpoint:6379/0
    REDIS_RESULT_URL            = redis://your-elasticache-endpoint:6379/1
    API_HOST                    = 0.0.0.0
    API_PORT                    = 8000
    API_KEY                     = your-secret-api-key
    STATIC_ANALYSIS_CONCURRENCY = 8  (optional override — auto-detected otherwise)
    LLM_CONCURRENCY             = 2  (optional override)
    LLM_BATCH_SIZE              = 4  (optional override — auto-detected from VRAM)
    OLLAMA_BASE_URL             = http://localhost:11434
    OLLAMA_MODEL                = qwen2.5:7b
    OLLAMA_TIMEOUT              = 300
"""

from __future__ import annotations

import logging
import logging.config
import multiprocessing
import os
import sys
from logging.handlers import RotatingFileHandler


# Logging bootstrap

os.makedirs("logs", exist_ok=True)

_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=_LOG_FMT)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """
    Attach the RotatingFileHandler to the root logger.

    Call this from main.py's lifespan AFTER FastAPI/uvicorn starts.
    uvicorn runs its own logging.config.dictConfig() during startup
    which wipes any handlers set before that point. Calling here
    ensures RotatingFileHandler survives and all log output goes to file.

    Safe to call multiple times — checks for existing handler first.
    """
    root = logging.getLogger()

    already_has_file_handler = any(
        isinstance(h, RotatingFileHandler) for h in root.handlers
    )
    if already_has_file_handler:
        return

    fmt = logging.Formatter(_LOG_FMT)

    file_handler = RotatingFileHandler(
        filename="logs/code_eval.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,               # keep last 5 = max 50 MB total
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

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


# External services

REDIS_URL        = os.getenv("REDIS_URL",        "redis://localhost:6379/0")
REDIS_RESULT_URL = os.getenv("REDIS_RESULT_URL", REDIS_URL.replace("/0", "/1"))
API_HOST         = os.getenv("API_HOST",         "0.0.0.0")
API_PORT         = int(os.getenv("API_PORT",     "8000"))
API_KEY          = os.getenv("API_KEY",          "")


# Hardware auto-detection
# Module A (Tree-sitter + Semgrep) — CPU-bound, uses prefork workers
# Module B (LLM via Ollama)        — GPU-bound, uses batch size

def _detect_hardware() -> dict:
    """
    Detect CPU cores and GPU resources.
    GPU detection is non-fatal — falls back to CPU if torch not installed.
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


STATIC_ANALYSIS_CONCURRENCY = _get_concurrency(
    "STATIC_ANALYSIS_CONCURRENCY",
    default=max(1, CPU_CORES - 1),
)

LLM_CONCURRENCY = _get_concurrency(
    "LLM_CONCURRENCY",
    default=NUM_GPUS if IS_GPU else 2,
)




def _calc_batch_size(vram_gb: float) -> int:
    """
    Calculate optimal LLM batch size from available VRAM.

    Args:
        vram_gb: Total VRAM on GPU 0 in gigabytes.

    Returns:
        Optimal batch size as a power of 2, minimum 1.
    """
    # Reserve 14GB for model weights + 4GB headroom
    available_gb = max(0.0, vram_gb - 14.0 - 4.0)

    # Each item in batch needs ~500MB
    max_items = int(available_gb / 0.5)

    # Return largest power of 2 that fits
    for size in [32, 16, 8, 4, 2]:
        if max_items >= size:
            return size
    return 1


def _get_batch_size() -> int:
    """
    Return the LLM batch size.

    Priority:
      1. LLM_BATCH_SIZE env var — explicit production override
      2. Auto-detected from VRAM — if GPU available
      3. 1 — CPU or Ollama (no batching possible)
    """
    env_val = os.getenv("LLM_BATCH_SIZE")
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            logger.warning(
                "Invalid LLM_BATCH_SIZE=%r, using auto-detected value", env_val
            )

    if IS_GPU:
        return _calc_batch_size(VRAM_GB)

    # CPU or Ollama — no batching possible
    return 1


LLM_BATCH_SIZE = _get_batch_size()

# Total concurrent LLM items = workers × batch_size
# Example: 2 GPU workers × batch_size 8 = 16 prompts processed simultaneously
LLM_THROUGHPUT = LLM_CONCURRENCY * LLM_BATCH_SIZE


# Celery pool — auto-detected from OS
#
# prefork: multi-process — Linux/macOS production
#   Each worker is a separate process with its own memory
#   N workers = N students analysed in parallel
#   Correct for CPU-bound work (Semgrep, Tree-sitter)
#
# solo: single-process — Windows development only
#   One task at a time
#   prefork doesn't work on Windows due to Python multiprocessing limits
#   Fine for local dev/testing, never use in production

CELERY_POOL           = "solo" if sys.platform == "win32" else "prefork"
CELERY_BROKER_URL     = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_RESULT_URL

TASK_ACKS_LATE             = True
TASK_REJECT_ON_WORKER_LOST = True
WORKER_PREFETCH_MULTIPLIER = 1
RESULT_EXPIRES_SECONDS     = 3600  # 1 hour


# Ollama / LLM model config

LLM_MODEL_PATH  = os.getenv("LLM_MODEL_PATH",  "models/llm")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")
OLLAMA_TIMEOUT  = float(os.getenv("OLLAMA_TIMEOUT", "300"))


# Idempotency

IDEMPOTENCY_TTL_SECONDS = RESULT_EXPIRES_SECONDS


# print_config

def print_config() -> None:
    """Print a human-readable summary of the active configuration."""
    batch_note = (
        f"GPU auto-detected (VRAM={VRAM_GB}GB)"
        if IS_GPU
        else "CPU/Ollama — always 1 (no batching)"
    )
    pool_note = (
        "prefork — multi-process, parallel (Linux production)"
        if CELERY_POOL == "prefork"
        else "solo — single-process (Windows dev only)"
    )
    print(f"""
CODE EVALUATION — PIPELINE CONFIG
-----------------------------------
Environment  : {ENV}
Mode         : {MODE.upper()}
GPU          : {GPU_NAME}
Num GPUs     : {NUM_GPUS}
VRAM         : {VRAM_GB:.1f} GB
CPU Cores    : {CPU_CORES}

Workers (Module A — Static Analysis)
  Pool        : {pool_note}
  Concurrency : {STATIC_ANALYSIS_CONCURRENCY} workers
  Parallel    : {STATIC_ANALYSIS_CONCURRENCY} students simultaneously

Workers (Module B — LLM Evaluation)
  Concurrency : {LLM_CONCURRENCY} workers
  Batch Size  : {LLM_BATCH_SIZE} ({batch_note})
  Throughput  : {LLM_THROUGHPUT} prompts per batch cycle

Redis Broker : {'elasticache' if 'localhost' not in REDIS_URL else 'localhost (dev)'}
Ollama URL   : {OLLAMA_BASE_URL}
Ollama Model : {OLLAMA_MODEL}
API Key      : {'set' if API_KEY else 'NOT SET (unprotected — dev mode)'}
Log File     : logs/code_eval.log (10MB × 5 files)
-----------------------------------
    """, flush=True)


# validate

def validate() -> bool:
    """
    Warn if critical settings are missing or wrong in production.
    Returns True if all checks pass.
    """
    warnings = []
    if IS_PRODUCTION:
        if "localhost" in REDIS_URL:
            warnings.append("REDIS_URL is localhost — use ElastiCache in production")
        if not API_KEY:
            warnings.append("API_KEY is not set — all endpoints are unprotected")
        if CELERY_POOL == "solo":
            warnings.append(
                "Celery pool is 'solo' in production — only 1 task at a time. "
                "Deploy on Linux to use prefork for parallel processing."
            )
    else:
        if not API_KEY:
            logger.info("API_KEY not set — running in open dev mode (expected)")

    for w in warnings:
        logger.warning("CONFIG WARNING: %s", w)

    return len(warnings) == 0