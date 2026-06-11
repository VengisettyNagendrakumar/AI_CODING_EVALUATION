"""

Celery application factory.

Two queues are active:
  static_analysis  — Module A (CPU-bound: Tree-sitter + Semgrep)
  llm_evaluation   — Module B (I/O-bound: LLM eval)

All configuration (broker URL, concurrency, queue names, reliability settings)
is imported from app.config — the single source of truth for the entire system.
Nothing is hardcoded here.
"""

from celery import Celery
from kombu import Queue
from app import config

celery_app = Celery(
    "code_eval_worker",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=[
        "app.worker.tasks",       # Module A — static analysis
        "app.worker.llm_tasks",   # Module B — LLM evaluation
        "app.worker.pipeline_tasks", # Module C — pipeline coordinator
    ],
)

# Queue definitions — one per module, independently scalable

celery_app.conf.task_queues = (
    Queue("static_analysis"),    # Module A — CPU-bound, Tree-sitter + Semgrep
    Queue("llm_evaluation"),     # Module B — I/O-bound, LLM evaluation
)

celery_app.conf.task_default_queue = "static_analysis"

celery_app.conf.task_routes = {
    "analyse_code": {"queue": "static_analysis"},
    "evaluate_llm": {"queue": "llm_evaluation"},
    "aggregate_and_report": {"queue": "static_analysis"},
}

# Apply all settings from config

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=config.RESULT_EXPIRES_SECONDS,
    task_acks_late=config.TASK_ACKS_LATE,
    task_reject_on_worker_lost=config.TASK_REJECT_ON_WORKER_LOST,
    worker_prefetch_multiplier=config.WORKER_PREFETCH_MULTIPLIER,
)

# Expose concurrency for run_async_app.py
WORKER_CONCURRENCY = config.STATIC_ANALYSIS_CONCURRENCY