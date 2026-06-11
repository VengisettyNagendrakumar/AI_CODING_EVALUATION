"""

Module C — Pipeline Orchestration API endpoints.

POST /api/v1/evaluate         — Submit code for full pipeline evaluation
GET  /api/v1/evaluate/{job_id} — Poll for unified report
"""

from __future__ import annotations

import json
import logging
import uuid

import redis as redis_lib
from celery import chord
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app import config
from app.dependencies import require_api_key
from app.models.pipeline_models import (
    PipelineEvalRequest,
    PipelineJobAcceptedResponse,
    PipelineStatusResponse,
)
from app.services.lang_detector import (
    detect_and_validate,
    LanguageNotSupportedError,
    LanguageMismatchError,
)
from app.services.ast_generator import generate_ast
from app.services.ast_metrics import extract_metrics
from app.worker.tasks import analyse_code
from app.worker.llm_tasks import evaluate_llm
from app.worker.celery_app import celery_app
from app.worker.pipeline_tasks import aggregate_and_report_task

logger = logging.getLogger(__name__)



def _normalize_code(raw: str) -> str:
    """
    Normalize raw code string before any processing.

    Handles two sources of the same submission:
      1. Real client (React editor, VS Code extension, backend form):
         Sends actual newlines — splitlines() handles correctly.

      2. Swagger UI / Postman / manual testing:
         Types literal \\n (backslash + n) instead of real newlines.
         We unescape these so the parser sees proper line breaks.

    Also strips trailing whitespace per line (prevents C++ preprocessor
    and editor quirks from causing spurious Tree-sitter parse errors).
    """
    # If the string contains literal \n (2 chars) but no real newlines,
    # it came from a client that didn't escape properly — unescape it.
    if "\\n" in raw and "\n" not in raw:
        raw = raw.replace("\\n", "\n")

    # Strip trailing whitespace from each line, remove trailing blank lines.
    lines = [line.rstrip() for line in raw.splitlines()]
    return "\n".join(lines).rstrip()

router = APIRouter()

# Redis client — lazy initialization for idempotency check and status polling
_redis_client = None


def _get_redis() -> redis_lib.Redis:
    """Return a shared Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis.from_url(
            config.CELERY_RESULT_BACKEND, decode_responses=True
        )
    return _redis_client

# TTL matches Celery result_expires (1 hour)
_IDEMPOTENCY_TTL_SECONDS = 3600
_IDEMPOTENCY_KEY_PREFIX = "idempotency:pipeline:"


# POST /api/v1/evaluate — Submit for full pipeline evaluation

@router.post(
    "/evaluate",
    response_model=PipelineJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
    summary="Submit code for full pipeline evaluation",
    description=(
        "Enqueues a full pipeline evaluation job (Static Code Analysis + LLM Evaluation). "
        "Returns a single master job_id immediately. "
        "Poll GET /api/v1/evaluate/{job_id} for the final aggregated report."
    ),
    tags=["Pipeline Evaluation"],
)
async def submit_pipeline_evaluation(request: PipelineEvalRequest) -> JSONResponse:
    """
    Enqueue parallel static and LLM evaluation tasks and return 202.
    """
    submission_id = request.submission_id
    language = request.language
    code = _normalize_code(request.code)

    # --- Step 1: Idempotency Check ---
    idem_key = f"{_IDEMPOTENCY_KEY_PREFIX}{submission_id}"
    existing_job_id = _get_redis().get(idem_key)
    if existing_job_id:
        logger.info(
            "Pipeline evaluation duplicate submission — reusing existing job — "
            "submission_id=%s job_id=%s",
            submission_id, existing_job_id,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id":        existing_job_id,
                "submission_id": submission_id,
                "status":        "queued",
                "error":         None,
            },
        )

    # --- Step 2: Language Detection & Synchronous AST Parsing ---
    try:
        detection = detect_and_validate(language, code)
        ast_result = generate_ast(detection, code)
        ast_metrics = extract_metrics(ast_result)
        has_syntax_error = ast_result.tree.root_node.has_error
    except (LanguageNotSupportedError, LanguageMismatchError) as exc:
        logger.warning(
            "Pipeline evaluation input validation failed — submission_id=%s: %s",
            submission_id, exc,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "submission_id": submission_id,
                "language":      language,
                "status":        "failed",
                "error":         str(exc),
            },
        )
    except Exception as exc:
        logger.exception("Unexpected exception during synchronous AST parsing:")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "submission_id": submission_id,
                "language":      language,
                "status":        "failed",
                "error":         f"Internal AST parsing error: {exc}",
            },
        )

    # Compile partial AST metrics payload for Celery task
    partial_metrics = {
        "cyclomatic_complexity": ast_metrics.cyclomatic_complexity,
        "nesting_depth":         ast_metrics.nesting_depth,
        "function_length":       ast_metrics.function_length,
        "parameter_count":       ast_metrics.parameter_count,
        "code_size_bytes":       len(code.encode("utf-8")),
        "large_submission":      len(code) > 50_000,
    }

    # --- Step 3: Generate Job UUIDs ---
    master_job_id = str(uuid.uuid4())
    static_job_id = str(uuid.uuid4())
    llm_job_id = str(uuid.uuid4())

    # --- Step 4: Save Job Mapping & Idempotency in Redis ---
    job_mapping = {
        "master_job_id": master_job_id,
        "static_job_id": static_job_id,
        "llm_job_id":    llm_job_id,
    }
    _get_redis().setex(f"eval_jobs:{master_job_id}", _IDEMPOTENCY_TTL_SECONDS, json.dumps(job_mapping))
    _get_redis().setex(idem_key, _IDEMPOTENCY_TTL_SECONDS, master_job_id)

    logger.info(
        "Enqueuing unified pipeline evaluation — submission_id=%s language=%s master_job_id=%s",
        submission_id, language, master_job_id,
    )

    # --- Step 5: Dispatch Celery Chord ---
    header = [
        analyse_code.signature(
            args=[static_job_id, submission_id, language, code, partial_metrics],
            task_id=static_job_id,
            queue="static_analysis",
        ),
        evaluate_llm.signature(
            args=[llm_job_id, submission_id, language, request.question, code],
            task_id=llm_job_id,
            queue="llm_evaluation",
        ),
    ]
    callback = aggregate_and_report_task.signature(
        args=[
            master_job_id,
            submission_id,
            language,
            request.total_test_cases,
            request.passed_test_cases,
            has_syntax_error,
        ],
        task_id=master_job_id,
        queue="static_analysis",
    )

    chord(header)(callback)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id":        master_job_id,
            "submission_id": submission_id,
            "status":        "queued",
            "error":         None,
        },
    )


# GET /api/v1/evaluate/{job_id} — Poll for unified report

@router.get(
    "/evaluate/{job_id}",
    response_model=PipelineStatusResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    summary="Poll Pipeline Evaluation result",
    description="Retrieve the status and combined results for a pipeline evaluation job.",
    tags=["Pipeline Evaluation"],
)
async def get_pipeline_evaluation_result(job_id: str) -> JSONResponse:
    """
    Poll the Redis backend for the pipeline result.
    If incomplete, check sub-tasks states dynamically.
    """
    raw_result = _get_redis().get(f"pipeline_result:{job_id}")

    # Case A: Completed or Failed result already cached in Redis
    if raw_result:
        try:
            result = json.loads(raw_result)
            return JSONResponse(status_code=status.HTTP_200_OK, content=result)
        except Exception:
            logger.error("Failed to deserialize pipeline result for job_id=%s", job_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "job_id": job_id,
                    "status": "failed",
                    "error":  "result_deserialization_error",
                },
            )

    # Case B: In-progress or Queued — look up job mappings to check sub-tasks
    mapping_raw = _get_redis().get(f"eval_jobs:{job_id}")
    if not mapping_raw:
        # Unknown Job ID or expired
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "job_id": job_id,
                "status": "queued",
                "error":  None,
            },
        )

    try:
        mapping = json.loads(mapping_raw)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "job_id": job_id,
                "status": "failed",
                "error":  "mapping_deserialization_error",
            },
        )

    static_job_id = mapping.get("static_job_id")
    llm_job_id = mapping.get("llm_job_id")

    # Poll sub-jobs state from Celery
    static_res = AsyncResult(static_job_id, app=celery_app)
    llm_res = AsyncResult(llm_job_id, app=celery_app)

    # If either task failed at Celery level (e.g. OOM or unhandled exception)
    if static_res.state == "FAILURE" or llm_res.state == "FAILURE":
        error_msg = ""
        if static_res.state == "FAILURE":
            error_msg += f"Static Analysis Celery task failed: {static_res.info}. "
        if llm_res.state == "FAILURE":
            error_msg += f"LLM Evaluation Celery task failed: {llm_res.info}."
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "job_id":      job_id,
                "status":      "failed",
                "job_mapping": mapping,
                "error":       error_msg.strip(),
            },
        )

    # Otherwise, both must be either PENDING, STARTED, or SUCCESS (but callback hasn't run yet)
    # We report it as "pending" (meaning processing)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "job_id":      job_id,
            "status":      "pending",
            "job_mapping": mapping,
            "error":       None,
        },
    )