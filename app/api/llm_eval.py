"""

Module B — LLM Evaluation API endpoints.

POST /api/v1/llm-eval         — Submit code for LLM evaluation (202 + job_id)
GET  /api/v1/llm-eval/{job_id} — Poll for result

Mirrors Module A's pattern:
  POST /api/v1/analyze
  GET  /api/v1/analyze/{job_id}
"""

from __future__ import annotations

import json
import logging
import uuid

import redis as redis_lib
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app import config
from app.dependencies import require_api_key
from app.models.llm_models import LLMEvalRequest, LLMJobAcceptedResponse
from app.worker.llm_tasks import evaluate_llm
from app.services.lang_detector import (
    detect_and_validate,
    LanguageNotSupportedError,
    LanguageMismatchError,
)

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

# Redis client — lazy initialization for polling results on GET

_redis_client = None


def _get_redis() -> redis_lib.Redis:
    """Return a shared Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis.from_url(
            config.CELERY_RESULT_BACKEND, decode_responses=True
        )
    return _redis_client


# POST /api/v1/llm-eval — submit for evaluation

@router.post(
    "/llm-eval",
    response_model=LLMJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
    summary="Submit code for LLM Evaluation (Async)",
    description=(
        "Enqueues an LLM evaluation job for the submitted code. "
        "Returns a job_id immediately. "
        "Poll GET /api/v1/llm-eval/{job_id} for the full result."
    ),
    tags=["LLM Evaluation"],
)
async def submit_llm_eval(request: LLMEvalRequest) -> JSONResponse:
    """
    Enqueue an LLM evaluation job and return 202 Accepted.
    """
    # Language + code validation
    # The pipeline endpoint (/api/v1/evaluate) does this via Tree-sitter.
    # The standalone LLM endpoint must do the same — otherwise someone
    # could send Java code claiming it's Python and Qwen evaluates wrongly.
    candidate_code = _normalize_code(request.candidate_code)

    try:
        detect_and_validate(request.language, candidate_code)
    except LanguageNotSupportedError as exc:
        logger.warning("LLM eval — unsupported language: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "submission_id": request.submission_id,
                "status": "failed",
                "error": str(exc),
            },
        )
    except LanguageMismatchError as exc:
        logger.warning("LLM eval — language mismatch: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "submission_id": request.submission_id,
                "status": "failed",
                "error": str(exc),
            },
        )

    job_id = str(uuid.uuid4())

    logger.info(
        "LLM eval enqueued — submission_id=%s language=%s job_id=%s",
        request.submission_id, request.language, job_id,
    )

    evaluate_llm.apply_async(
        args=[
            job_id,
            request.submission_id,
            request.language,
            request.question,
            candidate_code,
        ],
        task_id=job_id,
        queue="llm_evaluation",
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id":       job_id,
            "submission_id": request.submission_id,
            "status":       "queued",
            "error":        None,
        },
    )


# GET /api/v1/llm-eval/{job_id} — poll for result

@router.get(
    "/llm-eval/{job_id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    summary="Poll LLM Evaluation result",
    description=(
        "Returns the current status and result for an LLM evaluation job. "
        "Status values: queued | processing | completed | failed."
    ),
    tags=["LLM Evaluation"],
)
async def get_llm_eval_result(job_id: str) -> JSONResponse:
    """
    Retrieve LLM evaluation result from Redis.
    """
    raw = _get_redis().get(f"llm_result:{job_id}")

    if raw is None:
        # Not in Redis yet — either still queued or job_id is unknown
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "job_id": job_id,
                "status": "queued",
                "error":  None,
            },
        )

    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.error("Failed to deserialize LLM result for job_id=%s", job_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "job_id": job_id,
                "status": "failed",
                "error":  "result_deserialization_error",
            },
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=result)