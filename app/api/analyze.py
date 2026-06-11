"""

POST /analyze — the single endpoint of Module A.

Orchestration order:
  1. Idempotency check      (Redis — skip re-analysis of same submission)
  2. Detect + validate language (Tree-sitter)
  3. Generate AST            (Tree-sitter)
  4. Extract AST metrics     (ast_metrics)
  5. Enqueue Semgrep & Scoring job to Celery
  6. Return 202 Accepted with Job ID

Design note on submission size
-------------------------------
This is a coding evaluation system. Submissions are NEVER rejected based on
size — a student may submit any amount of code and it must be evaluated fairly.

Large submissions (> 50,000 characters) are:
  - Logged with a WARNING so infra can monitor resource usage.
  - Flagged via `large_submission=True` in the response for reviewer awareness.
  - Processed normally — Tree-sitter and Semgrep both handle large files;
    Semgrep's --timeout and --max-memory flags prevent runaway workers.

Idempotency
-----------
Double submissions happen constantly at lakh scale — network retries, client
bugs, accidental re-submits. Without a check, the same submission would spawn
two Celery tasks and could return two different scores for the same submission_id.

On every POST /analyze we check Redis for an existing job_id mapped to the
submission_id. If one exists, we return it immediately without re-enqueuing.
The Redis key expires after 2 hours (matching result_expires in celery_app.py).

Async
-----
Handler is `async def` so it participates properly in the uvicorn event loop
rather than blocking a threadpool slot during the Redis idempotency check.
"""

import logging
import uuid
from fastapi import APIRouter, Depends, status
from app.dependencies import require_api_key
from fastapi.responses import JSONResponse

from app.models.schemes import AnalysisRequest, JobAcceptedResponse, PartialMetrics
from app.services.lang_detector import (
    detect_and_validate,
    LanguageNotSupportedError,
    LanguageMismatchError,
)
from app.services.ast_generator import generate_ast
from app.services.ast_metrics import extract_metrics
from app.worker.tasks import analyse_code

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

# Threshold above which a submission is flagged as large.
# Does NOT reject — purely informational for monitoring and reviewer UI.
_LARGE_SUBMISSION_THRESHOLD = 50_000  # characters

# Redis key prefix for idempotency mapping: submission_id → job_id
_IDEMPOTENCY_KEY_PREFIX = "idempotency:submission:"
# TTL for idempotency keys — matches Celery result_expires (1 hour)
_IDEMPOTENCY_TTL_SECONDS = 3600


def _error_response(submission_id: str, language: str, status_code: int, message: str) -> JSONResponse:
    """Return a consistent error JSON response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "submission_id": submission_id,
            "language": language,
            "analysis_status": "failed",
            "error": message,
        },
    )


def _get_redis_client():
    """
    Get a Redis client for idempotency checks.
    Imported lazily so the module loads cleanly even without Redis running
    (unit tests mock this out).
    Returns None if Redis is unavailable — idempotency is best-effort.
    """
    try:
        import redis
        from app import config
        return redis.from_url(config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
    except Exception:
        return None


@router.post(
    "/analyze",
    dependencies=[Depends(require_api_key)],
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyse a code submission (Async)",
    description=(
        "Accepts a raw code snippet, performs fast synchronous AST analysis, "
        "and enqueues the heavy Semgrep analysis to a Celery worker. "
        "Returns a job_id which can be polled for the final V1 Static Analysis Score. "
        "All submissions are accepted regardless of size. "
        "Duplicate submission_ids within 1 hour return the existing job_id without re-enqueuing."
    ),
)
async def analyze(request: AnalysisRequest) -> JSONResponse:
    """
    Run fast AST metrics and enqueue full analysis.
    All submissions are processed — size never causes rejection.
    Idempotent: duplicate submission_id within TTL returns existing job.
    """
    code = _normalize_code(request.code)
    code_length = len(code)
    code_size_bytes = len(code.encode("utf-8"))
    is_large = code_length > _LARGE_SUBMISSION_THRESHOLD

    if is_large:
        logger.warning(
            "Large submission received — submission_id=%s language=%s "
            "code_length=%d code_size_bytes=%d threshold=%d — "
            "processing normally, Semgrep timeout/memory guards active.",
            request.submission_id,
            request.language,
            code_length,
            code_size_bytes,
            _LARGE_SUBMISSION_THRESHOLD,
        )
    else:
        logger.info(
            "Analysis started — submission_id=%s language=%s "
            "code_length=%d code_size_bytes=%d",
            request.submission_id,
            request.language,
            code_length,
            code_size_bytes,
        )

    # Step 1: Idempotency check
    # If this submission_id was already enqueued within the TTL window,
    # return the existing job_id without re-running analysis.
    idempotency_key = f"{_IDEMPOTENCY_KEY_PREFIX}{request.submission_id}"
    redis_client = _get_redis_client()
    if redis_client:
        try:
            existing_job_id = redis_client.get(idempotency_key)
            if existing_job_id:
                existing_job_id = existing_job_id.decode("utf-8") if isinstance(existing_job_id, bytes) else existing_job_id
                logger.info(
                    "Duplicate submission detected — submission_id=%s "
                    "returning existing job_id=%s",
                    request.submission_id,
                    existing_job_id,
                )
                # Return 202 with the existing job_id — client can poll it.
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={
                        "job_id": existing_job_id,
                        "submission_id": request.submission_id,
                        "status": "queued",
                        "partial": None,
                    },
                )
        except Exception as exc:
            # Redis unavailable — proceed without idempotency check.
            logger.warning("Idempotency check failed (Redis unavailable): %s", exc)

    # Step 2 & 3: Language validation + AST generation
    try:
        detection = detect_and_validate(request.language, code)
    except LanguageNotSupportedError as exc:
        logger.warning("Unsupported language: %s", exc)
        return _error_response(
            submission_id=request.submission_id,
            language=request.language,
            status_code=400,
            message=str(exc),
        )
    except LanguageMismatchError as exc:
        logger.warning("Language mismatch: %s", exc)
        return _error_response(
            submission_id=request.submission_id,
            language=request.language,
            status_code=400,
            message=str(exc),
        )

    ast_result = generate_ast(detection, code)

    # Step 4: AST metrics
    ast_metrics = extract_metrics(ast_result)

    partial = PartialMetrics(
        cyclomatic_complexity=ast_metrics.cyclomatic_complexity,
        nesting_depth=ast_metrics.nesting_depth,
        function_length=ast_metrics.function_length,
        parameter_count=ast_metrics.parameter_count,
        code_size_bytes=code_size_bytes,
        large_submission=is_large,
    )

    # Generate a Job ID
    job_id = str(uuid.uuid4())

    # Step 5: Store idempotency mapping
    if redis_client:
        try:
            redis_client.set(idempotency_key, job_id, ex=_IDEMPOTENCY_TTL_SECONDS)
        except Exception as exc:
            logger.warning("Failed to store idempotency key (Redis unavailable): %s", exc)

    # Step 6: Enqueue to Celery
    analyse_code.apply_async(
        args=[
            job_id,
            request.submission_id,
            detection.language,
            code,
            partial.model_dump(),
        ],
        task_id=job_id,
    )

    logger.info(
        "Analysis enqueued — submission_id=%s job_id=%s large=%s",
        request.submission_id,
        job_id,
        is_large,
    )

    # Step 7: Return 202 Accepted
    response = JobAcceptedResponse(
        job_id=job_id,
        submission_id=request.submission_id,
        status="queued",
        partial=partial,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=response.model_dump(),
    )