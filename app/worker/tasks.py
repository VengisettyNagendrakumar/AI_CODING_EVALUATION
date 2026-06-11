"""

Celery tasks for executing heavy static analysis (Semgrep) in the background.

Retry design
------------
Three categories of failure are handled differently:

1. TRANSIENT errors (network blip, Redis timeout, Semgrep process crash)
   → Retried automatically up to MAX_RETRIES times with exponential backoff.
   → Examples: subprocess.TimeoutExpired, OSError, ConnectionError.

2. PERMANENT errors (bad input, unsupported language, scoring config wrong)
   → NOT retried — retrying will produce the same failure every time.
   → Caught explicitly, logged, returned as analysis_status="failed".
   → Examples: KeyError on partial_ast_metrics, ValueError from scorer.

3. UNEXPECTED errors (bugs, unknown exceptions)
   → Retried up to MAX_RETRIES — may be transient.
   → If all retries exhausted, returned as analysis_status="failed".

Retry schedule (exponential backoff with jitter):
   Attempt 1: immediate
   Attempt 2: ~10s
   Attempt 3: ~30s
   Attempt 4: ~90s  (final)

This prevents a storm of simultaneous retries hammering a recovering service.
"""

import logging
import subprocess

from celery.exceptions import MaxRetriesExceededError

from app.worker.celery_app import celery_app
from app.services.semgrep_runner import run_semgrep
from app.services.metrics_aggregator import aggregate
from app.services.ast_metrics import ASTMetrics
from app.services.scorer import compute_scores

logger = logging.getLogger(__name__)

# Maximum number of retry attempts after the first failure.
# Total attempts = 1 (original) + 3 (retries) = 4.
_MAX_RETRIES = 3

# Base delay in seconds for exponential backoff.
# Actual delay = _RETRY_BACKOFF_BASE ^ retry_number
# Retry 1: 2^1 = 2s, Retry 2: 2^2 = 4s ... with Celery jitter applied on top.
_RETRY_BACKOFF_BASE = 10

# Exceptions that are worth retrying — transient infrastructure failures.
# Permanent logic errors (ValueError, KeyError) are intentionally excluded.
_RETRYABLE_EXCEPTIONS = (
    subprocess.TimeoutExpired,   # Semgrep subprocess timed out
    OSError,                     # temp file I/O failure
    ConnectionError,             # Redis broker blip
    TimeoutError,                # generic timeout
)


def _failed_result(submission_id: str, language: str, error: str) -> dict:
    """Return a consistent failed-analysis result dict."""
    return {
        "submission_id": submission_id,
        "language": language,
        "analysis_status": "failed",
        "error": error,
        "metrics": None,
        "scores": None,
    }


@celery_app.task(
    bind=True,
    name="analyse_code",
    # Retry configuration
    # max_retries:    total retries allowed after the first attempt fails.
    # default_retry_delay: ignored when countdown is set explicitly below,
    #                      but kept as a safety fallback.
    max_retries=_MAX_RETRIES,
    default_retry_delay=_RETRY_BACKOFF_BASE,
)
def analyse_code(
    self,
    job_id: str,
    submission_id: str,
    language: str,
    code: str,
    partial_ast_metrics: dict,
) -> dict:
    """
    Run Semgrep on the submitted code, aggregate with AST metrics, and compute final scores.
    large_submission flag is forwarded from partial_ast_metrics so Semgrep applies
    appropriate timeout/memory guards for large files without rejecting them.

    Retries automatically on transient failures (Semgrep crash, I/O error,
    broker blip) with exponential backoff. Permanent failures (bad config,
    malformed metrics) are returned as failed without retrying.
    """
    retry_num = self.request.retries  # 0 on first attempt, 1/2/3 on retries
    logger.info(
        "Task analyse_code started — job_id=%s submission_id=%s "
        "large=%s attempt=%d/%d",
        job_id,
        submission_id,
        partial_ast_metrics.get("large_submission", False),
        retry_num + 1,
        _MAX_RETRIES + 1,
    )

    # Step 1: Validate partial_ast_metrics keys up front.
    # This is a permanent error — bad input won't fix itself on retry.
    required_keys = {
        "cyclomatic_complexity", "nesting_depth",
        "function_length", "parameter_count",
    }
    missing = required_keys - partial_ast_metrics.keys()
    if missing:
        logger.error(
            "partial_ast_metrics missing keys — job_id=%s missing=%s — not retrying",
            job_id, sorted(missing),
        )
        return _failed_result(
            submission_id, language,
            f"Internal error: partial_ast_metrics missing keys: {sorted(missing)}",
        )

    # Step 2: Run Semgrep (transient failures are retried).
    try:
        large_submission: bool = partial_ast_metrics.get("large_submission", False)
        semgrep_result = run_semgrep(language, code, large_submission=large_submission)

    except _RETRYABLE_EXCEPTIONS as exc:
        # Transient — retry with exponential backoff.
        delay = _RETRY_BACKOFF_BASE ** (retry_num + 1)
        logger.warning(
            "Semgrep transient failure — job_id=%s attempt=%d error=%s "
            "retrying in %ds",
            job_id, retry_num + 1, exc, delay,
        )
        try:
            raise self.retry(exc=exc, countdown=delay)
        except MaxRetriesExceededError:
            logger.error(
                "All retries exhausted for job_id=%s — returning failed result",
                job_id,
            )
            return _failed_result(submission_id, language, f"Semgrep failed after {_MAX_RETRIES} retries: {exc}")

    except Exception as exc:
        # Unexpected — retry in case it's transient; give up after max retries.
        delay = _RETRY_BACKOFF_BASE ** (retry_num + 1)
        logger.exception(
            "Unexpected Semgrep error — job_id=%s attempt=%d — retrying in %ds",
            job_id, retry_num + 1, delay,
        )
        try:
            raise self.retry(exc=exc, countdown=delay)
        except MaxRetriesExceededError:
            logger.error(
                "All retries exhausted for job_id=%s — returning failed result",
                job_id,
            )
            return _failed_result(submission_id, language, f"Analysis failed after {_MAX_RETRIES} retries: {exc}")

    # Step 3: Aggregate + Score.
    # These are pure Python operations on already-validated data.
    # Failures here are permanent (config/logic bugs) — not retried.
    try:
        ast_metrics = ASTMetrics(
            cyclomatic_complexity=partial_ast_metrics["cyclomatic_complexity"],
            nesting_depth=partial_ast_metrics["nesting_depth"],
            function_length=partial_ast_metrics["function_length"],
            parameter_count=partial_ast_metrics["parameter_count"],
        )

        metrics = aggregate(
            ast_metrics,
            semgrep_result,
            code_size_bytes=partial_ast_metrics.get("code_size_bytes", 0),
            large_submission=large_submission,
        )
        scores  = compute_scores(metrics, semgrep_available=semgrep_result.semgrep_available)

    except (KeyError, ValueError) as exc:
        logger.error(
            "Scoring configuration error — job_id=%s error=%s — not retrying",
            job_id, exc,
        )
        return _failed_result(submission_id, language, f"Scoring error: {exc}")

    # Step 4: Build and return the result.
    #
    # analysis_status has 3 values:
    #   "completed" — Semgrep ran successfully, all scores are valid
    #   "degraded"  — Semgrep failed, AST metrics are valid but
    #                 security/reliability scores are unavailable.
    #                 Downstream should treat Semgrep-derived scores
    #                 as None, not as "perfect".
    #   "failed"    — entire analysis failed
    if not semgrep_result.semgrep_available:
        analysis_status = "degraded"
        logger.warning(
            "Task analyse_code degraded — Semgrep unavailable — "
            "job_id=%s reason=%s",
            job_id, semgrep_result.failure_reason,
        )
    else:
        analysis_status = "completed"

    result = {
        "submission_id": submission_id,
        "language": language,
        "metrics": {
            "cyclomatic_complexity":    metrics.cyclomatic_complexity,
            "nesting_depth":            metrics.nesting_depth,
            "function_length":          metrics.function_length,
            "parameter_count":          metrics.parameter_count,
            "security_issues":          metrics.security_issues,
            "reliability_issues":       metrics.reliability_issues,
            "best_practice_violations": metrics.best_practice_violations,
            "code_size_bytes":          partial_ast_metrics.get("code_size_bytes", 0),
            "large_submission":         large_submission,
        },
        "scores": {
            "complexity_score":      scores.complexity_score,
            # If Semgrep unavailable, surface None so backend knows these
            # are not real findings — do not use as "perfect scores"
            "security_score":        scores.security_score if semgrep_result.semgrep_available else None,
            "maintainability_score": scores.maintainability_score,
            "reliability_score":     scores.reliability_score if semgrep_result.semgrep_available else None,
            "best_practice_score":   scores.best_practice_score if semgrep_result.semgrep_available else None,
            "static_score":          scores.static_score,
        },
        "analysis_status": analysis_status,
        # Surface Semgrep failure reason so backend/reviewer can see it
        "semgrep_available": semgrep_result.semgrep_available,
        "semgrep_failure_reason": semgrep_result.failure_reason if not semgrep_result.semgrep_available else None,
    }

    logger.info(
        "Task analyse_code completed — job_id=%s attempt=%d/%d static_score=%s",
        job_id, retry_num + 1, _MAX_RETRIES + 1,
        result["scores"]["static_score"],
    )
    return result