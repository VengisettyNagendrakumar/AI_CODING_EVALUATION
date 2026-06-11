"""

Celery task for coordinating the score aggregation and pipeline evaluation report.
"""

from __future__ import annotations

import json
import logging
import time

import redis as redis_lib

from app.worker.celery_app import celery_app
from app import config
from app.services import orchestrator

logger = logging.getLogger(__name__)

# Redis client — lazy initialization
# Not created at import time to avoid crashing the worker if Redis is
# temporarily unavailable at startup.
_redis_client = None
_RESULT_TTL = config.RESULT_EXPIRES_SECONDS


def _get_redis() -> redis_lib.Redis:
    """Return a shared Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis.from_url(
            config.CELERY_RESULT_BACKEND, decode_responses=True
        )
    return _redis_client


def _store_pipeline_result(job_id: str, payload: dict) -> None:
    """Serialize and store the pipeline result dict in Redis with TTL."""
    _get_redis().setex(f"pipeline_result:{job_id}", _RESULT_TTL, json.dumps(payload))


def _store_pipeline_failure(
    job_id: str,
    submission_id: str,
    language: str,
    error_msg: str,
    static_result: dict | None = None,
    llm_result: dict | None = None,
    runtime_score: float = 0.0,
    job_mapping: dict | None = None,
) -> None:
    """Store a pipeline failure payload in Redis."""
    _store_pipeline_result(job_id, {
        "job_id":              job_id,
        "submission_id":       submission_id,
        "status":              "failed",
        "language":            language,
        "pipeline_version":    orchestrator.PIPELINE_VERSION,
        "aggregation_version": orchestrator.AGGREGATION_VERSION,
        "static_analysis":     static_result,
        "llm_evaluation":      llm_result,
        "scores": {
            "static_score":    static_result.get("scores", {}).get("static_score") if static_result and static_result.get("scores") else None,
            "llm_score":       llm_result.get("scores", {}).get("llm_score") if llm_result and llm_result.get("scores") else None,
            "runtime_score":   runtime_score,
            "final_score":     None,
        },
        "explainability":      None,
        "job_mapping":         job_mapping,
        "error":               error_msg,
    })


@celery_app.task(
    name="aggregate_and_report",
    acks_late=True,
    reject_on_worker_lost=True,
)
def aggregate_and_report_task(
    results: list[dict],
    job_id: str,
    submission_id: str,
    language: str,
    total_test_cases: int,
    passed_test_cases: int,
    has_syntax_error: bool,
) -> dict:
    """
    Chord callback task that aggregates static analysis and LLM evaluation results.

    Args:
        results:           List of task execution result dicts (from Celery group).
        job_id:            Master evaluation job identifier.
        submission_id:     Original submission identifier.
        language:          Programming language.
        total_test_cases:  Total test case count.
        passed_test_cases: Passed test case count.
        has_syntax_error:  True if the parsed tree has a syntax error.

    Returns:
        Final pipeline evaluation report dictionary.
    """
    logger.info("aggregate_and_report_task started — job_id=%s", job_id)

    # 1. Load job mapping from Redis for debugging traceability
    job_mapping = None
    mapping_raw = _get_redis().get(f"eval_jobs:{job_id}")
    if mapping_raw:
        try:
            job_mapping = json.loads(mapping_raw)
        except Exception as e:
            logger.warning("Failed to parse job mapping for job_id=%s: %s", job_id, e)

    # 2. Extract static and LLM results from the parallel results list
    static_result = None
    llm_result = None

    for res in results:
        if not isinstance(res, dict):
            continue
        if "analysis_status" in res:
            static_result = res
        else:
            llm_result = res


    # Check for empty results
    if static_result is None or llm_result is None:
        err = "Missing static analysis or LLM evaluation result in group output."
        logger.error("%s job_id=%s", err, job_id)
        _store_pipeline_failure(job_id, submission_id, language, err, static_result, llm_result, 0.0, job_mapping)
        return {"status": "failed", "error": err}

    # 3. Check for sub-task failures
    static_failed = static_result.get("analysis_status") == "failed"
    llm_failed = llm_result.get("status") == "failed"

    # Compute runtime score
    runtime_score = orchestrator.calculate_runtime_score(passed_test_cases, total_test_cases)

    if static_failed or llm_failed:
        err_msg = ""
        if static_failed:
            err_msg += f"Static Analysis failed: {static_result.get('error', 'unknown error')}. "
        if llm_failed:
            err_msg += f"LLM Evaluation failed: {llm_result.get('error', 'unknown error')}."
        
        logger.error("Sub-task failure detected in pipeline evaluation: %s", err_msg)
        _store_pipeline_failure(
            job_id, submission_id, language, err_msg.strip(),
            static_result, llm_result, runtime_score, job_mapping
        )
        return {"status": "failed", "error": err_msg.strip()}

    # 4. Extract individual scores
    static_score = None
    if static_result.get("scores"):
        static_score = static_result["scores"].get("static_score")

    llm_score = None
    if llm_result.get("scores"):
        llm_score = llm_result["scores"].get("llm_score")

    # 5. Aggregate scores and apply capping rules
    try:
        final_score, breakdown = orchestrator.aggregate_scores(
            static_score=static_score,
            llm_score=llm_score,
            runtime_score=runtime_score,
            has_syntax_error=has_syntax_error,
            static_failed=static_failed,
        )
    except Exception as exc:
        err = f"Failed to aggregate scores: {exc}"
        logger.exception("Aggregation error:")
        _store_pipeline_failure(job_id, submission_id, language, err, static_result, llm_result, runtime_score, job_mapping)
        return {"status": "failed", "error": err}

    # 6. Generate programmatic explainability summary
    explain_data = orchestrator.generate_explainability(
        static_result=static_result,
        llm_result=llm_result,
        runtime_score=runtime_score,
        has_syntax_error=has_syntax_error,
    )

    # 7. Assemble complete final result payload
    pipeline_result = {
        "job_id":              job_id,
        "submission_id":       submission_id,
        "status":              "completed",
        "language":            language,
        "pipeline_version":    orchestrator.PIPELINE_VERSION,
        "aggregation_version": orchestrator.AGGREGATION_VERSION,
        "static_analysis":     static_result,
        "llm_evaluation":      llm_result,
        "scores": {
            "static_score":    static_score,
            "llm_score":       llm_score,
            "runtime_score":   runtime_score,
            "final_score":     final_score,
        },
        "explainability":      explain_data,
        "job_mapping":         job_mapping,
        "error":               None,
    }

    # 8. Write payload to Redis
    _store_pipeline_result(job_id, pipeline_result)
    logger.info("Pipeline evaluation job completed successfully — job_id=%s final_score=%.1f", job_id, final_score)

    return pipeline_result