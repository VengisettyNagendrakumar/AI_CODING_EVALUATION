"""

Celery task for Module B — LLM Evaluation.

Task flow:
  1. Record start_time
  2. Build evaluation prompt (prompt_builder) — truncates if code > 12,000 chars
  3. Call Ollama / Qwen-2.5:7b (qwen_client) — attempt 1
  4. Parse the JSON response (parser)
       → if None: retry with simplified prompt — attempt 2
       → if still None: store { status: "failed", error: "parse_failure" }
  5. Compute weighted llm_score + confidence signal (evaluator)
  6. Compute processing_time_ms
  7. Store full result in Redis

Results are stored with the job_id as the Redis key so the API can
retrieve them via GET /api/v1/llm-eval/{job_id}.
"""

from __future__ import annotations

import json
import logging
import time

import redis as redis_lib

from app.worker.celery_app import celery_app
from app import config
from app.services.llm_eval import prompt_builder, qwen_client, parser, evaluator

logger = logging.getLogger(__name__)

# Redis client — lazy initialization

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


# Helpers

def _store_result(job_id: str, payload: dict) -> None:
    """Serialize and store the result dict in Redis with TTL."""
    _get_redis().setex(f"llm_result:{job_id}", _RESULT_TTL, json.dumps(payload))


def _store_failure(
    job_id: str,
    submission_id: str,
    language: str,
    error_msg: str,
    raw_response: str | None = None,
    processing_time_ms: int = 0,
    truncated: bool = False,
) -> None:
    """Store a standardised failure payload in Redis."""
    _store_result(job_id, {
        "job_id":             job_id,
        "submission_id":      submission_id,
        "status":             "failed",
        "language":           language,
        "model":              qwen_client.OLLAMA_MODEL,
        "prompt_version":     prompt_builder.PROMPT_VERSION,
        "processing_time_ms": processing_time_ms,
        "truncated":          truncated,
        "confidence":         None,
        "evaluation":         None,
        "scores":             None,
        "strengths":          None,
        "weaknesses":         None,
        "recommendations":    None,
        "raw_response":       raw_response,
        "error":              error_msg,
    })


# Celery task

@celery_app.task(
    name="evaluate_llm",
    bind=True,
    queue="llm_evaluation",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=0,  # retries handled internally (parse retry)
)
def evaluate_llm(
    self,
    job_id: str,
    submission_id: str,
    language: str,
    question: str,
    candidate_code: str,
) -> dict:
    """
    Evaluate a candidate's code submission using Qwen-2.5 7B via Ollama.
    """
    code_length = len(candidate_code)
    logger.info(
        "evaluate_llm started — job_id=%s submission_id=%s "
        "language=%s code_length=%d",
        job_id, submission_id, language, code_length,
    )

    start_time = time.time()

    # Step 1: Build prompt — truncates code if > MAX_CODE_CHARS
    prompt, was_truncated = prompt_builder.build_prompt(
        language=language,
        question=question,
        candidate_code=candidate_code,
    )

    if was_truncated:
        logger.warning(
            "Code truncated for LLM evaluation — job_id=%s "
            "original_length=%d truncated_to=%d",
            job_id, code_length, prompt_builder.MAX_CODE_CHARS,
        )

    # Step 2: Call Ollama — attempt 1
    raw_response_1: str | None = None
    try:
        raw_response_1 = qwen_client.generate(prompt)
    except Exception as exc:
        logger.error("Ollama call failed on attempt 1 — job_id=%s: %s", job_id, exc)
        elapsed_ms = int((time.time() - start_time) * 1000)
        _store_failure(
            job_id, submission_id, language, str(exc),
            None, elapsed_ms, was_truncated,
        )
        return {"status": "failed", "error": str(exc)}

    # Step 3: Parse response — attempt 1
    parsed = parser.parse(raw_response_1)

    if parsed is None:
        logger.warning(
            "Parse failed on attempt 1 — retrying — job_id=%s", job_id
        )

        retry_prompt, _ = prompt_builder.build_retry_prompt(
            language=language,
            question=question,
            candidate_code=candidate_code,
            previous_raw_response=raw_response_1,
        )

        raw_response_2: str | None = None
        try:
            raw_response_2 = qwen_client.generate(retry_prompt)
        except Exception as exc:
            logger.error(
                "Ollama call failed on retry — job_id=%s: %s", job_id, exc
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            _store_failure(
                job_id, submission_id, language, str(exc),
                raw_response_1, elapsed_ms, was_truncated,
            )
            return {"status": "failed", "error": str(exc)}

        parsed = parser.parse(raw_response_2)

        if parsed is None:
            logger.error(
                "Parse failed on retry — marking failed — job_id=%s", job_id
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            _store_failure(
                job_id, submission_id, language,
                "parse_failure: LLM did not return valid JSON after 2 attempts.",
                raw_response_2 or raw_response_1,
                elapsed_ms,
                was_truncated,
            )
            return {"status": "failed", "error": "parse_failure"}

        final_raw = raw_response_2
    else:
        final_raw = raw_response_1

    # Step 4: Compute scores, confidence, assemble result
    elapsed_ms = int((time.time() - start_time) * 1000)

    result = evaluator.build_result(
        job_id=job_id,
        submission_id=submission_id,
        language=language,
        evaluation=parsed,
        strengths=parsed.get("strengths", []),
        weaknesses=parsed.get("weaknesses", []),
        recommendations=parsed.get("recommendations", []),
        raw_response=final_raw,
        prompt_version=prompt_builder.PROMPT_VERSION,
        model=qwen_client.OLLAMA_MODEL,
        processing_time_ms=elapsed_ms,
        truncated=was_truncated,
    )

    # Step 5: Store in Redis
    _store_result(job_id, result)

    logger.info(
        "evaluate_llm completed — job_id=%s llm_score=%.1f "
        "confidence=%s time=%dms truncated=%s",
        job_id,
        result["scores"]["llm_score"],
        result["confidence"]["level"],
        elapsed_ms,
        was_truncated,
    )

    return result