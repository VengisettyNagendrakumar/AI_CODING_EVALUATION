"""

Computes the final llm_score from parsed LLM evaluation dimensions
and assembles the complete result dictionary stored in Redis.

No normalization is needed — the prompt instructs Qwen to return
integer scores 0-100 directly. This module only applies the weighted
average to produce llm_score.

Confidence signal
-----------------
The confidence field indicates how reliable this LLM evaluation is.

It is derived from the spread (std deviation) of the 5 dimension scores:
  - Low spread  (all scores close together) → high confidence
  - High spread (scores all over the place) → low confidence

Example:
  correctness=85, logic=82, optimization=80, edge_case=83, readability=84
  → spread=1.8  → confidence="high"   (consistent evaluation)

  correctness=95, logic=30, optimization=90, edge_case=20, readability=85
  → spread=32.4 → confidence="low"    (Qwen was uncertain/inconsistent)

Confidence levels:
  "high"   — spread <= 15  (trust the score, consistent evaluation)
  "medium" — spread <= 30  (use with caution, some inconsistency)
  "low"    — spread >  30  (flag for manual review, model was inconsistent)

In production: if confidence == "low", the backend can decide to
flag the submission for human review rather than using the AI score.
"""

from __future__ import annotations

import logging
import statistics

from app.utils.config_loader import get_llm_scoring_weights

logger = logging.getLogger(__name__)

# Confidence thresholds based on std deviation of dimension scores
_HIGH_CONFIDENCE_THRESHOLD   = 15.0
_MEDIUM_CONFIDENCE_THRESHOLD = 30.0


def _compute_confidence(evaluation: dict) -> tuple[str, float]:
    """
    Compute confidence level from the spread of dimension scores.

    Returns:
        (confidence_level, score_spread)
        confidence_level: "high" | "medium" | "low"
        score_spread: standard deviation of the 5 dimension scores (float)
    """
    scores = [
        evaluation["correctness"],
        evaluation["logic"],
        evaluation["optimization"],
        evaluation["edge_case_handling"],
        evaluation["readability"],
    ]

    spread = round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0

    if spread <= _HIGH_CONFIDENCE_THRESHOLD:
        level = "high"
    elif spread <= _MEDIUM_CONFIDENCE_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    return level, spread


def compute_llm_score(evaluation: dict) -> float:
    """
    Compute the weighted average llm_score from a parsed evaluation dict.

    Weights are loaded from app/config/llm_scoring_weights.yaml.
    Dimensions: correctness, logic, optimization, edge_case_handling, readability.

    Returns:
        llm_score as a float (0.0 – 100.0), rounded to 2 decimal places.
    """
    weights = get_llm_scoring_weights()

    llm_score = (
        evaluation["correctness"]        * weights["correctness"] +
        evaluation["logic"]              * weights["logic"] +
        evaluation["optimization"]       * weights["optimization"] +
        evaluation["edge_case_handling"] * weights["edge_case_handling"] +
        evaluation["readability"]        * weights["readability"]
    )

    return round(llm_score, 2)


def build_result(
    *,
    job_id: str,
    submission_id: str,
    language: str,
    evaluation: dict,
    strengths: list[str],
    weaknesses: list[str],
    recommendations: list[str],
    raw_response: str,
    prompt_version: str,
    model: str,
    processing_time_ms: int,
    truncated: bool = False,
) -> dict:
    """
    Assemble the complete result payload for storage in Redis.

    Args:
        job_id:              Celery task / job identifier.
        submission_id:       Original submission identifier.
        language:            Programming language of the submission.
        evaluation:          Parsed dimension scores dict (int 0-100).
        strengths:           List of strength strings from LLM.
        weaknesses:          List of weakness strings from LLM.
        recommendations:     List of recommendation strings from LLM.
        raw_response:        Full raw text output from Ollama (for debugging).
        prompt_version:      Prompt version tag (e.g. "v1").
        model:               Ollama model identifier (e.g. "qwen2.5:7b").
        processing_time_ms:  Wall-clock time from task start to completion.
        truncated:           True if the submitted code was truncated before
                             sending to Qwen (code exceeded MAX_CODE_CHARS).

    Returns:
        Complete result dict ready to be JSON-serialized and stored in Redis.
    """
    llm_score = compute_llm_score(evaluation)
    confidence_level, score_spread = _compute_confidence(evaluation)

    # If code was truncated, add a note to weaknesses so backend/student knows
    if truncated:
        weaknesses = list(weaknesses) + [
            f"Note: submission exceeded the maximum evaluable length "
            f"({prompt_version}) — only the first portion was evaluated."
        ]
        logger.warning(
            "LLM result built with truncated code — job_id=%s submission_id=%s",
            job_id, submission_id,
        )

    if confidence_level == "low":
        logger.warning(
            "Low confidence LLM result — job_id=%s spread=%.1f — "
            "consider flagging for manual review",
            job_id, score_spread,
        )

    return {
        "job_id":              job_id,
        "submission_id":       submission_id,
        "status":              "completed",
        "language":            language,
        "model":               model,
        "prompt_version":      prompt_version,
        "processing_time_ms":  processing_time_ms,
        # Confidence signal — use this to decide whether to trust the score
        "confidence": {
            "level":       confidence_level,   # "high" | "medium" | "low"
            "score_spread": score_spread,       # std deviation of 5 dimension scores
        },
        # Whether submitted code was truncated before sending to LLM
        "truncated": truncated,
        "evaluation": {
            "correctness":        evaluation["correctness"],
            "logic":              evaluation["logic"],
            "optimization":       evaluation["optimization"],
            "edge_case_handling": evaluation["edge_case_handling"],
            "readability":        evaluation["readability"],
        },
        "scores": {
            "correctness_score":  float(evaluation["correctness"]),
            "logic_score":        float(evaluation["logic"]),
            "optimization_score": float(evaluation["optimization"]),
            "edge_case_score":    float(evaluation["edge_case_handling"]),
            "readability_score":  float(evaluation["readability"]),
            "llm_score":          llm_score,
        },
        "strengths":       strengths,
        "weaknesses":      weaknesses,
        "recommendations": recommendations,
        "raw_response":    raw_response,
        "error":           None,
    }