"""

Score aggregation and programmatic explainability report generator.

"""

from __future__ import annotations

import logging
from app.utils.config_loader import get_aggregation_weights

logger = logging.getLogger(__name__)

PIPELINE_VERSION    = "v1"
AGGREGATION_VERSION = "v1"

# Neutral score when total_test_cases == 0
# 50 = not applicable — neither rewards nor penalises
_NEUTRAL_RUNTIME_SCORE = 50.0


def calculate_runtime_score(passed: int, total: int) -> float:
    """
    Calculate the runtime test case score on a scale of 0-100.

    Args:
        passed: Number of test cases passed.
        total:  Total number of test cases.

    Returns:
        Score 0-100. Returns _NEUTRAL_RUNTIME_SCORE (50.0) when total <= 0
        to avoid gifting 40 points when test data is missing or not applicable.
    """
    if total <= 0:
        logger.info(
            "calculate_runtime_score: total_test_cases=%d — "
            "returning neutral score %.1f (not applicable).",
            total, _NEUTRAL_RUNTIME_SCORE,
        )
        return _NEUTRAL_RUNTIME_SCORE

    ratio = max(0.0, min(1.0, float(passed) / float(total)))
    return round(ratio * 100.0, 2)


def aggregate_scores(
    static_score: float | None,
    llm_score: float | None,
    runtime_score: float,
    has_syntax_error: bool = False,
    static_failed: bool = False,
) -> tuple[float, dict[str, float]]:
    """
    Aggregate individual scores into a final composite score.
    """
    config = get_aggregation_weights()
    weights = config.get("weights", {})
    caps    = config.get("caps", {})

    s_score = static_score if static_score is not None else 0.0
    l_score = llm_score    if llm_score    is not None else 0.0

    weighted_score = (
        s_score      * weights.get("static_analysis",   0.20) +
        l_score      * weights.get("llm_evaluation",    0.40) +
        runtime_score * weights.get("runtime_execution", 0.40)
    )
    final_score = round(weighted_score, 2)

    cap_reason  = None
    applied_cap = 100.0

    if has_syntax_error or static_failed:
        syntax_cap = caps.get("syntax_failure_cap", 20.0)
        if final_score > syntax_cap:
            applied_cap = min(applied_cap, syntax_cap)
            cap_reason  = "Syntax or analysis compilation error detected."

    if runtime_score == 0.0:
        zero_tests_cap = caps.get("zero_tests_passed_cap", 20.0)
        if final_score > zero_tests_cap:
            applied_cap = min(applied_cap, zero_tests_cap)
            cap_reason  = "No runtime test cases passed."

    if cap_reason is not None:
        final_score = min(final_score, applied_cap)
        logger.info("Final score capped at %.1f. Reason: %s", final_score, cap_reason)

    breakdown = {
        "static_score":  s_score,
        "llm_score":     l_score,
        "runtime_score": runtime_score,
        "final_score":   final_score,
    }

    return final_score, breakdown


def generate_explainability(
    static_result: dict | None,
    llm_result: dict | None,
    runtime_score: float,
    has_syntax_error: bool = False,
) -> dict:
    strengths       = []
    weaknesses      = []
    recommendations = []

    # --- 1. Runtime ---
    if runtime_score == 100.0:
        strengths.append("Code successfully passes all runtime unit test cases.")
    elif runtime_score == _NEUTRAL_RUNTIME_SCORE:
        strengths.append("No runtime test cases were provided — evaluation based on static and LLM analysis only.")
    elif runtime_score >= 70.0:
        strengths.append("Passes the majority of functional test cases.")
        recommendations.append("Investigate boundary conditions and edge cases to pass the remaining tests.")
    elif runtime_score > 0.0:
        weaknesses.append("Several runtime test cases failed, indicating functional errors.")
        recommendations.append("Rewrite core components to match the functional requirements.")
    else:
        weaknesses.append("Fails all functional runtime test cases.")
        recommendations.append("Ensure your implementation runs locally and executes correctly before submission.")

    # --- 2. Static ---
    if static_result and static_result.get("analysis_status") in ("completed", "degraded"):
        metrics = static_result.get("metrics", {})
        scores  = static_result.get("scores",  {})

        cc = metrics.get("cyclomatic_complexity", 0)
        if cc > 15:
            weaknesses.append(f"High cyclomatic complexity ({cc}) makes the code difficult to trace.")
            recommendations.append("Refactor complex conditional trees into smaller helper functions.")
        elif 0 < cc <= 5:
            strengths.append("Excellent control-flow structure with low complexity.")

        if static_result.get("analysis_status") == "degraded":
            weaknesses.append(
                "Static security and reliability analysis was unavailable (Semgrep not installed). "
                "Security and reliability scores may be incomplete."
            )
        else:
            sec_issues  = metrics.get("security_issues", 0)
            rel_issues  = metrics.get("reliability_issues", 0)
            bp_violations = metrics.get("best_practice_violations", 0)

            if sec_issues > 0:
                weaknesses.append(f"Security: {sec_issues} risk pattern(s) flagged.")
                recommendations.append("Review and sanitize security issues reported by the static analyzer.")
            if rel_issues > 0:
                weaknesses.append(f"Reliability: {rel_issues} potential bug(s) flagged.")
                recommendations.append("Fix null pointer, indexing, or resource leak issues.")
            if bp_violations == 0 and sec_issues == 0 and rel_issues == 0:
                strengths.append("No static lint violations, safety warnings, or code smells detected.")

        static_score = scores.get("static_score", 0.0)
        if static_score and static_score >= 85.0:
            strengths.append("Highly maintainable and clean static code architecture.")

    # --- 3. LLM ---
    if llm_result and llm_result.get("status") == "completed":
        evaluation = llm_result.get("evaluation", {})

        llm_readability = evaluation.get("readability", 0)
        if llm_readability >= 80:
            strengths.append("High readability: code is clean and uses descriptive names.")
        elif llm_readability < 50:
            weaknesses.append("Poor readability: variable naming or formatting is obscure.")
            recommendations.append("Improve naming conventions to make code self-documenting.")

        llm_optimization = evaluation.get("optimization", 0)
        if llm_optimization >= 80:
            strengths.append("Optimized performance: avoids redundant actions or allocations.")
        elif llm_optimization < 50:
            weaknesses.append("Suboptimal logic with redundant operations.")
            recommendations.append("Optimize loops and avoid unnecessary memory allocations.")

        for s in llm_result.get("strengths",       [])[:2]: strengths.append(s)
        for w in llm_result.get("weaknesses",      [])[:2]: weaknesses.append(w)
        for r in llm_result.get("recommendations", [])[:2]: recommendations.append(r)

    # --- 4. Syntax ---
    if has_syntax_error:
        weaknesses.append("Syntax error detected in the submitted code.")
        recommendations.append("Correct unclosed braces, brackets, or invalid keywords.")

    # Deduplicate preserving order
    strengths       = list(dict.fromkeys(strengths))
    weaknesses      = list(dict.fromkeys(weaknesses))
    recommendations = list(dict.fromkeys(recommendations))

    # --- 5. Summary ---
    parts = []
    if has_syntax_error:
        parts.append("The evaluation pipeline encountered syntax errors during processing.")
    else:
        parts.append("The code submission has been analyzed across static, dynamic, and logical dimensions.")

    if runtime_score == _NEUTRAL_RUNTIME_SCORE:
        parts.append("No runtime test cases were evaluated.")
    elif runtime_score == 100.0:
        parts.append("It achieves full functional correctness on test cases.")
    else:
        parts.append(f"It executes with a functional correctness score of {runtime_score}%.")

    if static_result and static_result.get("scores"):
        ss = static_result["scores"].get("static_score")
        if ss is not None:
            parts.append(f"Static analysis scored the structure at {ss}%.")

    return {
        "strengths":       strengths,
        "weaknesses":      weaknesses,
        "recommendations": recommendations,
        "summary":         " ".join(parts),
    }