"""

Applies the weighted scoring formula to normalised metrics and produces a
ScoresResult with all dimension scores and the final static_score.

Scoring formula — normal (Semgrep available)
---------------------------------------------
static_score =
    weights.maintainability * maintainability_score
  + weights.security        * security_score
  + weights.complexity      * complexity_score
  + weights.reliability     * reliability_score
  + weights.best_practices  * best_practice_score

Scoring formula — degraded (Semgrep unavailable)
-------------------------------------------------
When Semgrep is unavailable, security/reliability/best_practice scores
are None. To avoid inflating static_score with fake zeros, we redistribute
the Semgrep weights across the remaining AST-derived dimensions:

  static_score =
    (weights.maintainability + semgrep_weights * 0.5) * maintainability_score
  + (weights.complexity      + semgrep_weights * 0.5) * complexity_score

This ensures static_score still reflects real code quality (AST metrics)
without silently treating zero Semgrep findings as perfect scores.

Maintainability score
---------------------
Average of 4 AST metrics:
  cyclomatic_complexity, nesting_depth, function_length, parameter_count
"""

from typing import Optional
from app.models.schemes import MetricsResult, ScoresResult
from app.services.normalizer import normalize
from app.utils.config_loader import get_scoring_weights

_REQUIRED_WEIGHT_KEYS = {
    "maintainability",
    "security",
    "complexity",
    "reliability",
    "best_practices",
}


def _validated_weights() -> dict:
    """
    Load and validate scoring weights.

    Raises:
        KeyError:   if a required key is missing from scoring_weights.yaml.
        ValueError: if weights do not sum to 1.0.
    """
    weights = get_scoring_weights()

    missing = _REQUIRED_WEIGHT_KEYS - weights.keys()
    if missing:
        raise KeyError(
            f"scoring_weights.yaml is missing required keys: {sorted(missing)}. "
            f"Add them and ensure all weights sum to 1.0."
        )

    total = sum(weights[k] for k in _REQUIRED_WEIGHT_KEYS)
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"scoring_weights.yaml weights must sum to 1.0, but got {total:.4f}. "
            f"Current values: { {k: weights[k] for k in _REQUIRED_WEIGHT_KEYS} }. "
            f"Adjust the weights so they sum to exactly 1.0."
        )

    return weights


def compute_scores(
    metrics: MetricsResult,
    semgrep_available: bool = True,
) -> ScoresResult:
    """
    Normalise raw metrics and compute weighted final score.

    Args:
        metrics:           MetricsResult from metrics_aggregator.aggregate().
        semgrep_available: False when Semgrep failed. When False, Semgrep-derived
                           scores are set to None and their weights are redistributed
                           across AST-derived dimensions to avoid score inflation.

    Returns:
        ScoresResult with all dimension scores and static_score.
        When semgrep_available=False: security_score, reliability_score,
        best_practice_score are None.
    """
    weights = _validated_weights()

    # --- Always computed from AST (never depends on Semgrep) ---
    complexity_score = normalize("cyclomatic_complexity", metrics.cyclomatic_complexity)

    n_complexity  = normalize("cyclomatic_complexity", metrics.cyclomatic_complexity)
    n_nesting     = normalize("nesting_depth",         metrics.nesting_depth)
    n_fn_length   = normalize("function_length",       metrics.function_length)
    n_param_count = normalize("parameter_count",       metrics.parameter_count)
    maintainability_score = (n_complexity + n_nesting + n_fn_length + n_param_count) / 4.0

    if semgrep_available:
        # --- Normal path — Semgrep ran successfully ---
        security_score      = normalize("security_issues",          metrics.security_issues)
        reliability_score   = normalize("reliability_issues",        metrics.reliability_issues)
        best_practice_score = normalize("best_practice_violations",  metrics.best_practice_violations)

        static_score = (
            weights["maintainability"] * maintainability_score
            + weights["security"]       * security_score
            + weights["complexity"]     * complexity_score
            + weights["reliability"]    * reliability_score
            + weights["best_practices"] * best_practice_score
        )

        return ScoresResult(
            complexity_score=round(complexity_score, 2),
            security_score=round(security_score, 2),
            maintainability_score=round(maintainability_score, 2),
            reliability_score=round(reliability_score, 2),
            best_practice_score=round(best_practice_score, 2),
            static_score=round(static_score, 2),
        )

    else:
        # --- Degraded path — Semgrep unavailable ---
        # Semgrep-derived scores are None — do NOT use zeros.
        # Redistribute Semgrep weights across AST dimensions so
        # static_score still reflects real measured code quality.
        semgrep_weight_total = (
            weights["security"]
            + weights["reliability"]
            + weights["best_practices"]
        )
        # Split redistributed weight evenly between complexity and maintainability
        redistributed = semgrep_weight_total / 2.0

        static_score = (
            (weights["maintainability"] + redistributed) * maintainability_score
            + (weights["complexity"]    + redistributed) * complexity_score
        )

        return ScoresResult(
            complexity_score=round(complexity_score, 2),
            maintainability_score=round(maintainability_score, 2),
            static_score=round(static_score, 2),
            # Explicitly None — Semgrep unavailable
            security_score=None,
            reliability_score=None,
            best_practice_score=None,
        )