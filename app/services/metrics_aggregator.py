"""

Combines AST metrics and Semgrep findings into a single MetricsResult.

This is a pure data-assembly step with no analysis logic of its own.
"""

from app.models.schemes import MetricsResult
from app.services.ast_metrics import ASTMetrics
from app.services.semgrep_runner import SemgrepResult


def aggregate(
    ast_metrics: ASTMetrics,
    semgrep_result: SemgrepResult,
    code_size_bytes: int = 0,
    large_submission: bool = False,
) -> MetricsResult:
    """
    Merge AST-derived and Semgrep-derived metrics into a MetricsResult.

    Args:
        ast_metrics:      Output of ast_metrics.extract_metrics().
        semgrep_result:   Output of semgrep_runner.run_semgrep().
        code_size_bytes:  UTF-8 byte size of the submitted code.
        large_submission: True when submission exceeded the large threshold.

    Returns:
        MetricsResult ready for the normalisation engine.
    """
    return MetricsResult(
        # AST metrics
        cyclomatic_complexity=ast_metrics.cyclomatic_complexity,
        nesting_depth=ast_metrics.nesting_depth,
        function_length=ast_metrics.function_length,
        parameter_count=ast_metrics.parameter_count,
        # Semgrep metrics
        security_issues=semgrep_result.security_issues,
        reliability_issues=semgrep_result.reliability_issues,
        best_practice_violations=semgrep_result.best_practice_violations,
        # Submission metadata
        code_size_bytes=code_size_bytes,
        large_submission=large_submission,
    )