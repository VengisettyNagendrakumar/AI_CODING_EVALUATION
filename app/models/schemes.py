"""

Pydantic v2 models for Module A request and response contracts.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# Request

class AnalysisRequest(BaseModel):
    """Payload sent by the client to the /analyze endpoint."""

    submission_id: str = Field(
        ...,
        description="Unique identifier for this submission.",
        examples=["sub_001"],
    )
    language: str = Field(
        ...,
        description="Programming language declared by the client (e.g. 'python').",
        examples=["python"],
    )
    code: str = Field(
        ...,
        description=(
            "Raw source code string to be analysed. "
            "No size limit is enforced — all submissions are evaluated regardless of length."
        ),
        min_length=1,
    )


# Intermediate / internal

class MetricsResult(BaseModel):
    """Raw metrics extracted from AST traversal and Semgrep analysis."""

    # AST-derived complexity metrics
    cyclomatic_complexity: int = Field(
        ..., ge=0, description="Number of linearly independent paths through the code."
    )
    nesting_depth: int = Field(
        ..., ge=0, description="Maximum depth of nested control-flow blocks."
    )
    function_length: int = Field(
        ..., ge=0, description="Maximum number of lines in any single function."
    )
    parameter_count: int = Field(
        ..., ge=0, description="Maximum number of parameters in any single function."
    )

    # Semgrep-derived quality metrics
    security_issues: int = Field(
        ..., ge=0, description="Number of security findings (ERROR severity)."
    )
    reliability_issues: int = Field(
        ..., ge=0, description="Number of reliability/correctness findings (WARNING severity)."
    )
    best_practice_violations: int = Field(
        ..., ge=0, description="Number of best-practice findings (INFO severity)."
    )

    # Submission metadata
    code_size_bytes: int = Field(
        ..., ge=0,
        description="Size of the submitted code in bytes. Informational only."
    )
    large_submission: bool = Field(
        default=False,
        description=(
            "True when the submission exceeds 50,000 characters. "
            "Semgrep analysis is still performed but may be slower."
        )
    )


# Scores

class ScoresResult(BaseModel):
    """
    Normalised per-dimension scores and the final weighted static score.

    Semgrep-derived scores (security_score, reliability_score,
    best_practice_score) are Optional because they are set to None
    when Semgrep is unavailable (analysis_status='degraded').

    Callers must check analysis_status before using these fields.
    When None, these scores must NOT be treated as perfect (100) —
    they are unknown.
    """
    complexity_score: float = Field(..., ge=0, le=100)
    maintainability_score: float = Field(..., ge=0, le=100)
    static_score: float = Field(..., ge=0, le=100, description="Final weighted composite score.")

    # None when Semgrep failed (analysis_status='degraded')
    security_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="None when Semgrep is unavailable — do not treat as 100."
    )
    reliability_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="None when Semgrep is unavailable — do not treat as 100."
    )
    best_practice_score: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="None when Semgrep is unavailable — do not treat as 100."
    )


# Response

class AnalysisResponse(BaseModel):
    """Full response returned by the /analyze endpoint."""

    submission_id: str
    language: str
    metrics: MetricsResult | None = Field(
        default=None,
        description="Raw metrics. None when analysis_status is 'failed'.",
    )
    scores: ScoresResult | None = Field(
        default=None,
        description=(
            "Normalised scores. None when analysis_status is 'failed'. "
            "When analysis_status is 'degraded', Semgrep-derived scores "
            "(security_score, reliability_score, best_practice_score) are None."
        ),
    )
    analysis_status: Literal["completed", "degraded", "failed"] = Field(
        ...,
        description=(
            "completed — all scores valid. "
            "degraded  — AST scores valid, Semgrep unavailable. "
            "failed    — entire analysis failed."
        ),
    )
    semgrep_available: bool = Field(
        default=True,
        description="False when Semgrep was unavailable during analysis."
    )
    semgrep_failure_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason when semgrep_available is False."
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error message when analysis_status is 'failed'.",
    )


class PartialMetrics(BaseModel):
    """AST metrics returned immediately upon accepting the job."""
    cyclomatic_complexity: int
    nesting_depth: int
    function_length: int
    parameter_count: int
    code_size_bytes: int
    large_submission: bool = False


class JobAcceptedResponse(BaseModel):
    """Response returned immediately when an analysis job is queued."""
    job_id: str
    submission_id: str
    status: Literal["queued"]
    partial: PartialMetrics


class JobStatusResponse(BaseModel):
    """Response returned when polling a job's status."""
    job_id: str
    submission_id: str | None = None
    status: Literal["pending", "completed", "degraded", "failed"]
    language: str | None = None
    metrics: MetricsResult | None = None
    scores: ScoresResult | None = None
    semgrep_available: bool = True
    semgrep_failure_reason: Optional[str] = None
    error: str | None = None