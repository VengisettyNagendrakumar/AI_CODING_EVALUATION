"""

Pydantic schemas for the Module C — Pipeline Orchestration API contract.
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator
from app.models.schemes import AnalysisResponse
from app.models.llm_models import LLMEvalResponse


# Request

class PipelineEvalRequest(BaseModel):
    submission_id: str = Field(
        ...,
        description="Unique identifier for this submission.",
        examples=["sub_001"],
    )
    language: str = Field(
        ...,
        description="Programming language of the candidate code (e.g. 'python', 'java').",
        examples=["python"],
    )
    code: str = Field(
        ...,
        description="The raw source code submitted by the candidate.",
        min_length=1,
        examples=["def hello():\n    return 'world'"],
    )
    question: str = Field(
        ...,
        description="The problem statement the candidate was asked to solve.",
        examples=["Write a function called hello that returns 'world'."],
    )
    total_test_cases: int = Field(
        default=0,
        description="Total number of runtime execution test cases.",
        ge=0,
        examples=[10],
    )
    passed_test_cases: int = Field(
        default=0,
        description="Number of runtime execution test cases passed by the candidate.",
        ge=0,
        examples=[8],
    )

    @model_validator(mode="after")
    def validate_test_cases(self) -> "PipelineEvalRequest":
        """
        Ensure passed_test_cases does not exceed total_test_cases.

        Without this check, passed=15 with total=5 would be accepted,
        and calculate_runtime_score would clamp it to 100 silently —
        an accidental backend bug becomes a perfect runtime score.
        """
        if self.total_test_cases > 0 and self.passed_test_cases > self.total_test_cases:
            raise ValueError(
                f"passed_test_cases ({self.passed_test_cases}) cannot exceed "
                f"total_test_cases ({self.total_test_cases}). "
                f"Check the test runner result before submitting."
            )
        return self


# 202 Accepted response (returned immediately after POST)

class PipelineJobAcceptedResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier — use to poll for results.")
    submission_id: str
    status: Literal["queued"] = Field(default="queued")
    error: Optional[str] = Field(default=None)


# Job Mapping Structure

class JobMapping(BaseModel):
    master_job_id: str
    static_job_id: str
    llm_job_id: str


# Programmatic Explainability

class ProgrammaticExplainability(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


# Pipeline Scores

class PipelineScores(BaseModel):
    static_score: Optional[float] = None
    llm_score: Optional[float] = None
    runtime_score: Optional[float] = None
    final_score: Optional[float] = None


# Poll Response

class PipelineStatusResponse(BaseModel):
    job_id: str
    submission_id: Optional[str] = None
    status: Literal["queued", "pending", "completed", "failed"] = Field(..., description="Job status")
    language: Optional[str] = None
    pipeline_version: Optional[str] = Field(default=None, description="Pipeline version, e.g. 'v1'.")
    aggregation_version: Optional[str] = Field(default=None, description="Aggregation version, e.g. 'v1'.")
    static_analysis: Optional[AnalysisResponse] = None
    llm_evaluation: Optional[LLMEvalResponse] = None
    scores: Optional[PipelineScores] = None
    explainability: Optional[ProgrammaticExplainability] = None
    job_mapping: Optional[JobMapping] = None
    error: Optional[str] = None