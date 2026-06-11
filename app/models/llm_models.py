"""

Pydantic schemas for the Module B — LLM Evaluation API contract.

Request  : LLMEvalRequest
Response : LLMJobAcceptedResponse (202), LLMEvalResponse (200)
"""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


# Request

class LLMEvalRequest(BaseModel):
    submission_id: str = Field(
        ...,
        description="Unique identifier for this submission.",
        examples=["sub_001"],
    )
    language: str = Field(
        ...,
        description="Programming language of the candidate code (e.g. 'python', 'java').",
        examples=["java"],
    )
    question: str = Field(
        ...,
        description="The problem statement the candidate was asked to solve.",
        examples=["Find the maximum element in an array."],
    )
    candidate_code: str = Field(
        ...,
        description="The raw source code submitted by the candidate.",
        examples=["public int findMax(int[] arr) { int max = arr[0]; ... }"],
    )


# 202 Accepted response

class LLMJobAcceptedResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier — use to poll for results.")
    submission_id: str
    status: str = Field(default="queued")
    error: Optional[str] = Field(default=None)


# Evaluation dimensions (raw 0-100 scores from LLM)

class LLMEvaluation(BaseModel):
    correctness: int = Field(..., ge=0, le=100)
    logic: int = Field(..., ge=0, le=100)
    optimization: int = Field(..., ge=0, le=100)
    edge_case_handling: int = Field(..., ge=0, le=100)
    readability: int = Field(..., ge=0, le=100)


# Computed scores (weighted average)

class LLMScores(BaseModel):
    correctness_score: float
    logic_score: float
    optimization_score: float
    edge_case_score: float
    readability_score: float
    llm_score: float = Field(..., description="Final weighted LLM score (0-100).")


# Confidence signal
# Derived from the std deviation of the 5 dimension scores.
# This is a heuristic proxy, not a probabilistic confidence measure.
#   "high"   — spread <= 15: evaluation is consistent, trust the score
#   "medium" — spread <= 30: some inconsistency, use with caution
#   "low"    — spread >  30: model was inconsistent, flag for human review

class LLMConfidence(BaseModel):
    level: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "Heuristic confidence level based on score spread. "
            "high=consistent, medium=some variance, low=flag for review."
        ),
    )
    score_spread: float = Field(
        ...,
        description=(
            "Standard deviation of the 5 dimension scores. "
            "Lower = more consistent evaluation."
        ),
    )


# Full GET response (returned when job is completed/failed)

class LLMEvalResponse(BaseModel):
    job_id: str
    submission_id: str
    status: str = Field(..., description="queued | processing | completed | failed")
    language: Optional[str] = None
    model: Optional[str] = Field(
        default=None,
        description="Ollama model used, e.g. 'qwen2.5:7b'.",
    )
    prompt_version: Optional[str] = Field(
        default=None,
        description="Prompt version tag, e.g. 'v1'.",
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Wall-clock time in milliseconds.",
    )
    # Confidence signal — heuristic based on score spread
    confidence: Optional[LLMConfidence] = Field(
        default=None,
        description=(
            "Heuristic confidence signal. "
            "Check level field: low confidence should trigger manual review."
        ),
    )
    # Whether submitted code was truncated before LLM evaluation
    truncated: bool = Field(
        default=False,
        description=(
            "True when submitted code exceeded the maximum evaluable length. "
            "Only the first portion was evaluated."
        ),
    )
    evaluation: Optional[LLMEvaluation] = None
    scores: Optional[LLMScores] = None
    strengths: Optional[list[str]] = None
    weaknesses: Optional[list[str]] = None
    recommendations: Optional[list[str]] = None
    raw_response: Optional[str] = Field(
        default=None,
        description="Full raw text output from Ollama for debugging.",
    )
    error: Optional[str] = None