"""
GET /analyze/{job_id} — Polling endpoint for Module A.
"""

import logging
from fastapi import APIRouter, Depends
from app.dependencies import require_api_key
from celery.result import AsyncResult
from app.worker.celery_app import celery_app
from app.models.schemes import JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/analyze/{job_id}",
    dependencies=[Depends(require_api_key)],
    response_model=JobStatusResponse,
    summary="Poll job status",
    description="Polls for the status of an enqueued analysis job.",
)
async def get_result(job_id: str) -> JobStatusResponse:
    """Fetch job status from Celery backend."""
    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.state == "PENDING":
        return JobStatusResponse(job_id=job_id, status="pending")

    elif task_result.state == "SUCCESS":
        res = task_result.result
        return JobStatusResponse(
            job_id=job_id,
            submission_id=res.get("submission_id"),
            status=res.get("analysis_status"),
            language=res.get("language"),
            metrics=res.get("metrics"),
            scores=res.get("scores"),
            semgrep_available=res.get("semgrep_available", True),
            semgrep_failure_reason=res.get("semgrep_failure_reason"),
            error=res.get("error"),
        )

    elif task_result.state == "FAILURE":
        # Should not typically happen as exceptions are caught inside the task,
        # but handles unexpected worker-level crashes.
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(task_result.info),
        )

    else:
        # STARTED, RETRY, or any custom state — still in progress.
        return JobStatusResponse(job_id=job_id, status="pending")