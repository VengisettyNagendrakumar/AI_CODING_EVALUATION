"""

FastAPI shared dependencies for Module A.

API Key Authentication
-----------------------
All /api/v1/* endpoints require a valid API key passed in the
X-API-Key request header.

The expected key is read from the API_KEY environment variable.
If API_KEY is not set, the service starts but logs a loud warning —
this prevents accidentally locking yourself out in development while
making clear that it must be set in production.

Usage in routes:
    from app.dependencies import require_api_key
    from fastapi import Depends

    @router.post("/analyze", dependencies=[Depends(require_api_key)])
    async def analyze(...):
        ...

Setting the key:
    Development:  export API_KEY=dev-secret-key
    Production:   set via AWS Parameter Store / Secrets Manager / K8s secret
                  injected as environment variable at container start

The backend team's API gateway can pass this key in the header when
forwarding requests, keeping it invisible to end users.
"""

import logging
import os
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# Load API key from environment at module import time (once per process).
_API_KEY: str | None = os.environ.get("API_KEY")

if not _API_KEY:
    logger.warning(
        "API_KEY environment variable is not set. "
        "All /api/v1/* endpoints are UNPROTECTED. "
        "Set API_KEY before deploying to production."
    )


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency — validates the X-API-Key header.

    Raises HTTP 401 if the header is missing.
    Raises HTTP 403 if the key is present but incorrect.

    If API_KEY env var is not set (development mode), this dependency
    is a no-op — all requests pass through with a warning logged once
    at startup.

    Args:
        x_api_key: Value of the X-API-Key request header (injected by FastAPI).
    """
    if not _API_KEY:
        # No key configured — allow all (development mode).
        return

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )

    if x_api_key != _API_KEY:
        logger.warning("Invalid API key attempt.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )