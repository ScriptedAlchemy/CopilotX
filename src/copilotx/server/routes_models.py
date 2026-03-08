"""Model listing and health routes: /v1/models, /health."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from copilotx import __version__

router = APIRouter(tags=["Models"])


@router.get("/v1/models")
async def list_models(request: Request):
    """List available models (OpenAI format)."""
    try:
        models = await request.app.state.runtime.list_models()
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": m["id"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": m.get("vendor", "github-copilot"),
                    }
                    for m in models
                ],
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Failed to fetch models: {e}"}},
        )


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    summary = await request.app.state.runtime.health_snapshot()
    return JSONResponse(
        content={
            "status": "ok",
            "version": __version__,
            **summary,
        }
    )
