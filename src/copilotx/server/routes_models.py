"""Model listing and health routes: /v1/models, /health."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from copilotx import __version__
from copilotx.auth.pool import TokenPool
from copilotx.server.app import get_ready_client

router = APIRouter(tags=["Models"])


@router.get("/v1/models")
async def list_models(request: Request):
    """List available models (OpenAI format)."""
    try:
        runtime = request.app.state.runtime
        if isinstance(runtime, TokenPool):
            models = await runtime.list_models()
        else:
            client = await get_ready_client(request.app.state)
            models = await client.list_models()
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
    runtime = request.app.state.runtime
    if isinstance(runtime, TokenPool):
        summary = await runtime.health_snapshot()
        return JSONResponse(
            content={
                "status": "ok",
                "version": __version__,
                **summary,
            }
        )

    tm = request.app.state.token_manager
    return JSONResponse(
        content={
            "status": "ok",
            "version": __version__,
            "authenticated": tm.is_authenticated,
            "token_valid": tm.copilot_token_valid,
            "token_expires_in": tm.expires_in_seconds,
        }
    )
