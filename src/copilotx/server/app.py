"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from copilotx import __version__, config
from copilotx.auth.pool import TokenPool
from copilotx.auth.token import TokenManager
from copilotx.proxy.client import CopilotClient

# ── CORS Configuration ──────────────────────────────────────────────

CORS_ORIGINS = [
    "https://polly.wang",
    "https://www.polly.wang",
    "http://127.0.0.1:1111",   # Zola dev server
    "http://localhost:1111",  # Zola dev server (localhost)
]


# ── API Key Middleware ──────────────────────────────────────────────


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key for remote requests.

    Rules:
    - no API key configured → all requests pass (backward compatible)
    - API key configured →
        - Requests from localhost pass only when COPILOTX_TRUST_LOCALHOST is truthy
        - Public paths (/health, /) → pass (health checks)
        - Other requests → require Authorization: Bearer <key>
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # CORS preflight requests must pass through (handled by CORSMiddleware)
        if request.method == "OPTIONS":
            return await call_next(request)

        # No API key configured → fully open (local mode)
        api_key = config.get_copilotx_api_key()
        if not api_key:
            return await call_next(request)

        # Public paths always accessible
        if request.url.path in config.PUBLIC_PATHS:
            return await call_next(request)

        # Localhost may be trusted for backward compatibility.
        client_host = request.client.host if request.client else ""
        if config.trust_localhost() and client_host in config.LOCALHOST_ADDRS:
            return await call_next(request)

        # Remote request → validate Bearer token
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # strip "Bearer "
        elif auth_header.startswith("bearer "):
            token = auth_header[7:]
        else:
            token = ""

        # Also accept x-api-key header (common pattern)
        if not token:
            token = request.headers.get("x-api-key", "")

        # Also accept api-key header (Azure OpenAI pattern)
        if not token:
            token = request.headers.get("api-key", "")

        if token != api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key. "
                        "Set Authorization: Bearer <your-key> header.",
                        "type": "authentication_error",
                    }
                },
            )

        return await call_next(request)


# ── Lifespan ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the CopilotClient lifecycle."""
    runtime = app.state.runtime
    if isinstance(runtime, TokenPool):
        await runtime.__aenter__()
        try:
            yield
        finally:
            await runtime.__aexit__(None, None, None)
        return

    tm: TokenManager = app.state.token_manager
    token = await tm.ensure_copilot_token()
    client = CopilotClient(token, api_base_url=tm.api_base_url)
    await client.__aenter__()
    app.state.client = client
    app.state.token_manager = tm
    try:
        yield
    finally:
        await client.__aexit__(None, None, None)


# ── App Factory ─────────────────────────────────────────────────────


def create_app(runtime: TokenManager | TokenPool) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="CopilotX",
        description="GitHub Copilot API proxy — local & remote",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    if isinstance(runtime, TokenPool):
        app.state.token_manager = None
    else:
        app.state.token_manager = runtime

    # Add CORS middleware (must be before other middlewares)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add API key middleware
    app.add_middleware(ApiKeyMiddleware)

    # Register routes
    from copilotx.server.routes_anthropic import router as anthropic_router
    from copilotx.server.routes_models import router as models_router
    from copilotx.server.routes_openai import router as openai_router
    from copilotx.server.routes_responses import router as responses_router

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.include_router(responses_router)
    app.include_router(models_router)

    return app


async def get_ready_client(app_state) -> CopilotClient:
    """Get a CopilotClient with a valid token, refreshing if needed."""
    tm: TokenManager = app_state.token_manager
    client: CopilotClient = app_state.client
    # Ensure token is fresh
    token = await tm.ensure_copilot_token()
    client.update_token(token)
    client.update_api_base(tm.api_base_url)
    return client


async def run_with_runtime(
    app_state,
    *,
    model: str | None,
    operation: Callable[[CopilotClient], Awaitable[Any]],
) -> Any:
    """Execute a non-streaming request against either the pool or legacy client."""
    runtime = app_state.runtime
    if isinstance(runtime, TokenPool):
        return await runtime.execute(model=model, operation=operation)

    client = await get_ready_client(app_state)
    return await operation(client)


async def stream_with_runtime(
    app_state,
    *,
    model: str | None,
    operation: Callable[[CopilotClient], AsyncIterator[bytes]],
) -> AsyncIterator[bytes]:
    """Stream a request against either the pool or legacy client."""
    runtime = app_state.runtime
    if isinstance(runtime, TokenPool):
        async for chunk in runtime.stream(model=model, operation=operation):
            yield chunk
        return

    client = await get_ready_client(app_state)
    async for chunk in operation(client):
        yield chunk
