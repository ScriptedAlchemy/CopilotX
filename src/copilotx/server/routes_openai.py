"""OpenAI-compatible routes: /v1/chat/completions.

This is nearly a direct passthrough to api.githubcopilot.com since
the Copilot backend already speaks OpenAI format.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from copilotx.proxy.streaming import sse_response
from copilotx.proxy.translator import (
    openai_chat_to_responses_request,
    openai_responses_to_chat_response,
    openai_responses_to_chat_stream,
)
from copilotx.server.app import run_with_runtime, stream_with_runtime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI"])


async def _prepend_first_chunk(
    first_chunk: bytes,
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    yield first_chunk
    async for chunk in stream:
        yield chunk


def _should_fallback_to_responses(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is None or response.status_code != 400:
        return False

    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, ValueError):
        payload = {}

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = str(error.get("code", ""))
    message = str(error.get("message", response.text)).lower()
    return (
        code == "unsupported_api_for_model"
        or "not accessible via the /chat/completions endpoint" in message
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=true) and non-streaming requests.
    """
    body = await request.json()
    model = body.get("model")
    responses_payload = openai_chat_to_responses_request(body)

    try:
        if body.get("stream", False):
            try:
                openai_stream = stream_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.chat_completions_stream(
                        deepcopy(body)
                    ),
                )
                first_chunk = await anext(openai_stream)
                return sse_response(_prepend_first_chunk(first_chunk, openai_stream))
            except Exception as stream_error:
                if not _should_fallback_to_responses(stream_error):
                    raise

                responses_payload.pop("stream", None)
                responses_result = await run_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.responses(
                        deepcopy(responses_payload),
                        vision=False,
                        initiator="user",
                    ),
                )
                return sse_response(openai_responses_to_chat_stream(responses_result))

        try:
            result = await run_with_runtime(
                request.app.state,
                model=model,
                operation=lambda client: client.chat_completions(deepcopy(body)),
            )
            return JSONResponse(content=result)
        except Exception as non_stream_error:
            if not _should_fallback_to_responses(non_stream_error):
                raise

            responses_result = await run_with_runtime(
                request.app.state,
                model=model,
                operation=lambda client: client.responses(
                    deepcopy(responses_payload),
                    vision=False,
                    initiator="user",
                ),
            )
            return JSONResponse(content=openai_responses_to_chat_response(responses_result))
    except Exception as e:
        logger.error("Chat completions error: %s", e)
        status_code = 502
        error_content = {
            "error": {
                "message": f"Copilot backend error: {e}",
                "type": "upstream_error",
            }
        }
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                # Try to parse and forward the backend's JSON error
                error_content = json.loads(e.response.text)
            except (json.JSONDecodeError, ValueError):
                error_content["error"]["message"] = e.response.text[:500]
        return JSONResponse(status_code=status_code, content=error_content)
