"""Anthropic-compatible routes: /v1/messages.

Translates Anthropic format requests to OpenAI format (which is what
the Copilot backend speaks), and translates responses back.
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
    anthropic_to_openai_request,
    anthropic_to_openai_responses_request,
    openai_responses_to_anthropic_response,
    openai_responses_to_anthropic_stream,
    openai_stream_to_anthropic_stream,
    openai_to_anthropic_response,
)
from copilotx.server.app import run_with_runtime, stream_with_runtime
from copilotx.server.request_features import (
    responses_request_has_vision_input,
    responses_request_initiator,
)
from copilotx.server.runtime import CHAT_COMPLETIONS_API, RESPONSES_API
from copilotx.server.upstream_compat import (
    is_chat_completions_unsupported_for_model,
    is_responses_unsupported_for_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Anthropic"])


async def _prepend_first_chunk(
    first_chunk: bytes,
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    yield first_chunk
    async for chunk in stream:
        yield chunk


@router.post("/v1/messages")
async def messages(request: Request):
    """Anthropic-compatible messages endpoint.

    Accepts Anthropic format, translates to OpenAI, calls Copilot backend,
    and translates the response back to Anthropic format.
    """
    body = await request.json()
    model = body.get("model", "gpt-4o")
    is_stream = body.get("stream", False)

    # Log the incoming request for debugging
    logger.info(
        "Anthropic request: model=%s stream=%s max_tokens=%s tools=%d keys=%s",
        model,
        is_stream,
        body.get("max_tokens"),
        len(body.get("tools", [])),
        list(body.keys()),
    )

    # Translate Anthropic request → OpenAI request
    openai_payload = anthropic_to_openai_request(body)
    responses_payload = anthropic_to_openai_responses_request(body)
    model = openai_payload.get("model")
    vision = responses_request_has_vision_input(responses_payload)
    initiator = responses_request_initiator(responses_payload)
    runtime = request.app.state.runtime
    preferred_api = runtime.preferred_api_surface(model, CHAT_COMPLETIONS_API)

    try:
        if is_stream:
            if preferred_api == RESPONSES_API:
                try:
                    responses_payload.pop("stream", None)
                    responses_resp = await run_with_runtime(
                        request.app.state,
                        model=model,
                        operation=lambda client: client.responses(
                            deepcopy(responses_payload),
                            vision=vision,
                            initiator=initiator,
                        ),
                    )
                    runtime.mark_api_success(model, RESPONSES_API)
                    anthropic_stream = openai_responses_to_anthropic_stream(
                        responses_resp,
                        model,
                    )
                    first_chunk = await anext(anthropic_stream)
                    return sse_response(
                        _prepend_first_chunk(first_chunk, anthropic_stream)
                    )
                except Exception as responses_error:
                    if not is_responses_unsupported_for_model(responses_error):
                        raise
                    runtime.mark_api_unsupported(model, RESPONSES_API)

            try:
                # Stream: OpenAI SSE → Anthropic SSE
                openai_stream = stream_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.chat_completions_stream(
                        deepcopy(openai_payload)
                    ),
                )
                anthropic_stream = openai_stream_to_anthropic_stream(
                    openai_stream,
                    model,
                )
                first_chunk = await anext(anthropic_stream)
                runtime.mark_api_success(model, CHAT_COMPLETIONS_API)
                return sse_response(
                    _prepend_first_chunk(first_chunk, anthropic_stream)
                )
            except Exception as stream_error:
                if not is_chat_completions_unsupported_for_model(stream_error):
                    raise
                runtime.mark_api_unsupported(model, CHAT_COMPLETIONS_API)

                responses_payload.pop("stream", None)
                responses_resp = await run_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.responses(
                        deepcopy(responses_payload),
                        vision=vision,
                        initiator=initiator,
                    ),
                )
                runtime.mark_api_success(model, RESPONSES_API)
                anthropic_stream = openai_responses_to_anthropic_stream(
                    responses_resp,
                    model,
                )
                first_chunk = await anext(anthropic_stream)
                return sse_response(
                    _prepend_first_chunk(first_chunk, anthropic_stream)
                )

        if preferred_api == RESPONSES_API:
            try:
                responses_resp = await run_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.responses(
                        deepcopy(responses_payload),
                        vision=vision,
                        initiator=initiator,
                    ),
                )
                runtime.mark_api_success(model, RESPONSES_API)
                anthropic_resp = openai_responses_to_anthropic_response(
                    responses_resp,
                    model,
                )
                return JSONResponse(content=anthropic_resp)
            except Exception as responses_error:
                if not is_responses_unsupported_for_model(responses_error):
                    raise
                runtime.mark_api_unsupported(model, RESPONSES_API)

        try:
            openai_resp = await run_with_runtime(
                request.app.state,
                model=model,
                operation=lambda client: client.chat_completions(
                    deepcopy(openai_payload)
                ),
            )
            runtime.mark_api_success(model, CHAT_COMPLETIONS_API)
            anthropic_resp = openai_to_anthropic_response(openai_resp, model)
        except Exception as non_stream_error:
            if not is_chat_completions_unsupported_for_model(non_stream_error):
                raise
            runtime.mark_api_unsupported(model, CHAT_COMPLETIONS_API)

            responses_resp = await run_with_runtime(
                request.app.state,
                model=model,
                operation=lambda client: client.responses(
                    deepcopy(responses_payload),
                    vision=vision,
                    initiator=initiator,
                ),
            )
            runtime.mark_api_success(model, RESPONSES_API)
            anthropic_resp = openai_responses_to_anthropic_response(
                responses_resp,
                model,
            )
        return JSONResponse(content=anthropic_resp)
    except Exception as e:
        logger.error("Copilot backend error: %s", e)
        status_code = 502
        error_content = {
            "type": "error",
            "error": {
                "type": "upstream_error",
                "message": f"Copilot backend error: {e}",
            },
        }
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try:
                # Try to parse backend JSON error and extract message
                backend_error = json.loads(e.response.text)
                if "error" in backend_error:
                    error_content["error"]["message"] = backend_error["error"].get(
                        "message",
                        str(backend_error["error"]),
                    )
                else:
                    error_content["error"]["message"] = e.response.text[:500]
            except (json.JSONDecodeError, ValueError):
                error_content["error"]["message"] = e.response.text[:500]
        return JSONResponse(status_code=status_code, content=error_content)
