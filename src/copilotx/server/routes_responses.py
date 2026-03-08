"""OpenAI Responses API route: /v1/responses.

Implements the OpenAI Responses API with:
  - Vision content detection → copilot-vision-request header
  - Agent initiator detection → X-Initiator header
  - apply_patch tool patching → custom→function type conversion
  - Stream ID synchronization → fix inconsistent item IDs
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from copilotx.proxy.responses_stream import fix_responses_stream
from copilotx.proxy.streaming import sse_response
from copilotx.proxy.translator import (
    openai_chat_to_responses_response,
    openai_chat_to_responses_stream,
    openai_responses_to_chat_request,
)
from copilotx.server.app import run_with_runtime, stream_with_runtime

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Responses"])


async def _prepend_first_chunk(first_chunk: bytes, stream):
    yield first_chunk
    async for chunk in stream:
        yield chunk


def _should_fallback_to_chat_completions(exc: Exception) -> bool:
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
        or "does not support responses api" in message
    )


@router.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API endpoint.

    Supports both streaming (stream=true) and non-streaming requests.
    Applies vision detection, initiator detection, and apply_patch patching.
    """
    body = await request.json()
    normalize_responses_request(body)

    # Detect vision content and initiator role
    vision = has_vision_input(body)
    initiator = "agent" if has_agent_initiator(body) else "user"
    model = body.get("model")
    chat_payload = openai_responses_to_chat_request(body)

    # Patch apply_patch tool (custom → function type)
    patch_apply_patch_tool(body)

    try:
        if body.get("stream", False):
            try:
                raw_stream = stream_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.responses_stream(
                        deepcopy(body), vision=vision, initiator=initiator
                    ),
                )
                # Apply stream ID synchronization
                fixed_stream = fix_responses_stream(raw_stream)
                first_chunk = await anext(fixed_stream)
                return sse_response(_prepend_first_chunk(first_chunk, fixed_stream))
            except Exception as stream_error:
                if not _should_fallback_to_chat_completions(stream_error):
                    raise

                chat_result = await run_with_runtime(
                    request.app.state,
                    model=model,
                    operation=lambda client: client.chat_completions(
                        deepcopy(chat_payload)
                    ),
                )
                return sse_response(openai_chat_to_responses_stream(chat_result, body))

        try:
            result = await run_with_runtime(
                request.app.state,
                model=model,
                operation=lambda client: client.responses(
                    deepcopy(body), vision=vision, initiator=initiator
                ),
            )
            return JSONResponse(content=result)
        except Exception as non_stream_error:
            if not _should_fallback_to_chat_completions(non_stream_error):
                raise

            chat_result = await run_with_runtime(
                request.app.state,
                model=model,
                operation=lambda client: client.chat_completions(
                    deepcopy(chat_payload)
                ),
            )
            return JSONResponse(content=openai_chat_to_responses_response(chat_result, body))
    except Exception as e:
        logger.error("Responses API error: %s", e)
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


# ═══════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════


def has_vision_input(body: dict) -> bool:
    """Check if the request input contains image/vision content."""
    input_data = body.get("input")
    if not isinstance(input_data, list):
        return False

    for item in input_data:
        if not isinstance(item, dict):
            continue
        # Check message content parts for images
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "input_image",
                    "image",
                    "image_url",
                ):
                    return True
    return False


def has_agent_initiator(body: dict) -> bool:
    """Check if the last input item indicates an agent (vs user) initiator."""
    input_data = body.get("input")
    if not isinstance(input_data, list) or not input_data:
        return False

    last_item = input_data[-1]
    if not isinstance(last_item, dict):
        return False

    role = last_item.get("role", "").lower()
    item_type = last_item.get("type", "").lower()

    # Assistant messages and function-related items are agent-initiated
    if role == "assistant":
        return True
    if item_type in ("function_call", "function_call_output", "reasoning"):
        return True

    return False


def patch_apply_patch_tool(body: dict) -> None:
    """Patch custom-type apply_patch tools to function type (in-place).

    Some clients (e.g., Codex) send apply_patch as a "custom" type tool,
    but GitHub Copilot's API expects it as a "function" type tool.
    """
    tools = body.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "custom" and tool.get("name") == "apply_patch":
            tool["type"] = "function"
            tool["description"] = "Use the `apply_patch` tool to edit files"
            tool["parameters"] = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The entire contents of the apply_patch command",
                    }
                },
                "required": ["input"],
            }
            tool["strict"] = False


def normalize_responses_request(body: dict) -> None:
    """Strip known client-side metadata that Copilot rejects on /responses."""
    input_data = body.get("input")
    if not isinstance(input_data, list):
        return

    for item in input_data:
        if not isinstance(item, dict):
            continue
        # Droid mission mode can include phase metadata on input items.
        # GitHub Copilot rejects it with:
        #   Unknown parameter: 'input[N].phase'
        item.pop("phase", None)
