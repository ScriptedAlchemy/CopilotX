"""Helpers for deriving request features from OpenAI Responses payloads."""

from __future__ import annotations

from typing import Any


def responses_request_has_vision_input(body: dict[str, Any]) -> bool:
    """Check whether a Responses payload contains image input."""
    input_data = body.get("input")
    if not isinstance(input_data, list):
        return False

    for item in input_data:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "input_image",
                "image",
                "image_url",
            ):
                return True
    return False


def responses_request_initiator(body: dict[str, Any]) -> str:
    """Infer the initiator header for a Responses payload."""
    input_data = body.get("input")
    if not isinstance(input_data, list) or not input_data:
        return "user"

    last_item = input_data[-1]
    if not isinstance(last_item, dict):
        return "user"

    role = str(last_item.get("role", "")).lower()
    item_type = str(last_item.get("type", "")).lower()
    if role == "assistant":
        return "agent"
    if item_type in ("function_call", "function_call_output", "reasoning"):
        return "agent"
    return "user"
