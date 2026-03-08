"""Helpers for parsing upstream model/API compatibility errors."""

from __future__ import annotations

import json
from typing import Any


def _error_payload(exc: Exception) -> tuple[int | None, str, dict[str, Any]]:
    response = getattr(exc, "response", None)
    if response is None:
        return None, "", {}

    text = str(getattr(response, "text", "") or "")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        payload = {}
    return response.status_code, text, payload


def is_chat_completions_unsupported_for_model(exc: Exception) -> bool:
    """Return True when upstream rejects a model on /chat/completions."""
    status_code, response_text, payload = _error_payload(exc)
    if status_code != 400:
        return False

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = str(error.get("code", ""))
    message = str(error.get("message", response_text)).lower()
    return (
        code == "unsupported_api_for_model"
        or "not accessible via the /chat/completions endpoint" in message
    )


def is_responses_unsupported_for_model(exc: Exception) -> bool:
    """Return True when upstream rejects a model on /responses."""
    status_code, response_text, payload = _error_payload(exc)
    if status_code != 400:
        return False

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = str(error.get("code", ""))
    message = str(error.get("message", response_text)).lower()
    return (
        code == "unsupported_api_for_model"
        or "does not support responses api" in message
    )
