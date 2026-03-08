"""Anthropic ↔ OpenAI format translator.

Copilot backend speaks OpenAI format natively.  This module translates:
  - Anthropic request  → OpenAI request   (inbound)
  - OpenAI response    → Anthropic response (outbound)
  - OpenAI SSE chunks  → Anthropic SSE events (streaming)

Includes full tool/function-calling support:
  - Anthropic tools definitions  → OpenAI tools/functions
  - Anthropic tool_use blocks    → OpenAI tool_calls
  - Anthropic tool_result blocks → OpenAI tool role messages
  - OpenAI tool_calls response   → Anthropic tool_use content blocks
  - OpenAI tool_calls streaming  → Anthropic tool_use SSE events
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  MODEL MAPPING: Anthropic model names → Copilot model names
# ═══════════════════════════════════════════════════════════════════

# Anthropic model names (sent by Claude Code) → Copilot-compatible names
ANTHROPIC_TO_COPILOT_MODEL_MAP = {
    # Claude Sonnet 4.5 variants
    "claude-sonnet-4-5-20250929": "claude-sonnet-4.5",
    "claude-sonnet-4.5-20250929": "claude-sonnet-4.5",
    "claude-4-5-sonnet": "claude-sonnet-4.5",
    "claude-4.5-sonnet": "claude-sonnet-4.5",
    # Claude Sonnet 4 variants
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-4-sonnet": "claude-sonnet-4",
    # Claude Opus 4.5 variants
    "claude-opus-4-5-20250929": "claude-opus-4.5",
    "claude-opus-4.5-20250929": "claude-opus-4.5",
    "claude-4-5-opus": "claude-opus-4.5",
    "claude-4.5-opus": "claude-opus-4.5",
    # Claude Opus 4.6 variants
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-opus-4.6": "claude-opus-4.6",
    "claude-4-6-opus": "claude-opus-4.6",
    "claude-4.6-opus": "claude-opus-4.6",
    # Claude Opus 4 variants
    "claude-opus-4-20250514": "claude-opus-41",
    "claude-opus-4": "claude-opus-41",
    "claude-4-opus": "claude-opus-41",
    # Claude Haiku 4.5 variants
    "claude-haiku-4-5": "claude-haiku-4.5",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "claude-4-5-haiku": "claude-haiku-4.5",
    "claude-4.5-haiku": "claude-haiku-4.5",
    # Claude 3.5 Sonnet (older naming)
    "claude-3-5-sonnet-20241022": "claude-sonnet-4",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4",
    "claude-3-5-sonnet": "claude-sonnet-4",
    "claude-3.5-sonnet": "claude-sonnet-4",
    # Claude 3 Opus (older naming)
    "claude-3-opus-20240229": "claude-opus-41",
    "claude-3-opus": "claude-opus-41",
    "claude-3.0-opus": "claude-opus-41",
    # Claude 3 Haiku
    "claude-3-haiku-20240307": "claude-haiku-4.5",
    "claude-3-haiku": "claude-haiku-4.5",
    "claude-3.0-haiku": "claude-haiku-4.5",
}


def map_anthropic_model_to_copilot(model: str) -> str:
    """Map Anthropic model names to Copilot-compatible model names.

    Claude Code sends Anthropic-style model names like 'claude-sonnet-4-5-20250929',
    but Copilot API expects names like 'claude-sonnet-4.5'.
    """
    # Direct mapping
    if model in ANTHROPIC_TO_COPILOT_MODEL_MAP:
        return ANTHROPIC_TO_COPILOT_MODEL_MAP[model]

    # If already a Copilot-compatible name (has dots like 4.5), return as-is
    if "." in model:
        return model

    # Fuzzy matching for unknown variants
    model_lower = model.lower()
    if "sonnet" in model_lower:
        if "4-5" in model_lower or "4.5" in model_lower:
            return "claude-sonnet-4.5"
        return "claude-sonnet-4"
    if "opus" in model_lower:
        if "4-6" in model_lower or "4.6" in model_lower:
            return "claude-opus-4.6"
        if "4-5" in model_lower or "4.5" in model_lower:
            return "claude-opus-4.5"
        return "claude-opus-41"
    if "haiku" in model_lower:
        return "claude-haiku-4.5"

    # Fall back to original model name (might be GPT model etc.)
    return model


# ═══════════════════════════════════════════════════════════════════
#  REQUEST: Anthropic → OpenAI
# ═══════════════════════════════════════════════════════════════════


def anthropic_to_openai_request(body: dict) -> dict:
    """Convert an Anthropic /v1/messages request to OpenAI /chat/completions format.

    Handles:
      - system messages (string or content-block list)
      - text / image content blocks
      - tool_use blocks → OpenAI assistant tool_calls
      - tool_result blocks → OpenAI tool-role messages
      - tools definitions → OpenAI tools/functions
      - tool_choice → OpenAI tool_choice
    """
    messages: list[dict[str, Any]] = []

    # System message
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic allows system as list of content blocks
            text_parts = [b["text"] for b in system if b.get("type") == "text"]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # Convert messages
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Separate tool-related blocks from regular content
            text_parts: list[dict[str, Any]] = []
            tool_use_blocks: list[dict[str, Any]] = []
            tool_result_blocks: list[dict[str, Any]] = []
            has_non_text = False

            for block in content:
                if isinstance(block, str):
                    text_parts.append({"type": "text", "text": block})
                elif block.get("type") == "text":
                    text_parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    # Anthropic image block → OpenAI image_url
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data_b64 = source.get("data", "")
                        text_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data_b64}",
                            },
                        })
                        has_non_text = True
                    elif source.get("type") == "url":
                        text_parts.append({
                            "type": "image_url",
                            "image_url": {"url": source.get("url", "")},
                        })
                        has_non_text = True
                elif block.get("type") == "tool_use":
                    tool_use_blocks.append(block)
                elif block.get("type") == "tool_result":
                    tool_result_blocks.append(block)

            # --- Handle assistant messages with tool_use blocks ---
            if role == "assistant" and tool_use_blocks:
                # Build the assistant message with tool_calls
                assistant_msg: dict[str, Any] = {"role": "assistant"}

                # Text content (may be None if assistant only calls tools)
                if text_parts:
                    text_content = "\n".join(
                        p["text"] for p in text_parts if p.get("type") == "text"
                    )
                    assistant_msg["content"] = text_content if text_content else None
                else:
                    assistant_msg["content"] = None

                # Convert tool_use blocks → OpenAI tool_calls
                assistant_msg["tool_calls"] = [
                    {
                        "id": tu.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                    for tu in tool_use_blocks
                ]
                messages.append(assistant_msg)

            # --- Handle user messages with tool_result blocks ---
            elif tool_result_blocks:
                # If there's also regular text content, add it first
                if text_parts:
                    if has_non_text or len(text_parts) > 1:
                        messages.append({"role": role, "content": text_parts})
                    else:
                        text = "\n".join(
                            p["text"] for p in text_parts if p.get("type") == "text"
                        )
                        if text:
                            messages.append({"role": role, "content": text})

                # Convert each tool_result → OpenAI tool message
                for tr in tool_result_blocks:
                    tool_content = tr.get("content", "")
                    # Anthropic tool_result content can be string or list of blocks
                    if isinstance(tool_content, list):
                        parts = []
                        for tc_block in tool_content:
                            if isinstance(tc_block, str):
                                parts.append(tc_block)
                            elif tc_block.get("type") == "text":
                                parts.append(tc_block["text"])
                        tool_content = "\n".join(parts)
                    elif not isinstance(tool_content, str):
                        tool_content = json.dumps(tool_content)

                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tool_content,
                    }
                    # Propagate error status
                    if tr.get("is_error"):
                        # OpenAI doesn't have a direct equivalent; embed in content
                        tool_msg["content"] = f"[ERROR] {tool_content}"
                    messages.append(tool_msg)

            # --- Regular content (no tool blocks) ---
            else:
                if has_non_text or len(text_parts) > 1:
                    messages.append({"role": role, "content": text_parts})
                else:
                    text = "\n".join(
                        p["text"] for p in text_parts if p.get("type") == "text"
                    )
                    messages.append({"role": role, "content": text})
        else:
            messages.append({"role": role, "content": str(content) if content else ""})

    # Map Anthropic model name to Copilot-compatible name
    anthropic_model = body.get("model", "gpt-4o")
    copilot_model = map_anthropic_model_to_copilot(anthropic_model)

    # Build OpenAI request
    openai_req: dict[str, Any] = {
        "model": copilot_model,
        "messages": messages,
    }

    # Map parameters
    if "max_tokens" in body:
        openai_req["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        openai_req["temperature"] = body["temperature"]
    if "top_p" in body:
        openai_req["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        openai_req["stop"] = body["stop_sequences"]
    if "stream" in body:
        openai_req["stream"] = body["stream"]

    # ── Tools conversion ────────────────────────────────────────
    if "tools" in body:
        openai_req["tools"] = _convert_anthropic_tools(body["tools"])
        logger.debug(
            "Converted %d Anthropic tools → OpenAI format", len(body["tools"])
        )

    # ── tool_choice conversion ──────────────────────────────────
    if "tool_choice" in body:
        openai_req["tool_choice"] = _convert_anthropic_tool_choice(
            body["tool_choice"]
        )

    return openai_req


def _convert_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tools definitions to OpenAI tools format.

    Anthropic:  {"name": ..., "description": ..., "input_schema": {...}}
    OpenAI:     {
        "type": "function",
        "function": {"name": ..., "description": ..., "parameters": {...}},
    }
    """
    openai_tools = []
    for tool in tools:
        # Handle different Anthropic tool types
        tool_type = tool.get("type", "custom")

        if tool_type in ("computer_20241022", "bash_20241022", "text_editor_20241022"):
            # Anthropic built-in tools — convert to function calls
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", tool_type),
                        "description": tool.get(
                            "description", f"Anthropic {tool_type} tool"
                        ),
                        "parameters": tool.get(
                            "input_schema", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        else:
            # Standard custom tool
            func_def: dict[str, Any] = {
                "name": tool["name"],
            }
            if "description" in tool:
                func_def["description"] = tool["description"]

            # input_schema → parameters
            schema = tool.get("input_schema", {})
            if schema:
                func_def["parameters"] = schema
            else:
                func_def["parameters"] = {"type": "object", "properties": {}}

            openai_tools.append({
                "type": "function",
                "function": func_def,
            })

    return openai_tools


def _convert_anthropic_tool_choice(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to OpenAI tool_choice.

    Anthropic:                    OpenAI:
      {"type": "auto"}       →    "auto"
      {"type": "any"}        →    "required"
      {"type": "tool", "name": X} → {"type": "function", "function": {"name": X}}
      "auto" (string)        →    "auto"
      "any"  (string)        →    "required"
      "none" (string)        →    "none"
    """
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required"
        return tool_choice  # "auto", "none"

    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "auto")
        if tc_type == "auto":
            return "auto"
        if tc_type == "any":
            return "required"
        if tc_type == "none":
            return "none"
        if tc_type == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
    return "auto"


def _convert_anthropic_tools_to_responses(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tools to OpenAI Responses tool format."""
    response_tools = []
    for tool in tools:
        tool_type = tool.get("type", "custom")
        response_tool = {
            "type": "function",
            "name": tool.get("name", tool_type),
            "description": tool.get(
                "description",
                f"Anthropic {tool_type} tool",
            ),
            "parameters": tool.get(
                "input_schema",
                {"type": "object", "properties": {}},
            ),
        }
        response_tools.append(response_tool)
    return response_tools


def _convert_anthropic_tool_choice_to_responses(tool_choice: Any) -> Any:
    """Convert Anthropic tool_choice to Responses API tool_choice."""
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required"
        return tool_choice

    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "auto")
        if tc_type == "auto":
            return "auto"
        if tc_type == "any":
            return "required"
        if tc_type == "none":
            return "none"
        if tc_type == "tool":
            return {"type": "function", "name": tool_choice["name"]}

    return "auto"


def _anthropic_tool_result_to_output_text(content: Any) -> str:
    """Flatten Anthropic tool_result content into a Responses output string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    if isinstance(content, (dict, list)):
        return json.dumps(content)
    return str(content)


def anthropic_to_openai_responses_request(body: dict) -> dict[str, Any]:
    """Convert Anthropic /v1/messages requests to OpenAI Responses format."""
    input_items: list[dict[str, Any]] = []

    system = body.get("system")
    if system:
        if isinstance(system, str):
            input_items.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system}],
                }
            )
        elif isinstance(system, list):
            text_parts = [b["text"] for b in system if b.get("type") == "text"]
            if text_parts:
                input_items.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "\n".join(text_parts),
                            }
                        ],
                    }
                )

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content")

        if isinstance(content, str):
            input_items.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
            continue

        if not isinstance(content, list):
            continue

        message_content: list[dict[str, Any]] = []
        assistant_tool_uses: list[dict[str, Any]] = []
        user_tool_results: list[dict[str, Any]] = []

        for block in content:
            if isinstance(block, str):
                message_content.append({"type": "input_text", "text": block})
                continue

            block_type = block.get("type")
            if block_type == "text":
                message_content.append(
                    {"type": "input_text", "text": block.get("text", "")}
                )
            elif block_type == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data_b64 = source.get("data", "")
                    message_content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{media_type};base64,{data_b64}",
                        }
                    )
                elif source.get("type") == "url":
                    message_content.append(
                        {
                            "type": "input_image",
                            "image_url": source.get("url", ""),
                        }
                    )
            elif block_type == "tool_use" and role == "assistant":
                assistant_tool_uses.append(block)
            elif block_type == "tool_result" and role == "user":
                user_tool_results.append(block)

        if message_content:
            input_items.append({"role": role, "content": message_content})

        for tool_use in assistant_tool_uses:
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": tool_use.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "name": tool_use.get("name", ""),
                    "arguments": json.dumps(tool_use.get("input", {})),
                }
            )

        for tool_result in user_tool_results:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_result.get("tool_use_id", ""),
                    "output": _anthropic_tool_result_to_output_text(
                        tool_result.get("content", "")
                    ),
                }
            )

    anthropic_model = body.get("model", "gpt-4o")
    copilot_model = map_anthropic_model_to_copilot(anthropic_model)

    responses_req: dict[str, Any] = {
        "model": copilot_model,
        "input": input_items,
    }
    if "max_tokens" in body:
        responses_req["max_output_tokens"] = body["max_tokens"]
    if "temperature" in body:
        responses_req["temperature"] = body["temperature"]
    if "top_p" in body:
        responses_req["top_p"] = body["top_p"]
    if "stream" in body:
        responses_req["stream"] = body["stream"]
    if "tools" in body:
        responses_req["tools"] = _convert_anthropic_tools_to_responses(body["tools"])
    if "tool_choice" in body:
        responses_req["tool_choice"] = _convert_anthropic_tool_choice_to_responses(
            body["tool_choice"]
        )

    return responses_req


def _openai_chat_content_to_responses_parts(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI chat-completions message content to Responses input parts."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return []

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "input_text", "text": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            parts.append({"type": "input_text", "text": str(item.get("text", ""))})
        elif item_type == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = str(image_url)
            if url:
                parts.append({"type": "input_image", "image_url": url})

    return parts


def _openai_chat_tool_result_to_output_text(content: Any) -> str:
    """Flatten OpenAI tool message content into Responses output text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))

    return "\n".join(part for part in text_parts if part)


def _openai_chat_tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat-completions tool definitions to Responses tool definitions."""
    responses_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "custom":
            responses_tool = {
                "type": "function",
                "name": tool.get("name", ""),
                "parameters": tool.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            if "description" in tool:
                responses_tool["description"] = tool["description"]
            responses_tools.append(responses_tool)
            continue

        function = tool.get("function", {}) if tool.get("type") == "function" else {}
        if not function:
            continue

        responses_tool = {
            "type": "function",
            "name": function.get("name", ""),
            "parameters": function.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        if "description" in function:
            responses_tool["description"] = function["description"]
        responses_tools.append(responses_tool)

    return responses_tools


def _openai_chat_tool_choice_to_responses(tool_choice: Any) -> Any:
    """Convert chat-completions tool_choice to Responses format."""
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function", {})
        return {"type": "function", "name": function.get("name", "")}
    return tool_choice


def _responses_input_part_to_openai_chat_part(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Responses content part to OpenAI chat-completions content."""
    part_type = part.get("type")
    if part_type == "input_text":
        return {"type": "text", "text": str(part.get("text", ""))}

    if part_type in ("input_image", "image", "image_url"):
        image_url = part.get("image_url") or part.get("url") or part.get("image")
        if isinstance(image_url, dict):
            image_url = image_url.get("url", "")
        if image_url:
            return {"type": "image_url", "image_url": {"url": str(image_url)}}

    return None


def _responses_input_item_to_openai_chat_content(item: dict[str, Any]) -> Any:
    """Convert a Responses input item to OpenAI chat-completions content."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chat_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        converted = _responses_input_part_to_openai_chat_part(part)
        if converted is not None:
            chat_parts.append(converted)

    if not chat_parts:
        return ""
    if len(chat_parts) == 1 and chat_parts[0].get("type") == "text":
        return chat_parts[0]["text"]
    return chat_parts


def _responses_tools_to_openai_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses tool definitions to chat-completions tool definitions."""
    chat_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        function: dict[str, Any] = {
            "name": tool.get("name", ""),
            "parameters": tool.get(
                "parameters",
                {"type": "object", "properties": {}},
            ),
        }
        if "description" in tool:
            function["description"] = tool["description"]

        chat_tools.append({"type": "function", "function": function})

    return chat_tools


def _responses_tool_choice_to_openai_chat(tool_choice: Any) -> Any:
    """Convert Responses tool_choice to chat-completions format."""
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name", "")},
        }
    return tool_choice


def openai_responses_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Responses request to chat-completions format."""
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if not pending_tool_calls:
            return
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": pending_tool_calls.copy(),
            }
        )
        pending_tool_calls.clear()

    for item in body.get("input", []):
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": item.get("call_id")
                    or item.get("id")
                    or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )
            continue

        if item_type == "function_call_output":
            flush_tool_calls()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": str(item.get("output", "")),
                }
            )
            continue

        flush_tool_calls()
        role = item.get("role")
        if not role:
            continue
        messages.append(
            {
                "role": role,
                "content": _responses_input_item_to_openai_chat_content(item),
            }
        )

    flush_tool_calls()

    chat_req: dict[str, Any] = {
        "model": body.get("model", "gpt-4o"),
        "messages": messages or [{"role": "user", "content": ""}],
    }
    if "temperature" in body:
        chat_req["temperature"] = body["temperature"]
    if "top_p" in body:
        chat_req["top_p"] = body["top_p"]
    if "max_output_tokens" in body:
        chat_req["max_completion_tokens"] = body["max_output_tokens"]
    if "tools" in body:
        chat_req["tools"] = _responses_tools_to_openai_chat_tools(body["tools"])
    if "tool_choice" in body:
        chat_req["tool_choice"] = _responses_tool_choice_to_openai_chat(
            body["tool_choice"]
        )
    if "parallel_tool_calls" in body:
        chat_req["parallel_tool_calls"] = body["parallel_tool_calls"]

    return chat_req


def openai_chat_to_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI chat-completions requests to Responses requests."""
    input_items: list[dict[str, Any]] = []

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": _openai_chat_tool_result_to_output_text(content),
                }
            )
            continue

        message_content = _openai_chat_content_to_responses_parts(content)
        if message_content:
            input_items.append({"role": role, "content": message_content})

        if role != "assistant":
            continue

        for tool_call in msg.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", "{}"),
                }
            )

    responses_req: dict[str, Any] = {
        "model": body.get("model", "gpt-4o"),
        "input": input_items,
    }
    if "temperature" in body:
        responses_req["temperature"] = body["temperature"]
    if "top_p" in body:
        responses_req["top_p"] = body["top_p"]
    if "stream" in body:
        responses_req["stream"] = body["stream"]
    if "max_completion_tokens" in body:
        responses_req["max_output_tokens"] = body["max_completion_tokens"]
    elif "max_tokens" in body:
        responses_req["max_output_tokens"] = body["max_tokens"]
    if "tools" in body:
        responses_req["tools"] = _openai_chat_tools_to_responses(body["tools"])
    if "tool_choice" in body:
        responses_req["tool_choice"] = _openai_chat_tool_choice_to_responses(
            body["tool_choice"]
        )

    return responses_req


# ═══════════════════════════════════════════════════════════════════
#  RESPONSE: OpenAI → Anthropic (non-streaming)
# ═══════════════════════════════════════════════════════════════════


def openai_to_anthropic_response(openai_resp: dict, model: str) -> dict:
    """Convert an OpenAI chat completion response to Anthropic /v1/messages format.

    Handles text content, tool_calls, and mixed responses.

    IMPORTANT: Copilot backend may split text and tool_calls into separate choices:
      choices[0] = {"message": {"content": "text..."}, "finish_reason": "tool_calls"}
      choices[1] = {"message": {"tool_calls": [...]},   "finish_reason": "tool_calls"}
    We must merge ALL choices to build the complete Anthropic response.
    """
    choices = openai_resp.get("choices", [{}])

    # Merge content and tool_calls from ALL choices
    # (Copilot backend splits them into separate choices)
    content_text = ""
    all_tool_calls: list[dict] = []
    finish_reason = "end_turn"

    for choice in choices:
        message = choice.get("message", {})

        # Collect text content
        text = message.get("content")
        if text:
            if content_text:
                content_text += "\n" + text
            else:
                content_text = text

        # Collect tool_calls
        tc_list = message.get("tool_calls")
        if tc_list:
            all_tool_calls.extend(tc_list)

        # Use the most specific finish_reason
        fr = choice.get("finish_reason")
        if fr == "tool_calls":
            finish_reason = "tool_calls"
        elif fr and finish_reason not in ("tool_calls",):
            finish_reason = fr

    logger.debug(
        "OpenAI response: %d choices, text=%d chars, tool_calls=%d, finish=%s",
        len(choices), len(content_text), len(all_tool_calls), finish_reason,
    )

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []

    # Add text block if present
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    # Convert OpenAI tool_calls → Anthropic tool_use blocks
    if all_tool_calls:
        for tc in all_tool_calls:
            func = tc.get("function", {})
            # Parse arguments JSON string → dict
            try:
                tool_input = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tool_input = {}

            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": tool_input,
            })

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    # Map finish_reason
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
        "tool_calls": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # Usage
    usage = openai_resp.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def openai_responses_to_anthropic_response(
    response: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Convert an OpenAI Responses API response to Anthropic format."""
    content_blocks: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            text_parts = []
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(str(part.get("text", "")))
            if text_parts:
                content_blocks.append({"type": "text", "text": "".join(text_parts)})
        elif item_type == "function_call":
            try:
                tool_input = json.loads(item.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                tool_input = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": item.get("name", ""),
                    "input": tool_input,
                }
            )
            stop_reason = "tool_use"

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    incomplete = response.get("incomplete_details") or {}
    if stop_reason != "tool_use" and incomplete.get("reason") == "max_output_tokens":
        stop_reason = "max_tokens"

    usage = response.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
    }


def openai_responses_to_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Responses API response to chat-completions format."""
    message_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    message_text_parts.append(str(part.get("text", "")))
        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id")
                    or item.get("id")
                    or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )

    finish_reason = "stop"
    incomplete = response.get("incomplete_details") or {}
    if tool_calls:
        finish_reason = "tool_calls"
    elif incomplete.get("reason") == "max_output_tokens":
        finish_reason = "length"

    content = "".join(message_text_parts)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = response.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _openai_chat_sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def _responses_sse_event(event: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def _openai_chat_message_text(content: Any) -> str:
    """Flatten a chat-completions assistant message content into plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text_parts.append(str(part.get("text", "")))

    return "".join(text_parts)


def openai_chat_to_responses_response(
    response: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Convert a chat-completions response to OpenAI Responses format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    finish_reason = "stop"

    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if isinstance(message, dict):
            content = _openai_chat_message_text(message.get("content"))
            if content:
                text_parts.append(content)
            for tool_call in message.get("tool_calls", []):
                if isinstance(tool_call, dict):
                    tool_calls.append(tool_call)

        choice_finish = str(choice.get("finish_reason") or "")
        if choice_finish == "tool_calls":
            finish_reason = choice_finish
        elif choice_finish and finish_reason != "tool_calls":
            finish_reason = choice_finish

    output: list[dict[str, Any]] = []
    joined_text = "".join(text_parts)
    if joined_text or not tool_calls:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": joined_text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        output.append(
            {
                "id": call_id,
                "type": "function_call",
                "call_id": call_id,
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )

    usage = response.get("usage", {})
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    text_cfg = request.get("text") if isinstance(request.get("text"), dict) else {}

    result: dict[str, Any] = {
        "id": response.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
        "object": "response",
        "created_at": int(response.get("created", time.time())),
        "model": response.get("model", request.get("model", "")),
        "output": output,
        "parallel_tool_calls": request.get("parallel_tool_calls", bool(tool_calls)),
        "previous_response_id": request.get("previous_response_id"),
        "status": "completed",
        "store": bool(request.get("store", False)),
        "text": {
            "format": {"type": "text"},
            "verbosity": text_cfg.get("verbosity", "medium"),
        },
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools", []),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if finish_reason == "length":
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    else:
        result["incomplete_details"] = None
    if "prompt_cache_retention" in request:
        result["prompt_cache_retention"] = request["prompt_cache_retention"]
    if "safety_identifier" in request:
        result["safety_identifier"] = request["safety_identifier"]

    return result


async def openai_chat_to_responses_stream(
    response: dict[str, Any],
    request: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Emit a Responses SSE stream from a completed chat-completions response."""
    completed = openai_chat_to_responses_response(response, request)
    in_progress = dict(completed)
    in_progress["status"] = "in_progress"
    in_progress["output"] = []

    sequence_number = 0
    yield _responses_sse_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": in_progress,
        },
    )
    sequence_number += 1
    yield _responses_sse_event(
        "response.in_progress",
        {
            "type": "response.in_progress",
            "sequence_number": sequence_number,
            "response": in_progress,
        },
    )
    sequence_number += 1

    for output_index, item in enumerate(completed.get("output", [])):
        if item.get("type") == "message":
            message_item = {
                "id": item["id"],
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "phase": "final_answer",
                "content": [],
            }
            yield _responses_sse_event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": message_item,
                },
            )
            sequence_number += 1

            part = {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            }
            yield _responses_sse_event(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item_id": item["id"],
                    "content_index": 0,
                    "part": part,
                },
            )
            sequence_number += 1

            text = ""
            content = item.get("content", [])
            if content:
                text = str(content[0].get("text", ""))
            if text:
                yield _responses_sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item["id"],
                        "content_index": 0,
                        "delta": text,
                    },
                )
                sequence_number += 1
                yield _responses_sse_event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item["id"],
                        "content_index": 0,
                        "text": text,
                    },
                )
                sequence_number += 1

            completed_part = dict(part)
            completed_part["text"] = text
            yield _responses_sse_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item_id": item["id"],
                    "content_index": 0,
                    "part": completed_part,
                },
            )
            sequence_number += 1
            yield _responses_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": item,
                },
            )
            sequence_number += 1
            continue

        yield _responses_sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": sequence_number,
                "output_index": output_index,
                "item": item,
            },
        )
        sequence_number += 1
        yield _responses_sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": sequence_number,
                "output_index": output_index,
                "item": item,
            },
        )
        sequence_number += 1

    yield _responses_sse_event(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": completed,
        },
    )


async def openai_responses_to_chat_stream(
    response: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Emit a chat-completions SSE stream from a completed Responses response."""
    chat_response = openai_responses_to_chat_response(response)
    chunk_base = {
        "id": chat_response["id"],
        "object": "chat.completion.chunk",
        "created": chat_response["created"],
        "model": chat_response["model"],
    }

    yield _openai_chat_sse_event(
        {
            **chunk_base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    message = chat_response["choices"][0]["message"]
    content = message.get("content")
    if content:
        yield _openai_chat_sse_event(
            {
                **chunk_base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
        )

    for index, tool_call in enumerate(message.get("tool_calls", [])):
        yield _openai_chat_sse_event(
            {
                **chunk_base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": tool_call["function"],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield _openai_chat_sse_event(
        {
            **chunk_base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": chat_response["choices"][0]["finish_reason"],
                }
            ],
        }
    )
    yield b"data: [DONE]\n\n"


async def openai_responses_to_anthropic_stream(
    response: dict[str, Any],
    model: str,
) -> AsyncIterator[bytes]:
    """Emit an Anthropic SSE stream from a completed Responses API response."""
    anthropic_response = openai_responses_to_anthropic_response(response, model)
    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": anthropic_response["id"],
                "type": "message",
                "role": "assistant",
                "model": anthropic_response["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": anthropic_response["usage"]["input_tokens"],
                    "output_tokens": 0,
                },
            },
        },
    )

    for index, block in enumerate(anthropic_response["content"]):
        if block["type"] == "text":
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if block.get("text"):
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": block["text"],
                        },
                    },
                )
            yield _sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
            continue

        yield _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                },
            },
        )
        if block.get("input"):
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            )
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )

    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": anthropic_response["stop_reason"],
                "stop_sequence": None,
            },
            "usage": {
                "output_tokens": anthropic_response["usage"]["output_tokens"],
            },
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


# ═══════════════════════════════════════════════════════════════════
#  STREAMING: OpenAI SSE → Anthropic SSE
# ═══════════════════════════════════════════════════════════════════


async def openai_stream_to_anthropic_stream(
    openai_lines: AsyncIterator[bytes],
    model: str,
) -> AsyncIterator[bytes]:
    """Translate OpenAI SSE stream to Anthropic SSE stream format.

    Anthropic streaming protocol:
      event: message_start       → message metadata
      event: content_block_start → start of content block
      event: content_block_delta → incremental text or tool input JSON
      event: content_block_stop  → end of content block
      event: message_delta       → stop reason + usage
      event: message_stop        → end of message

    Handles both text content and tool_calls streaming:
      - OpenAI delta.content       → Anthropic text_delta
      - OpenAI delta.tool_calls    → Anthropic tool_use content blocks
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    sent_start = False
    sent_text_block_start = False

    # Track tool call blocks: index → {id, name, block_index, started}
    tool_call_trackers: dict[int, dict[str, Any]] = {}
    # Next content_block index (0 = text, 1+ = tool_use)
    next_block_index = 0
    # Track the text block index
    text_block_index = 0
    # Accumulated finish_reason
    finish_reason = "end_turn"

    async for raw_line in openai_lines:
        line = raw_line.decode("utf-8").strip()

        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:]  # strip "data: "
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Emit message_start once
        if not sent_start:
            start_event = {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
            yield _sse_event("message_start", start_event)
            sent_start = True

        # Extract delta from ALL choices (Copilot may split text/tool_calls
        # into separate choices with different indices)
        content = None
        tool_calls = None
        chunk_finish = None

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            # Collect text content from any choice
            c = delta.get("content")
            if c:
                content = c

            # Collect tool_calls from any choice
            tc = delta.get("tool_calls")
            if tc:
                tool_calls = tc

            # Track finish reason from any choice
            fr = choice.get("finish_reason")
            if fr:
                chunk_finish = fr

        # Track finish reason
        if chunk_finish:
            if chunk_finish == "tool_calls":
                finish_reason = "tool_use"
            elif chunk_finish == "length":
                finish_reason = "max_tokens"
            else:
                finish_reason = "end_turn"

        # ── Handle text content ────────────────────────────────
        if content:
            if not sent_text_block_start:
                text_block_index = next_block_index
                next_block_index += 1
                block_start = {
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""},
                }
                yield _sse_event("content_block_start", block_start)
                sent_text_block_start = True

            block_delta = {
                "type": "content_block_delta",
                "index": text_block_index,
                "delta": {"type": "text_delta", "text": content},
            }
            yield _sse_event("content_block_delta", block_delta)

        # ── Handle tool_calls streaming ────────────────────────
        if tool_calls:
            for tc_delta in tool_calls:
                tc_index = tc_delta.get("index", 0)
                tc_id = tc_delta.get("id")
                tc_func = tc_delta.get("function", {})
                tc_name = tc_func.get("name")
                tc_args = tc_func.get("arguments", "")

                if tc_index not in tool_call_trackers:
                    # Close text block first if still open
                    if sent_text_block_start and not any(
                        t.get("text_closed") for t in tool_call_trackers.values()
                    ) and not tool_call_trackers:
                        yield _sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": text_block_index},
                        )

                    # New tool call — create tracker and emit content_block_start
                    block_idx = next_block_index
                    next_block_index += 1

                    tool_id = tc_id or f"toolu_{uuid.uuid4().hex[:24]}"
                    tool_name = tc_name or ""

                    tool_call_trackers[tc_index] = {
                        "id": tool_id,
                        "name": tool_name,
                        "block_index": block_idx,
                        "started": True,
                        "text_closed": True,
                    }

                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": {},
                        },
                    })
                else:
                    # Update existing tracker with name if provided
                    tracker = tool_call_trackers[tc_index]
                    if tc_id and not tracker["id"]:
                        tracker["id"] = tc_id
                    if tc_name and not tracker["name"]:
                        tracker["name"] = tc_name

                # Emit argument deltas as input_json_delta
                if tc_args:
                    tracker = tool_call_trackers[tc_index]
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tracker["block_index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc_args,
                        },
                    })

        # Track usage if present
        if "usage" in chunk:
            input_tokens = chunk["usage"].get("prompt_tokens", input_tokens)
            output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

    # ── Finalize — close all open blocks + message ─────────────

    # Close text block if it was opened and no tool calls closed it
    if sent_text_block_start and not tool_call_trackers:
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": text_block_index},
        )

    # Close all tool call blocks
    for _tc_idx, tracker in sorted(tool_call_trackers.items()):
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": tracker["block_index"]},
        )

    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": finish_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


def _sse_event(event_type: str, data: dict) -> bytes:
    """Format a single Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
