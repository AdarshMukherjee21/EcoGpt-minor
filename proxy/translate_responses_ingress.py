"""OpenAI Responses API ingress: Responses request -> Anthropic Messages request,
Anthropic output (JSON or SSE stream) back to Responses format.

The egress direction (Anthropic -> Responses) lives in translate_request.py /
translate_response.py; this module lets Responses-native clients enter the same
pipeline. Composition note: an upstream that itself only speaks Responses is
served via translate_request on the way out, so Responses in -> Messages ->
Responses out round-trips through the pipeline unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .sse_parse import SseParser

DEFAULT_MAX_TOKENS = 4096
EFFORT_BUDGETS = {"low": 4000, "medium": 8000, "high": 16000}


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _input_items(items: Any) -> list[dict[str, Any]]:
    if items is None:
        return []
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    return [item for item in items if isinstance(item, dict)]


def _message_content(item: dict[str, Any]) -> list[dict[str, Any]]:
    content = item.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
            blocks.append({"type": "text", "text": part.get("text", "")})
    return blocks


def from_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    """Responses request -> Anthropic Messages request (model set by the caller)."""
    messages: list[dict[str, Any]] = []
    for item in _input_items(body.get("input")):
        kind = item.get("type", "message")
        if kind == "message":
            if item.get("type") == "message" and item.get("role") == "assistant":
                blocks = _message_content(item)
            else:
                blocks = _message_content(item)
            messages.append({"role": item.get("role") or "user", "content": blocks or ""})
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": item.get("call_id") or item.get("id") or "",
                            "name": item.get("name", ""),
                            "input": _parse_arguments(item.get("arguments")),
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.get("call_id") or "",
                            "content": item.get("output") or "",
                        }
                    ],
                }
            )
        elif kind == "input_image":
            url = item.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            block = _image_block(str(url))
            if block:
                messages.append({"role": "user", "content": [block]})
    messages = _merge_same_roles(messages)
    request: dict[str, Any] = {
        "max_tokens": body.get("max_output_tokens") or DEFAULT_MAX_TOKENS,
        "messages": messages,
        "model": "",
        "stream": bool(body.get("stream")),
    }
    if body.get("instructions"):
        request["system"] = body["instructions"]
    tools = _tools(body.get("tools"))
    if tools:
        request["tools"] = tools
        choice = _tool_choice(body.get("tool_choice"))
        if choice is not None:
            request["tool_choice"] = choice
    reasoning = body.get("reasoning") or {}
    budget = EFFORT_BUDGETS.get(reasoning.get("effort"))
    if budget:
        request["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if "temperature" in body and body["temperature"] is not None:
        request["temperature"] = body["temperature"]
    if "top_p" in body and body["top_p"] is not None:
        request["top_p"] = body["top_p"]
    return request


def _merge_same_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            content = message.get("content")
            if isinstance(content, list):
                if isinstance(merged[-1]["content"], list):
                    merged[-1]["content"] += content
                elif merged[-1]["content"]:
                    merged[-1]["content"] += "\n" + "".join(
                        block.get("text", "") for block in content if block.get("type") == "text"
                    )
                else:
                    merged[-1]["content"] = content
            elif content:
                merged[-1]["content"] = content
        else:
            merged.append(dict(message))
    return merged


def _image_block(url: str) -> dict[str, Any] | None:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media = header[5:].split(";", 1)[0] or "image/png"
        if data:
            return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    elif url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    mapped = []
    for tool in tools or []:
        if tool.get("type") != "function" or not tool.get("name"):
            continue
        entry: dict[str, Any] = {"name": tool["name"]}
        if tool.get("description"):
            entry["description"] = tool["description"]
        entry["input_schema"] = tool.get("parameters") or {"type": "object", "properties": {}}
        mapped.append(entry)
    return mapped


def _tool_choice(choice: Any) -> dict[str, Any] | None:
    if choice == "auto" or choice is None:
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict) and choice.get("type") == "function" and choice.get("name"):
        return {"type": "tool", "name": choice["name"]}
    return None


def usage_to_responses(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    cached = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    return {
        "input_tokens": (usage.get("input_tokens") or 0) + cached + cache_write,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": usage.get("output_tokens") or 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": (usage.get("input_tokens") or 0) + cached + cache_write + (usage.get("output_tokens") or 0),
    }


def _response_object(message: dict[str, Any], requested_model: str) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for block in message.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "type": "function_call",
                    "id": f"fc_{block.get('id', '')}",
                    "call_id": block.get("id") or "",
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                    "status": "completed",
                }
            )
    incomplete = message.get("stop_reason") == "max_tokens"
    if text_parts:
        output.append(
            {
                "type": "message",
                "id": "msg_" + str(message.get("id", "proxy")),
                "role": "assistant",
                "status": "incomplete" if incomplete else "completed",
                "content": [{"type": "output_text", "text": "".join(text_parts), "annotations": []}],
            }
        )
    output += tool_calls
    return {
        "id": "resp_" + str(message.get("id", "proxy")),
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if incomplete else "completed",
        "model": requested_model,
        "output": output,
        "usage": usage_to_responses(message.get("usage")),
    }


def to_responses_response(message: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Anthropic message (non-stream) -> Responses API response object."""
    return _response_object(message, requested_model)


class AnthropicToResponses:
    """Converts an Anthropic Messages SSE byte stream into Responses API SSE bytes."""

    def __init__(self, requested_model: str) -> None:
        self.model = requested_model
        self.parser = SseParser()
        self.response_id = "resp_proxy"
        self.message_id = "msg_proxy"
        self.created = int(time.time())
        self.started = False
        self.output_index = 0
        self.open_block_indexes: dict[int, tuple[int, str]] = {}  # anthropic index -> (output_index, kind)
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.error: str | None = None
        self._tool_meta: dict[int, dict[str, str]] = {}

    def _sse(self, event: dict[str, Any], name: str | None = None) -> bytes:
        name = name or event.get("type", "")
        return f"event: {name}\ndata: {json.dumps(event)}\n\n".encode("utf-8")

    def feed(self, chunk: bytes | str) -> bytes:
        out = b""
        for event in self.parser.feed(chunk):
            out += self._event(event)
        return out

    def _response_prefix(self) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": "in_progress",
            "model": self.model,
            "output": [],
        }

    def _event(self, event: dict[str, Any]) -> bytes:
        kind = event.get("type", "")
        if kind == "message_start":
            message = event.get("message") or {}
            self.message_id = "msg_" + str(message.get("id", "proxy"))
            self.response_id = "resp_" + str(message.get("id", "proxy"))
            self.usage.update(message.get("usage") or {})
            self.started = True
            return self._sse({"type": "response.created", "response": self._response_prefix()})
        if kind == "content_block_start":
            anthropic_index = event.get("index", 0)
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                output_index = self.output_index
                self.output_index += 1
                self.open_block_indexes[anthropic_index] = (output_index, "tool_use")
                self._remember_tool(anthropic_index, block.get("id") or "", block.get("name", ""))
                return self._sse(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": f"fc_{block.get('id', '')}",
                            "call_id": block.get("id") or "",
                            "name": block.get("name", ""),
                            "arguments": "",
                            "status": "in_progress",
                        },
                    }
                )
            output_index = self.output_index
            self.output_index += 1
            self.open_block_indexes[anthropic_index] = (output_index, "message")
            return self._sse(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "message",
                        "id": self.message_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }
            )
        if kind == "content_block_delta":
            anthropic_index = event.get("index", 0)
            entry = self.open_block_indexes.get(anthropic_index)
            delta = event.get("delta") or {}
            if entry is None:
                return b""
            output_index, block_kind = entry
            if block_kind == "message" and delta.get("type") == "text_delta":
                return self._sse(
                    {
                        "type": "response.output_text.delta",
                        "item_id": self.message_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": delta.get("text", ""),
                    }
                )
            if block_kind == "tool_use" and delta.get("type") == "input_json_delta":
                return self._sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": f"fc_{self._call_id(anthropic_index)}",
                        "output_index": output_index,
                        "delta": delta.get("partial_json", ""),
                    }
                )
            return b""
        if kind == "content_block_stop":
            anthropic_index = event.get("index", 0)
            entry = self.open_block_indexes.pop(anthropic_index, None)
            if entry is None:
                return b""
            output_index, block_kind = entry
            if block_kind == "message":
                item = {
                    "type": "message",
                    "id": self.message_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "", "annotations": []}],
                }
            else:
                item = {
                    "type": "function_call",
                    "id": f"fc_{self._call_id(anthropic_index)}",
                    "call_id": self._call_id(anthropic_index),
                    "name": self._call_name(anthropic_index),
                    "arguments": "",
                    "status": "completed",
                }
            return self._sse({"type": "response.output_item.done", "output_index": output_index, "item": item})
        if kind == "message_delta":
            self.usage.update(event.get("usage") or {})
            reason = (event.get("delta") or {}).get("stop_reason")
            if reason:
                self.stop_reason = reason
            return b""
        if kind == "error":
            self.error = (event.get("error") or {}).get("message", "Upstream error")
            return self._sse({"type": "error", "message": self.error})
        if kind == "message_stop":
            incomplete = self.stop_reason == "max_tokens"
            response = self._response_prefix()
            response["status"] = "incomplete" if incomplete else "completed"
            if incomplete:
                response["incomplete_details"] = {"reason": "max_output_tokens"}
            response["usage"] = usage_to_responses(self.usage)
            return self._sse({"type": "response.completed", "response": response})
        return b""

    def _remember_tool(self, anthropic_index: int, call_id: str, name: str) -> None:
        self._tool_meta[anthropic_index] = {"call_id": call_id, "name": name}

    def _call_id(self, anthropic_index: int) -> str:
        return self._tool_meta.get(anthropic_index, {}).get("call_id", "")

    def _call_name(self, anthropic_index: int) -> str:
        return self._tool_meta.get(anthropic_index, {}).get("name", "")
