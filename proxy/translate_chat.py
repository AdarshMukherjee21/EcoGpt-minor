"""OpenAI Chat Completions <-> Anthropic Messages translation.

Ingress direction: a /v1/chat/completions request is converted to an Anthropic
Messages request, run through the standard pipeline, and the Anthropic output
(JSON or SSE byte stream) is converted back to Chat Completions format here.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .sse_parse import SseParser

DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS = {
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "end_turn": "stop",
    "refusal": "stop",
}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text", "input_text")
        )
    return ""


def _image_block(url: str) -> dict[str, Any] | None:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media = header[5:].split(";", 1)[0] or "image/png"
        if data:
            return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
        return None
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _part_blocks(part: dict[str, Any]) -> list[dict[str, Any]]:
    kind = part.get("type")
    if kind == "text":
        return [{"type": "text", "text": part.get("text", "")}]
    if kind in ("input_text", "output_text"):
        return [{"type": "text", "text": part.get("text", "")}]
    if kind == "image_url":
        block = _image_block((part.get("image_url") or {}).get("url", ""))
        return [block] if block else []
    return []


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict):
            blocks += _part_blocks(part)
    return blocks


def _tool_use_blocks(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    blocks = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": function.get("name", ""),
                "input": arguments if isinstance(arguments, dict) else {},
            }
        )
    return blocks


def from_chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    """Chat Completions request -> Anthropic Messages request (model set by the caller)."""
    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []
    for message in body.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id") or "",
                            "content": _text_of(content),
                        }
                    ],
                }
            )
            continue
        blocks = _content_blocks(content)
        if role == "assistant":
            blocks += _tool_use_blocks(message.get("tool_calls"))
        # Consecutive same-role messages (e.g. tool result after user text) are
        # merged; Anthropic is happiest with alternating roles.
        if messages and messages[-1].get("role") == role and blocks:
            messages[-1]["content"] += blocks
        else:
            messages.append({"role": role, "content": blocks})
    request: dict[str, Any] = {
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS,
        "messages": messages,
        "model": "",
        "stream": bool(body.get("stream")),
    }
    if system_parts:
        request["system"] = "\n\n".join(system_parts)
    tools = _tools(body.get("tools"))
    if tools:
        request["tools"] = tools
        choice = _tool_choice(body.get("tool_choice"))
        if choice is not None:
            request["tool_choice"] = choice
    if body.get("stop"):
        stop = body["stop"]
        request["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if "temperature" in body and body["temperature"] is not None:
        request["temperature"] = body["temperature"]
    if "top_p" in body and body["top_p"] is not None:
        request["top_p"] = body["top_p"]
    return request


def _tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    mapped = []
    for tool in tools or []:
        function = tool.get("function") or tool
        if tool.get("type") not in (None, "function") or not function.get("name"):
            continue
        entry: dict[str, Any] = {"name": function["name"]}
        if function.get("description"):
            entry["description"] = function["description"]
        entry["input_schema"] = function.get("parameters") or {"type": "object", "properties": {}}
        mapped.append(entry)
    return mapped


def _tool_choice(choice: Any) -> dict[str, Any] | None:
    if choice == "auto" or choice is None:
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict):
        function = choice.get("function") or choice
        if function.get("name"):
            return {"type": "tool", "name": function["name"]}
    return None


def to_chat_completions(message: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Anthropic message (non-stream) -> Chat Completions response."""
    content_parts: list[str] = []
    tool_calls = []
    for block in message.get("content") or []:
        if block.get("type") == "text":
            content_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input", {}))},
                }
            )
    usage = message.get("usage") or {}
    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    return {
        "id": "chatcmpl-" + str(message.get("id", "proxy")),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": _STOP_REASONS.get(message.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def chat_error(status: int, message: str, kind: str = "api_error") -> dict[str, Any]:
    code = {"invalid_request_error": "invalid_request_error", "not_found_error": "not_found_error"}.get(kind, "api_error")
    return {"error": {"message": message, "type": code, "param": None, "code": code}}


class AnthropicToChat:
    """Converts an Anthropic Messages SSE byte stream into Chat Completions SSE bytes."""

    def __init__(self, requested_model: str) -> None:
        self.model = requested_model
        self.parser = SseParser()
        self.completion_id = "chatcmpl-proxy"
        self.created = int(time.time())
        self.started = False
        self.tool_indexes: dict[int, int] = {}
        self.usage: dict[str, Any] = {}
        self.stop_reason = "stop"
        self.error: str | None = None

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
        data = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data)}\n\n".encode("utf-8")

    def feed(self, chunk: bytes | str) -> bytes:
        out = b""
        for event in self.parser.feed(chunk):
            out += self._event(event)
        return out

    def _event(self, event: dict[str, Any]) -> bytes:
        kind = event.get("type", "")
        if kind == "message_start":
            message = event.get("message") or {}
            self.completion_id = "chatcmpl-" + str(message.get("id", "proxy"))
            self.usage.update(message.get("usage") or {})
            self.started = True
            return self._chunk({"role": "assistant", "content": ""})
        if kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = event.get("index", 0)
                self.tool_indexes[index] = len(self.tool_indexes)
                return self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self.tool_indexes[index],
                                "id": block.get("id") or "",
                                "type": "function",
                                "function": {"name": block.get("name", ""), "arguments": ""},
                            }
                        ]
                    }
                )
            return b""
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                return self._chunk({"content": delta.get("text", "")})
            if delta.get("type") == "input_json_delta":
                index = self.tool_indexes.get(event.get("index", -1))
                if index is None:
                    return b""
                return self._chunk(
                    {"tool_calls": [{"index": index, "function": {"arguments": delta.get("partial_json", "")}}]}
                )
            return b""
        if kind == "message_delta":
            self.usage.update(event.get("usage") or {})
            reason = (event.get("delta") or {}).get("stop_reason")
            if reason:
                self.stop_reason = _STOP_REASONS.get(reason, "stop")
            return b""
        if kind == "error":
            self.error = (event.get("error") or {}).get("message", "Upstream error")
            return f"data: {json.dumps({'error': {'message': self.error, 'type': 'api_error'}})}\n\n".encode("utf-8")
        if kind == "message_stop":
            finish = "tool_calls" if self.tool_indexes else self.stop_reason
            if self.error:
                finish = "stop"
            tail = self._chunk({}, finish_reason=finish) + b"data: [DONE]\n\n"
            return tail
        return b""


def chat_usage_to_anthropic(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return {
        "input_tokens": max((usage.get("prompt_tokens") or 0) - cached, 0),
        "output_tokens": usage.get("completion_tokens") or 0,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": usage.get("cache_creation_tokens") or 0,
    }


def _image_url(block: dict[str, Any]) -> str:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    if source.get("type") == "url":
        return source.get("url", "")
    return ""


def _parse_arguments(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def to_chat_completions_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Anthropic Messages request -> Chat Completions request."""
    messages: list[dict[str, Any]] = []
    system_text = body.get("system")
    if system_text:
        messages.append({"role": "system", "content": _text_of(system_text)})
    for message in body.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}] if content else [])
        if role == "assistant":
            parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            tool_calls = [
                {
                    "id": b.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {"name": b.get("name", ""), "arguments": json.dumps(b.get("input", {}))},
                }
                for i, b in enumerate(blocks) if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue
        parts: list[dict[str, Any]] = []
        for block in blocks:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                url = _image_url(block)
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "tool_result":
                tool_content = block.get("content")
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": tool_content if isinstance(tool_content, str) else _text_of(tool_content),
                })
        if parts:
            messages.append({"role": role, "content": parts})
    request: dict[str, Any] = {"model": model, "messages": messages, "stream": bool(body.get("stream"))}
    if body.get("max_tokens"):
        request["max_tokens"] = body["max_tokens"]
    tools = [
        {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                          "parameters": tool["input_schema"]}}
        for tool in body.get("tools") or [] if "input_schema" in tool
    ]
    if tools:
        request["tools"] = tools
        choice = body.get("tool_choice") or {}
        kind = choice.get("type")
        if kind == "tool":
            request["tool_choice"] = {"type": "function", "function": {"name": choice.get("name", "")}}
        elif kind in ("auto", "any", "none"):
            request["tool_choice"] = {"auto": "auto", "any": "required", "none": "none"}.get(kind)
        if choice.get("disable_parallel_tool_use"):
            request["parallel_tool_calls"] = False
    if body.get("stop_sequences"):
        request["stop"] = body["stop_sequences"]
    if body.get("temperature") is not None:
        request["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        request["top_p"] = body["top_p"]
    if request.pop("stream"):
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
    return request


_CC_STOP_REASONS = {
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "stop": "end_turn",
    "content_filter": "end_turn",
}


def to_anthropic_message_from_chat(data: dict[str, Any], requested_model: str) -> dict[str, Any]:
    """Chat Completions response -> Anthropic message."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        content += [{"type": "text", "text": p.get("text", "")} for p in text
                    if isinstance(p, dict) and p.get("type") == "text"]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"call_{len(content)}",
                "name": function.get("name", ""),
                "input": _parse_arguments(function.get("arguments")),
            }
        )
    finish = choice.get("finish_reason")
    return {
        "id": "msg_" + str(data.get("id", "proxy")),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": _CC_STOP_REASONS.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": chat_usage_to_anthropic(data.get("usage")),
    }


class ChatToAnthropic:
    """Converts a Chat Completions SSE byte stream into Anthropic Messages SSE bytes."""

    def __init__(self, requested_model: str) -> None:
        self.model = requested_model
        self.parser = SseParser()
        self.message_id = "msg_proxy"
        self.started = False
        self.next_index = 0
        self.open_index: int | None = None
        self.open_kind: str | None = None
        self.tool_indexes: dict[int, int] = {}
        self.usage_raw: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.cost: str | None = None

    def feed(self, chunk: bytes | str) -> bytes:
        out = b""
        for event in self.parser.feed(chunk):
            out += self._event(event)
        return out

    def _emit(self, name: str, data: dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode("utf-8")

    def _start(self) -> bytes:
        if self.started:
            return b""
        self.started = True
        return self._emit(
            "message_start",
            {"type": "message_start", "message": {"id": self.message_id, "type": "message", "role": "assistant",
                                                  "model": self.model, "content": [], "stop_reason": None,
                                                  "stop_sequence": None,
                                                  "usage": {"input_tokens": 0, "output_tokens": 0}}},
        )

    def _open(self, kind: str, block: dict[str, Any]) -> bytes:
        self.open_index = self.next_index
        self.next_index += 1
        self.open_kind = kind
        return self._emit("content_block_start",
                          {"type": "content_block_start", "index": self.open_index, "content_block": block})

    def _close(self) -> bytes:
        if self.open_index is None:
            return b""
        out = self._emit("content_block_stop", {"type": "content_block_stop", "index": self.open_index})
        self.open_index = None
        self.open_kind = None
        return out

    def _event(self, chunk: dict[str, Any]) -> bytes:
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage_raw = usage
            cost = usage.get("cost")
            if cost is not None:
                self.cost = str(cost)
        choices = chunk.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        finish = choices[0].get("finish_reason") if choices else None
        out = b""
        if not self.started:
            out += self._start()
        if delta.get("content"):
            if self.open_kind != "text":
                out += self._close()
                out += self._open("text", {"type": "text", "text": ""})
            out += self._emit("content_block_delta",
                              {"type": "content_block_delta", "index": self.open_index,
                               "delta": {"type": "text_delta", "text": delta["content"]}})
        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            function = call.get("function") or {}
            if call.get("id") or function.get("name"):
                out += self._close()
                self.tool_indexes[index] = self.next_index
                out += self._open("tool", {"type": "tool_use", "id": call.get("id") or f"call_{index}",
                                           "name": function.get("name", ""), "input": {}})
            arguments = function.get("arguments")
            if arguments and index in self.tool_indexes:
                out += self._emit("content_block_delta",
                                  {"type": "content_block_delta", "index": self.tool_indexes[index],
                                   "delta": {"type": "input_json_delta", "partial_json": arguments}})
        if finish:
            if finish == "tool_calls":
                self.stop_reason = "tool_use"
                out += self._close()
            else:
                self.stop_reason = _CC_STOP_REASONS.get(finish, "end_turn")
        if finish and not any(c.get("finish_reason") for c in choices[1:]):
            out += self._emit("message_delta",
                              {"type": "message_delta",
                               "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                               "usage": chat_usage_to_anthropic(self.usage_raw)})
            out += self._emit("message_stop", {"type": "message_stop"})
        return out



def err_to_chat(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Anthropic-shaped error payload -> Chat Completions error payload."""
    error = payload.get("error") or {}
    return chat_error(status, str(error.get("message", "Upstream error")), str(error.get("type", "api_error")))
