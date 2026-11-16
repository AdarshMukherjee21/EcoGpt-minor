import json

from proxy.translate_chat import (
    AnthropicToChat,
    from_chat_completions,
    to_chat_completions,
)


def test_from_chat_completions_maps_system_tools_and_stops():
    body = {
        "model": "claude-haiku-5-5",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{\"q\": \"weather\"}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        "max_completion_tokens": 256,
        "temperature": 0.4,
        "top_p": 0.9,
        "stop": ["END"],
        "tools": [{"type": "function", "function": {
            "name": "lookup", "description": "finds things",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }}],
        "tool_choice": "auto",
    }
    request = from_chat_completions(body)
    assert request["max_tokens"] == 256
    assert request["system"] == "be brief"
    assert request["stop_sequences"] == ["END"]
    assert request["temperature"] == 0.4
    assert request["tools"] == [{
        "name": "lookup", "description": "finds things",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }]
    assert request["tool_choice"] == {"type": "auto"}
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert request["messages"][1]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "weather"}}
    ]
    assert request["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}
    ]


_TOOL = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


def test_from_chat_completions_tool_choice_variants():
    assert from_chat_completions({"messages": [], "tools": _TOOL, "tool_choice": "required"})["tool_choice"] == {"type": "any"}
    assert from_chat_completions({"messages": [], "tools": _TOOL, "tool_choice": "none"})["tool_choice"] == {"type": "none"}
    assert from_chat_completions({"messages": [], "tools": _TOOL, "tool_choice": "auto"})["tool_choice"] == {"type": "auto"}
    assert from_chat_completions({"messages": [], "tools": _TOOL, "tool_choice": {
        "type": "function", "function": {"name": "f"}}})["tool_choice"] == {"type": "tool", "name": "f"}
    # no tools -> no tool_choice on the Anthropic request
    assert "tool_choice" not in from_chat_completions({"messages": [], "tool_choice": "auto"})


def test_from_chat_completions_image_data_url():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,QUJD"}},
    ]}]}
    request = from_chat_completions(body)
    block = request["messages"][0]["content"][1]
    assert block == {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": "QUJD"}}


def test_to_chat_completions_text_tools_and_usage():
    message = {
        "id": "msg_up",
        "content": [
            {"type": "text", "text": "run it"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 4},
    }
    data = to_chat_completions(message, "claude-haiku-5-5")
    assert data["object"] == "chat.completion"
    assert data["model"] == "claude-haiku-5-5"
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "run it"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == json.dumps({"q": "x"})
    assert data["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


ANTHROPIC_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_s1","model":"claude-haiku-5-5",'
    '"usage":{"input_tokens":5}}}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)

TOOL_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_t1"}}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"toolu_9","name":"lookup","input":{}}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":"}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"\\"x\\"}"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def _converted_sse(sse: str) -> list[dict]:
    converter = AnthropicToChat("claude-haiku-5-5")
    chunks = converter.feed(sse.encode("utf-8")).decode("utf-8")
    return [json.loads(line[6:]) for line in chunks.splitlines()
            if line.startswith("data: ") and not line.startswith("data: [DONE]")]


def test_anthropic_to_chat_streams_text():
    events = _converted_sse(ANTHROPIC_SSE)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    texts = [e["choices"][0]["delta"].get("content", "") for e in events[1:-1]]
    assert "".join(texts) == "Hello"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_anthropic_to_chat_handles_tool_calls():
    events = _converted_sse(TOOL_SSE)
    tool_chunks = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "lookup"
    args = "".join(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "") for chunk in tool_chunks
    )
    assert json.loads(args) == {"q": "x"}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_to_chat_survives_split_chunks():
    converter = AnthropicToChat("claude-haiku-5-5")
    raw = ANTHROPIC_SSE.encode()
    out = b""
    for starts in range(0, len(raw), 7):
        out += converter.feed(raw[starts:starts + 7])
    events = [json.loads(line[6:]) for line in out.decode().splitlines()
              if line.startswith("data: ") and not line.startswith("data: [DONE]")]
    texts = "".join(e["choices"][0]["delta"].get("content", "") for e in events
                    if e.get("choices") and e["choices"][0]["delta"].get("content"))
    assert texts == "Hello"
