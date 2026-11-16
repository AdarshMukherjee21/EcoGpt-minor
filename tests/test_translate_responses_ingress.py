import json

from proxy.translate_responses_ingress import (
    AnthropicToResponses,
    from_responses_request,
    to_responses_response,
)


def test_from_responses_request_full_mapping():
    body = {
        "model": "claude-haiku-5-5",
        "instructions": "be brief",
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "call_1", "name": "lookup",
             "arguments": "{\"q\":\"x\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "thanks"}]},
        ],
        "max_output_tokens": 512,
        "tools": [{"type": "function", "name": "lookup", "description": "finds",
                   "parameters": {"type": "object", "properties": {}}}],
        "reasoning": {"effort": "high"},
        "stream": True,
    }
    request = from_responses_request(body)
    assert request["max_tokens"] == 512
    assert request["system"] == "be brief"
    assert request["stream"] is True
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert request["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    roles = [m["role"] for m in request["messages"]]
    # the trailing user text merges into the tool_result message (valid Anthropic)
    assert roles == ["user", "assistant", "user"]
    assert request["messages"][1]["content"][0] == {
        "type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}}
    assert request["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"},
        {"type": "text", "text": "thanks"},
    ]


def test_from_responses_request_string_input_and_default_tokens():
    request = from_responses_request({"model": "m", "input": "hello", "tool_choice": "required",
                                      "tools": [{"type": "function", "name": "f",
                                                 "parameters": {"type": "object", "properties": {}}}]})
    assert request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert request["max_tokens"] == 4096
    assert request["tool_choice"] == {"type": "any"}


def test_to_responses_response_shape():
    message = {
        "id": "msg_up",
        "content": [
            {"type": "text", "text": "Answer"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 4,
                  "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2},
    }
    data = to_responses_response(message, "claude-haiku-5-5")
    assert data["id"].startswith("resp_")
    assert data["status"] == "completed"
    assert data["model"] == "claude-haiku-5-5"
    kinds = [item["type"] for item in data["output"]]
    assert kinds == ["message", "function_call"]
    assert data["output"][0]["content"][0]["text"] == "Answer"
    assert data["output"][1]["call_id"] == "toolu_1"
    usage = data["usage"]
    assert usage["input_tokens"] == 15  # 10 + 3 cached + 2 cache write
    assert usage["input_tokens_details"]["cached_tokens"] == 3
    assert usage["total_tokens"] == 19


RESPONSES_ANTHROPIC_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_s1",'
    '"usage":{"input_tokens":5}}}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"toolu_9","name":"lookup","input":{}}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":1}"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
    '"usage":{"output_tokens":7}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def test_anthropic_to_responses_stream_tool_call():
    converter = AnthropicToResponses("claude-haiku-5-5")
    chunks = converter.feed(RESPONSES_ANTHROPIC_SSE.encode()).decode()
    events = [json.loads(line[6:]) for line in chunks.splitlines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "response.created"
    added = [e for e in events if e["type"] == "response.output_item.added"][0]
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "lookup"
    deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert deltas[0]["delta"] == '{"q":1}'
    done = [e for e in events if e["type"] == "response.output_item.done"][0]
    assert done["item"]["status"] == "completed"
    completed = [e for e in events if e["type"] == "response.completed"][0]
    assert completed["response"]["usage"]["output_tokens"] == 7
