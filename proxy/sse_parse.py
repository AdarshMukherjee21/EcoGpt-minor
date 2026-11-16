"""Incremental SSE byte parser: feed raw stream chunks, get data-payload events.

Anthropic passthrough streams and the Responses->Anthropic converter both emit
byte chunks of an SSE stream; the ingress path (chat completions, responses)
needs those events back as dicts to re-encode in the client's format.
"""

from __future__ import annotations

import json
from typing import Any


class SseParser:
    """Buffers partial SSE frames across arbitrary byte-chunk boundaries."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes | str) -> list[dict[str, Any]]:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        self._buffer += chunk
        events: list[dict[str, Any]] = []
        while b"\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(b"\n\n", 1)
            data_lines = [
                line[5:].strip()
                for line in frame.split(b"\n")
                if line.startswith(b"data:")
            ]
            if not data_lines:
                continue
            try:
                event = json.loads(b"\n".join(data_lines).decode("utf-8"))
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events
