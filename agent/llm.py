"""LLM clients that support tool (function) calling.

The model never runs code itself. It replies with a request like
    {"name": "read_file", "arguments": "{\"path\": \"calc.py\"}"}
our loop runs that tool and sends the result back as a "tool" message.
"""

from __future__ import annotations

import itertools
import json
import ssl
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None  # the model sent invalid JSON


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...


def parse_tool_calls(raw: list[dict] | None) -> list[ToolCall]:
    calls = []
    for i, tc in enumerate(raw or []):
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            error = None if isinstance(args, dict) else "arguments must be a JSON object"
        except json.JSONDecodeError as exc:
            args, error = {}, f"invalid JSON arguments: {exc}"
        calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args if isinstance(args, dict) else {}, raw_args, error))
    return calls


def _ssl_verify():
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:  # pragma: no cover
        return True


class OpenAICompatibleModel:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120,
                 client: httpx.Client | None = None, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = model
        self.temperature = temperature
        self._client = client or httpx.Client(timeout=timeout, verify=_ssl_verify())

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        payload = {
            "model": self.name,
            "messages": messages,
            "tools": [{"type": "function", "function": t} for t in tools],
            "tool_choice": "auto",
            "temperature": self.temperature,
        }
        try:
            r = self._client.post(f"{self.base_url}/chat/completions", json=payload,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as exc:
            raise LLMError(f"network error: {exc}") from exc
        if r.status_code != 200:
            raise LLMError(f"provider returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        msg = data["choices"][0]["message"]
        return Reply(msg.get("content"), parse_tool_calls(msg.get("tool_calls")), data.get("usage", {}))


class ScriptedModel:
    """Replays a fixed list of replies. Used by tests and the offline demo, so the
    agent loop (tool execution, errors, stopping) can be tested without an API."""

    name = "scripted"

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            return Reply("I have nothing more to do.")
        return self.replies.pop(0)


_ids = itertools.count(1)


def call(name: str, **arguments: Any) -> ToolCall:
    """Shortcut for building scripted tool calls."""
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments, raw_arguments=json.dumps(arguments))
