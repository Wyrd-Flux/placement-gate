"""Typed loopback-only Ollama chat adapter.

This is deliberately not a generic HTTP client.  It admits only ``/api/tags``,
``/api/ps``, and ``/api/chat`` on an explicitly supplied IPv4 loopback
endpoint, never follows redirects, and never pulls a missing model.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import socket
from dataclasses import dataclass

from .backends import ChatBackend, ChatSendError

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _assistant_text(message: dict) -> str:
    """Normalize only Ollama's two closed native tool-call encodings."""
    content = message.get("content")
    if not isinstance(content, str):
        raise ChatSendError("chat response content is not text")
    calls = message.get("tool_calls")
    if calls is None:
        return content
    if content.strip():
        raise ChatSendError("assistant content and native tool call are ambiguous")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ChatSendError("assistant must emit exactly one native tool call")
    call = calls[0]
    if not isinstance(call, dict) or set(call) != {"id", "function"}:
        raise ChatSendError("native tool call wrapper is not exact")
    if not isinstance(call["id"], str) or not call["id"].strip():
        raise ChatSendError("native tool call id is missing")
    function = call["function"]
    if (not isinstance(function, dict) or
            set(function) != {"index", "name", "arguments"} or
            not isinstance(function["index"], int) or
            isinstance(function["index"], bool) or
            function["index"] != 0 or
            not isinstance(function["name"], str) or
            not isinstance(function["arguments"], dict)):
        raise ChatSendError("native tool function is not exact")
    name = function["name"]
    arguments = function["arguments"]
    wrapper_keys = {"cmd", "args"}
    if set(arguments) & wrapper_keys and set(arguments) != wrapper_keys:
        raise ChatSendError("wrapped native tool arguments are not exact")
    if set(arguments) == wrapper_keys:
        cmd = arguments["cmd"]
        args = arguments["args"]
        if not isinstance(cmd, str) or not isinstance(args, dict):
            raise ChatSendError("wrapped native tool arguments are not exact")
        admitted_names = {
            "assistant", "tool", "container.exec", cmd, f"tool.{cmd}",
        }
        if name not in admitted_names:
            raise ChatSendError("native tool wrapper name does not bind its command")
    else:
        if name.startswith("tool."):
            cmd = name.removeprefix("tool.")
        else:
            cmd = name
        args = arguments
    argument_order = {
        "read:work.status": ("path",),
        "read:work.list": ("path", "relative_path", "max_depth"),
        "read:work.read": ("path", "relative_path", "max_bytes"),
        "read:work.chunk": ("path", "relative_path", "offset", "max_bytes"),
    }.get(cmd)
    if argument_order is not None and not (set(args) - set(argument_order)):
        args = {key: args[key] for key in argument_order if key in args}
    return json.dumps(
        {"cmd": cmd, "args": args},
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class OllamaChatProfile:
    model: str
    num_gpu: int
    num_ctx: int
    keep_alive: int | str
    require_cpu_only: bool = False
    num_predict: int = 2048
    think: bool = False
    temperature: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-blank exact tag")
        if not isinstance(self.num_gpu, int) or self.num_gpu < 0:
            raise ValueError("num_gpu must be a non-negative integer")
        if not isinstance(self.num_ctx, int) or not 1 <= self.num_ctx <= 262144:
            raise ValueError("num_ctx is outside the admitted range")
        if not isinstance(self.num_predict, int) or not 1 <= self.num_predict <= 32768:
            raise ValueError("num_predict is outside the admitted range")
        if not isinstance(self.keep_alive, (int, str)):
            raise ValueError("keep_alive must be an integer or duration string")
        if self.require_cpu_only and self.num_gpu != 0:
            raise ValueError("CPU-only profile must set num_gpu=0")
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(
                self.temperature, (int, float)
            ):
                raise ValueError("temperature must be a finite number or None")
            if not math.isfinite(float(self.temperature)):
                raise ValueError("temperature must be a finite number or None")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError("seed must be an integer or None")
            if not 0 <= self.seed < 2**32:
                raise ValueError("seed is outside the admitted range")

    def options(self) -> dict[str, int | float]:
        options: dict[str, int | float] = {
            "num_gpu": self.num_gpu,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.temperature is not None:
            options["temperature"] = float(self.temperature)
        if self.seed is not None:
            options["seed"] = self.seed
        return options


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    digest: str
    done_reason: str | None


class OllamaChatAdapter(ChatBackend):
    """Exact-model chat over one explicitly configured loopback endpoint."""

    def __init__(
        self,
        *,
        endpoint: tuple[str, int],
        timeout: float = 600.0,
        max_response_bytes: int = 64 * 1024 * 1024,
    ):
        host, port = endpoint
        if host != "127.0.0.1" or not isinstance(port, int) or not 0 < port < 65536:
            raise ChatSendError("chat endpoint must be explicit 127.0.0.1:<port>")
        self._host = host
        self._port = port
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes

    def endpoint_id(self) -> str:
        return f"{self._host}:{self._port}"

    def resolve_model_digest(self, model: str) -> str:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise ChatSendError("/api/tags response has no models list")
        matches = [
            item for item in models
            if isinstance(item, dict) and item.get("name") == model
        ]
        if len(matches) != 1:
            raise ChatSendError(
                f"model {model!r} must resolve exactly once; got {len(matches)}"
            )
        digest = matches[0].get("digest")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ChatSendError("resolved model digest is missing or malformed")
        return digest

    def send_chat(
        self,
        model: str,
        messages: list[dict],
        *,
        profile: OllamaChatProfile | None = None,
        expected_digest: str | None = None,
        capture: dict | None = None,
    ) -> dict:
        """Send one non-streamed chat.

        ``capture`` (optional) receives raw protocol facts from the already
        captured response without changing the returned envelope: the full
        assistant ``message`` dict, ``done_reason`` and ``usage``.
        """
        active, digest = self._prepare_dispatch(
            model, messages, profile, expected_digest
        )
        payload = self._request(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "think": active.think,
                "options": active.options(),
                "keep_alive": active.keep_alive,
            },
        )
        if payload.get("done") is not True:
            raise ChatSendError("chat response is not complete")
        if payload.get("model") != model:
            raise ChatSendError("chat response model does not match request")
        message = payload.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ChatSendError("chat response has no assistant message")
        text = _assistant_text(message)
        if active.require_cpu_only:
            self._verify_cpu_only(model)
        result = ChatResult(text, model, digest, payload.get("done_reason"))
        usage = self._usage(payload)
        if capture is not None:
            capture["protocol_message"] = message
            capture["done_reason"] = result.done_reason
            capture["usage"] = usage
        return {
            "message": {"role": "assistant", "content": result.text},
            "model": result.model,
            "digest": result.digest,
            "done": True,
            "done_reason": result.done_reason,
            "usage": usage,
        }

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        *,
        profile: OllamaChatProfile | None = None,
        expected_digest: str | None = None,
        frame_sink: list | None = None,
    ):
        """Yield validated assistant chunks; final durability stays external.

        ``frame_sink`` (optional) receives every raw (already-parsed) stream
        frame so the governed boundary can emit protocol observations without
        re-capturing the transport.
        """
        active, digest = self._prepare_dispatch(
            model, messages, profile, expected_digest
        )
        done_seen = False
        for payload in self._stream_request(
            {
                "model": model,
                "messages": messages,
                "stream": True,
                "think": active.think,
                "options": active.options(),
                "keep_alive": active.keep_alive,
            }
        ):
            if done_seen:
                raise ChatSendError("stream emitted data after terminal frame")
            if frame_sink is not None:
                frame_sink.append(payload)
            if payload.get("model") != model:
                raise ChatSendError("stream response model does not match request")
            message = payload.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ChatSendError("stream frame has no assistant message")
            content = _assistant_text(message)
            done = payload.get("done")
            if not isinstance(done, bool):
                raise ChatSendError("stream frame has malformed done flag")
            event = {
                "content": content,
                "done": done,
                "model": model,
                "digest": digest,
                "done_reason": payload.get("done_reason"),
                "usage": self._usage(payload) if done else None,
            }
            if done:
                done_seen = True
            yield event
        if not done_seen:
            raise ChatSendError("stream ended without a terminal frame")
        if active.require_cpu_only:
            self._verify_cpu_only(model)

    def _prepare_dispatch(
        self,
        model: str,
        messages: list[dict],
        profile: OllamaChatProfile | None,
        expected_digest: str | None,
    ) -> tuple[OllamaChatProfile, str]:
        active = profile or OllamaChatProfile(model, 0, 4096, 0)
        if active.model != model:
            raise ChatSendError("profile model does not match requested model")
        if not isinstance(messages, list) or not messages:
            raise ChatSendError("messages must be a non-empty list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ChatSendError("message shape is not admitted")
        digest = self.resolve_model_digest(model)
        if expected_digest is not None and digest != expected_digest:
            raise ChatSendError("model digest changed before dispatch")
        return active, digest

    @staticmethod
    def _usage(payload: dict) -> dict[str, int | None]:
        prompt_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        for label, value in (
            ("prompt_eval_count", prompt_tokens),
            ("eval_count", output_tokens),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ChatSendError(f"{label} is malformed")
        return {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }

    def _stream_request(self, body: dict):
        raw_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        total = 0
        try:
            conn.request(
                "POST", "/api/chat", body=raw_body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            if 300 <= response.status < 400:
                raise ChatSendError("redirect rejected")
            if response.status != 200:
                raise ChatSendError(f"Ollama returned HTTP {response.status}")
            content_type = response.getheader("Content-Type") or ""
            if "json" not in content_type.lower() and "ndjson" not in content_type.lower():
                raise ChatSendError("Ollama stream content type is not JSON")
            while True:
                remaining = self._max_response_bytes - total
                if remaining <= 0:
                    raise ChatSendError("Ollama stream exceeds configured byte cap")
                line = response.readline(remaining + 1)
                if not line:
                    break
                total += len(line)
                if total > self._max_response_bytes:
                    raise ChatSendError("Ollama stream exceeds configured byte cap")
                try:
                    payload = json.loads(line.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ChatSendError("Ollama returned malformed stream JSON") from exc
                if not isinstance(payload, dict):
                    raise ChatSendError("Ollama stream frame must be a JSON object")
                if payload.get("error"):
                    raise ChatSendError("Ollama stream returned an error frame")
                yield payload
        except (OSError, socket.timeout) as exc:
            raise ChatSendError(f"Ollama loopback stream failed: {exc}") from exc
        finally:
            conn.close()

    def _verify_cpu_only(self, model: str) -> None:
        payload = self._request("GET", "/api/ps")
        models = payload.get("models")
        if not isinstance(models, list):
            raise ChatSendError("/api/ps response has no models list")
        matches = [
            item for item in models
            if isinstance(item, dict) and item.get("name") == model
        ]
        if len(matches) != 1 or matches[0].get("size_vram") != 0:
            raise ChatSendError("CPU-only placement was not observed after response")

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if (method, path) not in {
            ("GET", "/api/tags"),
            ("GET", "/api/ps"),
            ("POST", "/api/chat"),
        }:
            raise ChatSendError("operation is not admitted by the chat adapter")
        raw_body = None
        headers: dict[str, str] = {}
        if body is not None:
            raw_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            conn.request(method, path, body=raw_body, headers=headers)
            response = conn.getresponse()
            if 300 <= response.status < 400:
                raise ChatSendError("redirect rejected")
            raw = response.read(self._max_response_bytes + 1)
            if len(raw) > self._max_response_bytes:
                raise ChatSendError("Ollama response exceeds configured byte cap")
            if response.status != 200:
                raise ChatSendError(f"Ollama returned HTTP {response.status}")
            content_type = response.getheader("Content-Type") or ""
            if "json" not in content_type.lower():
                raise ChatSendError("Ollama response content type is not JSON")
            try:
                payload = json.loads(raw.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ChatSendError("Ollama returned malformed JSON") from exc
            if not isinstance(payload, dict):
                raise ChatSendError("Ollama response must be a JSON object")
            return payload
        except (OSError, socket.timeout) as exc:
            raise ChatSendError(f"Ollama loopback request failed: {exc}") from exc
        finally:
            conn.close()
