"""Minimal Ollama REST client restricted to loopback for offline inference."""

from __future__ import annotations

import base64
import json
import socket
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class OllamaError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, details: Any = None):
        super().__init__(message)
        self.retryable = retryable
        self.details = details


class LocalOllama:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout_seconds: int = 600):
        parsed = urlparse(host)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in LOCAL_HOSTS
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Offline mode only allows Ollama on localhost over HTTP.")
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # Do not inherit machine proxy settings for a request that must remain local.
        self._opener = build_opener(ProxyHandler({}))

    def _request(self, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(
            f"{self.host}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {502, 503, 504}
            raise OllamaError(
                f"Ollama returned HTTP {exc.code}", retryable=retryable, details=body
            ) from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            raise OllamaError(str(exc), retryable=True) from exc
        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON", details=body) from exc
        if not isinstance(value, dict):
            raise OllamaError("Ollama returned a non-object response", details=value)
        return value

    def version(self) -> str:
        return str(self._request("/api/version").get("version", "unknown"))

    def model_details(self, model_name: str) -> dict[str, Any]:
        models = self._request("/api/tags").get("models", [])
        match = next(
            (entry for entry in models if entry.get("model") == model_name or entry.get("name") == model_name),
            None,
        )
        if match is None:
            raise OllamaError(
                f"Model '{model_name}' is not present in the local Ollama model store. "
                "Pull it before disconnecting from the network."
            )
        if not match.get("digest"):
            raise OllamaError(f"Local Ollama did not return a digest for model '{model_name}'.")
        return {
            "model": model_name,
            "digest": match.get("digest"),
            "size_bytes": match.get("size"),
            "details": match.get("details", {}),
        }

    def chat(
        self,
        model_name: str,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt, "images": [encoded_image]}],
            "format": schema,
            "stream": False,
            "options": options,
        }
        return self._request("/api/chat", payload)


