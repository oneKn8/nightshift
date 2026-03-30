"""Multi-provider LLM API dispatch."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class DispatchResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "gpt": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "o1": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "o3": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "claude": ("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("google", "https://generativelanguage.googleapis.com/v1beta", "GOOGLE_API_KEY"),
    "deepseek": ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}


class Dispatcher:
    """Routes LLM calls to the correct provider API."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=120.0)

    def dispatch_sync(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> DispatchResult:
        """Send messages to the appropriate LLM API."""
        provider, model_id = self._parse_model(model)
        api_key = self._get_key(provider)

        if provider == "anthropic":
            return self._dispatch_anthropic(messages, model_id, api_key, **kwargs)
        else:
            return self._dispatch_openai_compat(messages, model_id, provider, api_key, **kwargs)

    def _parse_model(self, model: str) -> tuple[str, str]:
        """Determine provider from model name."""
        for prefix, (provider, _, _) in _PROVIDERS.items():
            if model.startswith(prefix):
                return provider, model
        return "openai", model

    def _get_key(self, provider: str) -> str:
        for _, (prov, _, env_var) in _PROVIDERS.items():
            if prov == provider:
                key = os.environ.get(env_var, "")
                if key:
                    return key
        raise ValueError(
            f"API key not found for provider '{provider}'. "
            f"Set the appropriate environment variable."
        )

    def _format_messages(
        self, messages: list[dict[str, str]], provider: str
    ) -> list[dict[str, str]]:
        """Format messages for provider. OpenAI format is the baseline."""
        return messages

    def _dispatch_openai_compat(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str,
        api_key: str,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch to OpenAI-compatible API (OpenAI, DeepSeek, etc.)."""
        base_url = "https://api.openai.com/v1"
        for _, (prov, url, _) in _PROVIDERS.items():
            if prov == provider:
                base_url = url
                break

        resp = self._client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages, **kwargs},
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        content = data["choices"][0]["message"]["content"]
        return DispatchResult(
            content=content,
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw_response=data,
        )

    def _dispatch_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str,
        api_key: str,
        **kwargs: Any,
    ) -> DispatchResult:
        """Dispatch to Anthropic Messages API."""
        system_msg = ""
        non_system = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                non_system.append(m)

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": non_system,
        }
        if system_msg:
            body["system"] = system_msg.strip()

        resp = self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        content = data["content"][0]["text"]
        return DispatchResult(
            content=content,
            model=model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw_response=data,
        )

    def close(self) -> None:
        self._client.close()
