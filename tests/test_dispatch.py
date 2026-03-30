"""Tests for LLM API dispatch."""
import pytest
from nightshift.dispatch import Dispatcher, DispatchResult


class TestDispatcher:
    def test_parse_model_provider_openai(self):
        d = Dispatcher()
        provider, model = d._parse_model("gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_parse_model_provider_anthropic(self):
        d = Dispatcher()
        provider, model = d._parse_model("claude-sonnet-4-20250514")
        assert provider == "anthropic"

    def test_parse_model_provider_google(self):
        d = Dispatcher()
        provider, model = d._parse_model("gemini-2.0-flash")
        assert provider == "google"

    def test_parse_model_provider_deepseek(self):
        d = Dispatcher()
        provider, model = d._parse_model("deepseek-chat")
        assert provider == "deepseek"

    def test_parse_model_provider_unknown(self):
        d = Dispatcher()
        provider, model = d._parse_model("some-random-model")
        assert provider == "openai"

    def test_format_messages_openai(self):
        d = Dispatcher()
        msgs = [{"role": "user", "content": "hi"}]
        formatted = d._format_messages(msgs, "openai")
        assert formatted == msgs

    def test_dispatch_result_structure(self):
        r = DispatchResult(
            content="hello",
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            raw_response={"id": "123"},
        )
        assert r.content == "hello"
        assert r.total_tokens == 15

    def test_dispatch_without_api_key_raises(self):
        d = Dispatcher()
        msgs = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="API key"):
            d.dispatch_sync(msgs, "gpt-4o")
