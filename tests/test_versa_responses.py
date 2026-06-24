"""
Tests for the Versa OpenAI Responses API completion function.

These exercise the streaming-to-ModelResponse adaptation and the chat->responses
kwarg translation with a fake responses function, so no network is required.
"""

from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

from lab_llm import LLMApi, make_versa_openai_responses_completion
from lab_llm.versa.openai import (
    _chat_kwargs_to_responses_kwargs,
    _to_responses_provider_model,
)

REASONING_MODEL = "azure/gpt-5-mini"


class Answer(BaseModel):
    answer: int


def _completed_response(text='{"answer": 42}'):
    """A fake `response.completed` payload (output + usage)."""
    return NS(
        output=[
            NS(type="reasoning", summary=[NS(type="summary_text", text="thinking")]),
            NS(type="message", content=[NS(type="output_text", text=text)]),
        ],
        usage=NS(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            input_tokens_details=NS(cached_tokens=4),
            output_tokens_details=NS(reasoning_tokens=5),
        ),
    )


def make_fake_responses(captured, text='{"answer": 42}'):
    """A stand-in for litellm.responses that records kwargs and streams events."""
    def fake(**kwargs):
        captured.update(kwargs)
        assert kwargs.get("stream") is True, "wrapper must request a streamed response"
        return iter([
            NS(type="response.created", response=NS()),
            NS(type="response.reasoning_summary_text.delta", delta="think"),
            NS(type="response.output_text.delta", delta=text[: len(text) // 2]),
            NS(type="response.output_text.delta", delta=text[len(text) // 2:]),
            NS(type="response.completed", response=_completed_response(text)),
        ])
    return fake


def test_kwarg_translation_for_reasoning_model():
    out = _chat_kwargs_to_responses_kwargs(REASONING_MODEL, {
        "reasoning_effort": "medium",
        "verbosity": "low",
        "max_tokens": 100,
        "temperature": 1.0,
        "seed": 7,
        "response_format": Answer,
    })
    # Reasoning + keepalive summary
    assert out["reasoning"] == {"summary": "auto", "effort": "medium"}
    # max_tokens -> max_output_tokens
    assert out["max_output_tokens"] == 100
    # structured output -> text_format
    assert out["text_format"] is Answer
    # sampling params dropped for reasoning calls; chat-only params dropped
    assert "temperature" not in out
    assert "seed" not in out
    assert "max_tokens" not in out
    assert "response_format" not in out
    # verbosity is not sent alongside structured output (text_format owns `text`)
    assert "text" not in out


def test_reasoning_model_detected_without_explicit_effort():
    # No reasoning_effort given, but the model id names a reasoning model.
    out = _chat_kwargs_to_responses_kwargs(REASONING_MODEL, {"temperature": 0.5})
    assert out["reasoning"] == {"summary": "auto"}
    assert "temperature" not in out  # dropped: still a reasoning model


def test_non_reasoning_model_sends_no_reasoning_keeps_sampling():
    out = _chat_kwargs_to_responses_kwargs("azure/gpt-4o", {
        "temperature": 0.3, "top_p": 0.9, "verbosity": "high",
    })
    assert "reasoning" not in out
    assert out["temperature"] == 0.3
    assert out["top_p"] == 0.9
    assert out["text"] == {"verbosity": "high"}


def test_max_completion_tokens_takes_precedence_and_zero_is_kept():
    out = _chat_kwargs_to_responses_kwargs(REASONING_MODEL, {
        "max_completion_tokens": 0, "max_tokens": 50,
    })
    assert out["max_output_tokens"] == 0  # 0 is not treated as absent


def test_tools_raise_not_implemented():
    with pytest.raises(NotImplementedError, match="Tool-calling"):
        _chat_kwargs_to_responses_kwargs(REASONING_MODEL, {"tools": [{"type": "function"}]})


def test_public_stream_raises():
    with pytest.raises(ValueError, match="stream=True"):
        _chat_kwargs_to_responses_kwargs(REASONING_MODEL, {"stream": True})


def test_wrapper_adapts_stream_to_model_response():
    captured = {}
    completion = make_versa_openai_responses_completion(
        completion_func=make_fake_responses(captured),
        endpoint="https://example.openai.azure.com/openai/v1/",
        api_key="key",
        api_version="2024-12-01-preview",
    )

    resp = completion(
        REASONING_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="medium",
        max_tokens=100,
    )

    # Content aggregated from the completed payload
    assert resp.choices[0].message.content == '{"answer": 42}'
    # Usage mapped onto chat-completions shape
    assert resp.usage.prompt_tokens == 11
    assert resp.usage.completion_tokens == 7
    assert resp.usage.total_tokens == 18
    assert resp.usage.completion_tokens_details.reasoning_tokens == 5
    assert resp.usage.prompt_tokens_details.cached_tokens == 4

    # Request was shaped for the Responses API
    assert captured["input"] == [{"role": "user", "content": "hi"}]
    # Routed through litellm's openai/ provider (Versa serves Responses on its
    # OpenAI-compatible /openai/v1 surface); the azure/ id is normalized.
    assert captured["model"] == "openai/gpt-5-mini"
    assert captured["reasoning"] == {"summary": "auto", "effort": "medium"}
    assert captured["api_base"] == "https://example.openai.azure.com/openai/v1/"
    assert captured["api_key"] == "key"

    # The aggregated response must round-trip for caching (no streaming leaks out)
    assert resp.model_dump()["choices"][0]["message"]["content"] == '{"answer": 42}'


def test_llmapi_run_with_structured_output_and_usage():
    captured = {}
    api = LLMApi(
        make_versa_openai_responses_completion(
            completion_func=make_fake_responses(captured),
            endpoint="https://example.openai.azure.com/openai/v1/",
            api_key="key",
        ),
        track_usage=True,
    )

    result = api.run(
        "hi",
        model=REASONING_MODEL,
        reasoning_effort="medium",
        max_tokens=100,
        response_format=Answer,
    )

    assert isinstance(result, Answer)
    assert result.answer == 42
    assert api.usage_tracker.last_usage()["reasoning_tokens"] == 5
    # response_format converted to text_format; not forwarded raw
    assert "response_format" not in captured
    assert captured["text_format"] is Answer


def test_failed_event_raises():
    def fake(**kwargs):
        return iter([NS(type="response.failed", response=NS(error="boom"))])

    completion = make_versa_openai_responses_completion(
        completion_func=fake, endpoint="https://x/openai/v1/", api_key="k"
    )
    with pytest.raises(RuntimeError, match="boom"):
        completion(REASONING_MODEL, messages=[{"role": "user", "content": "hi"}])


def test_incomplete_event_raises():
    def fake(**kwargs):
        return iter([
            NS(type="response.output_text.delta", delta="partial"),
            NS(type="response.incomplete",
               response=NS(incomplete_details=NS(reason="max_output_tokens"))),
        ])

    completion = make_versa_openai_responses_completion(
        completion_func=fake, endpoint="https://x/openai/v1/", api_key="k"
    )
    with pytest.raises(RuntimeError, match="incomplete.*max_output_tokens"):
        completion(REASONING_MODEL, messages=[{"role": "user", "content": "hi"}])


def test_stream_without_completed_raises():
    def fake(**kwargs):
        return iter([NS(type="response.output_text.delta", delta="partial")])

    completion = make_versa_openai_responses_completion(
        completion_func=fake, endpoint="https://x/openai/v1/", api_key="k"
    )
    with pytest.raises(RuntimeError, match="without a response.completed"):
        completion(REASONING_MODEL, messages=[{"role": "user", "content": "hi"}])


def test_missing_endpoint_raises(monkeypatch):
    monkeypatch.delenv("VERSA_RESPONSES_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="VERSA_RESPONSES_ENDPOINT"):
        make_versa_openai_responses_completion(api_key="k")


@pytest.mark.parametrize("given,expected", [
    ("azure/gpt-5-mini-2025-08-07", "openai/gpt-5-mini-2025-08-07"),
    ("openai/gpt-5-mini", "openai/gpt-5-mini"),
    ("gpt-5-mini", "openai/gpt-5-mini"),
])
def test_provider_normalized_to_openai(given, expected):
    # Versa serves Responses on its OpenAI-compatible /openai/v1 surface, so the
    # request must route through litellm's openai/ provider regardless of prefix.
    assert _to_responses_provider_model(given) == expected


def test_api_version_not_forwarded_by_default():
    captured = {}
    completion = make_versa_openai_responses_completion(
        completion_func=make_fake_responses(captured),
        endpoint="https://x/openai/v1", api_key="k",
    )
    completion(REASONING_MODEL, messages=[{"role": "user", "content": "hi"}])
    # The /openai/v1 surface takes no api-version; none should be injected.
    assert "api_version" not in captured


def test_api_version_forwarded_when_set():
    captured = {}
    completion = make_versa_openai_responses_completion(
        completion_func=make_fake_responses(captured),
        endpoint="https://x/openai/v1", api_key="k", api_version="preview",
    )
    completion(REASONING_MODEL, messages=[{"role": "user", "content": "hi"}])
    assert captured["api_version"] == "preview"
