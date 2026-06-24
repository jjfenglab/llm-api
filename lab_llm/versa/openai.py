import os
from typing import Callable, Any, Optional, List
from ..constants import VersaOpenAI, VERSA_API_VERSION
from ..types import CompletionFunction
from ..parameter_wrappers import ModelRamp
import logging
from functools import wraps

from litellm import ModelResponse
from litellm.types.utils import Usage

VersaOpenAIModelRamp = ModelRamp([
    VersaOpenAI.GPT4_O_2024_11,
    VersaOpenAI.GPT5_NANO_2025_08,
    VersaOpenAI.GPT5_MINI_2025_08,
    VersaOpenAI.GPT5_2025_08,
    VersaOpenAI.O_1_2024_12
])

def make_versa_openai_completion(completion_func: Optional[CompletionFunction] = None, api_key: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> CompletionFunction:
    """
    Wrapper around litellm.completion that configures Azure OpenAI parameters from environment variables.

    Maps environment variables:
    - VERSA_ENDPOINT -> azure_endpoint (with /openai/ appended)
    - VERSA_API_KEY -> api_key
    - Uses VERSA_API_VERSION (defaults to "2024-10-21") -> api_version

    Args:
        completion_func: Function to wrap. If none is passed, uses litellm.completion.

    Returns:
        Wrapped function that passes in Versa Azure OpenAI credentials
    """
    # Get Azure OpenAI configuration from environment
    endpoint = endpoint or os.getenv("VERSA_ENDPOINT")
    api_key = api_key or os.getenv("VERSA_API_KEY")
    api_version = api_version or os.getenv("VERSA_API_VERSION", VERSA_API_VERSION)

    if not endpoint:
        raise ValueError("Versa endpoint must be provided via VERSA_ENDPOINT environment variable")
    if not api_key:
        raise ValueError("API key must be provided via VERSA_API_KEY environment variable")

    logging.info("Versa Azure OpenAI Endpoint URL: %s", endpoint)

    # Set Azure OpenAI parameters for litellm
    azure_kwargs = {
        "api_key": api_key,
        "api_base": endpoint,
        "api_version": api_version
    }

    if not completion_func:
        import litellm
        completion_func = litellm.completion

    @wraps(completion_func)
    def wrapper(model: str, **kwargs) -> Any:
        return completion_func(model, **{**kwargs, **azure_kwargs})
    return wrapper


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute- or dict-style access (litellm hands back either shape)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _messages_to_responses_input(messages: List) -> List[dict]:
    """Normalize chat-style messages into Responses API ``input`` items."""
    items = []
    for m in messages:
        if isinstance(m, dict):
            items.append(m)
        elif hasattr(m, "model_dump"):
            items.append(m.model_dump())
        else:
            items.append(dict(m))
    return items


# Chat-completions kwargs the Responses API does not accept, or that we
# translate / handle explicitly below. Everything else is passed through.
_RESPONSES_DROP_KWARGS = frozenset({
    "reasoning_effort", "reasoning_summary", "verbosity", "response_format",
    "max_tokens", "max_completion_tokens", "temperature", "top_p", "timeout",
    # Chat-completions-only params with no Responses API equivalent:
    "seed", "n", "stream", "stream_options", "stop", "frequency_penalty",
    "presence_penalty", "logit_bias", "logprobs", "top_logprobs",
    "functions", "function_call", "deployment_id", "model_list",
    # Tool-calling via the Responses path is not supported yet (the aggregated
    # ModelResponse carries content only), so drop tool params rather than send
    # an incompatible schema.
    "tools", "tool_choice", "parallel_tool_calls",
})


def _looks_like_reasoning_model(model: Optional[str]) -> bool:
    """Heuristic: does this model id name a reasoning model (GPT-5, o-series)?"""
    if not model:
        return False
    name = model.split("/")[-1].lower()
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


def _to_responses_provider_model(model: str) -> str:
    """Route the Responses call through litellm's OpenAI-compatible provider.

    Versa serves the Responses API on its OpenAI-compatible ``/openai/v1``
    surface (the same gateway the chat-completions deployments proxy fronts),
    not on the deployment-less Azure ``/openai/responses`` path. With
    ``model="azure/..."`` litellm builds the Azure URL
    (``.../openai/v1/openai/responses?api-version=...``) and the gateway 404s, so
    we normalize whatever provider prefix the caller used (``azure/``,
    ``openai/``, or none) to ``openai/`` and let ``api_base`` point at
    ``/openai/v1``. The caller-facing model id is preserved on the returned
    ModelResponse; only the litellm request is re-providered.
    """
    name = model.split("/", 1)[1] if "/" in model else model
    return f"openai/{name}"


def _chat_kwargs_to_responses_kwargs(model: Optional[str], kwargs: dict) -> dict:
    """Translate chat-completions kwargs into Responses API kwargs.

    Streaming is internal to this adapter and tool-calling is not supported yet,
    so both are rejected loudly rather than silently mishandled (``LLMApi.run``
    forwards ``stream`` and ``tools`` straight through).
    """
    if kwargs.get("stream"):
        raise ValueError(
            "stream=True is not supported here; the Responses adapter streams "
            "internally and returns an aggregated, non-streaming ModelResponse."
        )
    if kwargs.get("tools"):
        raise NotImplementedError(
            "Tool-calling is not supported by make_versa_openai_responses_completion "
            "yet (the aggregated ModelResponse carries content only). Use "
            "make_versa_openai_completion for tools."
        )
    if kwargs.get("tool_choice") not in (None, "none"):
        raise NotImplementedError(
            "tool_choice is not supported by make_versa_openai_responses_completion yet."
        )

    out: dict = {}

    # Only reasoning models accept (and need) the `reasoning` object. The
    # streamed reasoning summary is the whole point: it emits bytes while the
    # model thinks, keeping the connection alive past the gateway idle timeout.
    # Sending `reasoning` to a non-reasoning deployment (e.g. gpt-4o) 400s.
    effort = kwargs.get("reasoning_effort")
    is_reasoning = effort is not None or _looks_like_reasoning_model(model)
    if is_reasoning:
        reasoning = {"summary": kwargs.get("reasoning_summary", "auto")}
        if effort is not None:
            reasoning["effort"] = effort
        out["reasoning"] = reasoning
    else:
        # Sampling params are only valid for non-reasoning models.
        for k in ("temperature", "top_p"):
            if kwargs.get(k) is not None:
                out[k] = kwargs[k]

    response_format = kwargs.get("response_format")
    if response_format is not None:
        # litellm builds ``text.format`` from a pydantic model (or accepts a
        # JSON-schema dict) via ``text_format``.
        out["text_format"] = response_format
    elif kwargs.get("verbosity") is not None:
        out["text"] = {"verbosity": kwargs["verbosity"]}

    # Deterministic max-output mapping (do not treat 0 as absent).
    if kwargs.get("max_completion_tokens") is not None:
        out["max_output_tokens"] = kwargs["max_completion_tokens"]
    elif kwargs.get("max_tokens") is not None:
        out["max_output_tokens"] = kwargs["max_tokens"]

    if kwargs.get("timeout") is not None:
        out["timeout"] = kwargs["timeout"]

    # Pass through anything we did not explicitly translate or drop.
    for k, v in kwargs.items():
        if k not in _RESPONSES_DROP_KWARGS and k not in out:
            out[k] = v

    return out


def _consume_responses_stream(stream: Any):
    """Drain a streaming Responses iterator.

    Returns ``(final_response, streamed_text)``. Iterating the stream is what
    consumes the keep-alive reasoning-summary deltas; we keep the final
    ``response.completed`` payload (full output + usage) and, as a fallback,
    the accumulated output-text deltas.
    """
    final_response = None
    text_deltas: List[str] = []
    for event in stream:
        etype = _get(event, "type")
        if etype == "response.output_text.delta":
            delta = _get(event, "delta")
            if delta:
                text_deltas.append(delta)
        elif etype == "response.completed":
            final_response = _get(event, "response")
        elif etype == "response.incomplete":
            # Truncation (e.g. max_output_tokens) — fail closed rather than
            # cache/return a partial answer as if it were complete.
            reason = _get(_get(_get(event, "response"), "incomplete_details"), "reason", "unknown")
            raise RuntimeError(
                f"Versa Responses API response incomplete (reason: {reason}); "
                "raise max_output_tokens or shorten the input."
            )
        elif etype == "response.failed":
            err = _get(_get(event, "response"), "error")
            raise RuntimeError(f"Versa Responses API request failed: {err}")
    if final_response is None:
        # A disconnect after reasoning but before completion must not look like
        # a successful empty answer.
        raise RuntimeError(
            "Versa Responses API stream ended without a response.completed event."
        )
    return final_response, "".join(text_deltas)


def _extract_output_text(final_response: Any) -> str:
    """Concatenate ``output_text`` parts from a completed Responses payload."""
    parts: List[str] = []
    for item in (_get(final_response, "output") or []):
        if _get(item, "type") != "message":
            continue
        for part in (_get(item, "content") or []):
            if _get(part, "type") == "output_text":
                parts.append(_get(part, "text") or "")
    return "".join(parts)


def _build_responses_usage(final_response: Any) -> Optional[Usage]:
    """Map Responses API usage onto a chat-completions ``Usage`` object so the
    existing UsageTracker / cache logic reads it unchanged."""
    ru = _get(final_response, "usage")
    if ru is None:
        return None
    input_tokens = _get(ru, "input_tokens", 0) or 0
    output_tokens = _get(ru, "output_tokens", 0) or 0
    total_tokens = _get(ru, "total_tokens", 0) or (input_tokens + output_tokens)
    cached = _get(_get(ru, "input_tokens_details"), "cached_tokens")
    reasoning = _get(_get(ru, "output_tokens_details"), "reasoning_tokens")
    return Usage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details={"cached_tokens": cached} if cached else None,
        completion_tokens_details={"reasoning_tokens": reasoning} if reasoning else None,
    )


def _responses_to_model_response(model: str, final_response: Any, streamed_text: str) -> ModelResponse:
    """Adapt a completed Responses payload into a chat-completions ModelResponse.

    Choices/message are passed as plain dicts so ModelResponse builds its own
    (non-streaming) internal objects — this keeps ``model_dump()`` faithful for
    the caching round-trip.
    """
    text = _extract_output_text(final_response) or streamed_text
    usage = _build_responses_usage(final_response)
    kwargs: dict = {
        "model": model,
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }],
    }
    if usage is not None:
        kwargs["usage"] = usage
    return ModelResponse(**kwargs)


def make_versa_openai_responses_completion(
    completion_func: Optional[CompletionFunction] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> CompletionFunction:
    """
    Versa Azure OpenAI completion function that drives the **Responses API**
    instead of chat-completions, for reasoning models (GPT-5, o-series).

    Reasoning models can think for minutes while chat-completions streams
    nothing, so the Versa gateway closes the connection at its idle timeout
    (~300s). The Responses API, streamed with ``reasoning.summary="auto"``,
    emits reasoning-summary deltas while the model thinks, keeping the
    connection alive. The stream is consumed internally and an aggregated,
    non-streamed ``ModelResponse`` is returned, so this composes transparently
    with ``CachingCompletion``, ``UsageTracker`` and ``LLMApi.run`` — none of
    them observe the streaming.

    Maps environment variables:
        VERSA_RESPONSES_ENDPOINT     -> api_base (the OpenAI-compatible v1 base)
        VERSA_API_KEY                -> api_key
        VERSA_RESPONSES_API_VERSION  -> api_version (optional; not used by the
                                        /openai/v1 surface, forwarded only if set)

    The Responses endpoint is required explicitly (it is a distinct URL from the
    chat-completions ``VERSA_ENDPOINT`` and is not derived from it). Set it to the
    OpenAI-compatible v1 base, e.g.
    ``https://<resource>/openai/v1`` (for Versa: ``$RESOURCE_ENDPOINT/openai/v1``).

    Routing: Versa serves Responses on its OpenAI-compatible ``/openai/v1``
    surface, so the request is sent through litellm's ``openai/`` provider
    regardless of how the model was spelled for the Azure chat-completions path
    (an ``azure/...`` id is normalized to ``openai/...``). The deployment-less
    Azure route litellm would otherwise build (``.../openai/responses?api-version=``)
    is not exposed by the gateway and returns 404. This path has been confirmed
    against the live Versa endpoint with a GPT-5 reasoning call.

    Note: tool-calling and public ``stream=True`` are not supported through this
    path (both raise); for tools use ``make_versa_openai_completion``.

    Args:
        completion_func: Responses function to wrap. Defaults to litellm.responses.

    Returns:
        A CompletionFunction returning an aggregated ModelResponse.
    """
    endpoint = endpoint or os.getenv("VERSA_RESPONSES_ENDPOINT")
    api_key = api_key or os.getenv("VERSA_API_KEY")
    # The OpenAI-compatible /openai/v1 surface does not use an api-version, so we
    # do not inject one by default. The override stays available for a future
    # surface that needs it, but it is forwarded only when explicitly set.
    api_version = api_version or os.getenv("VERSA_RESPONSES_API_VERSION")

    if not endpoint:
        raise ValueError(
            "Versa Responses endpoint must be provided via the "
            "VERSA_RESPONSES_ENDPOINT environment variable "
            "(e.g. https://<resource>/openai/v1)."
        )
    if not api_key:
        raise ValueError("API key must be provided via VERSA_API_KEY environment variable")

    logging.info("Versa OpenAI Responses Endpoint URL: %s", endpoint)

    request_auth = {"api_key": api_key, "api_base": endpoint}
    if api_version:
        request_auth["api_version"] = api_version

    if completion_func is None:
        import litellm
        completion_func = litellm.responses

    def wrapper(model: str, messages: Optional[List] = None, **kwargs) -> ModelResponse:
        if messages is None:
            messages = []
        request_kwargs = _chat_kwargs_to_responses_kwargs(model, kwargs)
        stream = completion_func(
            model=_to_responses_provider_model(model),
            input=_messages_to_responses_input(messages),
            stream=True,
            **request_kwargs,
            **request_auth,
        )
        final_response, streamed_text = _consume_responses_stream(stream)
        return _responses_to_model_response(model, final_response, streamed_text)

    return wrapper