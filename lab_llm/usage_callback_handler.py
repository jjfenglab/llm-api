from typing import Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageCallbackHandler(BaseCallbackHandler):
    """Callback handler that captures token usage metadata from LLM responses.

    Designed to work alongside ErrorCallbackHandler. When registered as a
    callback on a LangChain client, it captures token usage from each LLM
    call via the on_llm_end hook.

    Works with structured output (with_structured_output) because the callback
    fires on the inner LLM call before output parsing.

    Note: This handler is not thread-safe. The reset/on_llm_end/last_usage
    cycle uses mutable instance state, so concurrent get_output() calls on the
    same LLMApi instance from multiple threads may produce incorrect results.
    For concurrent usage, instantiate separate LLMApi instances per thread.
    """

    def __init__(self):
        super().__init__()
        self._usage_data: Optional[dict] = None

    def reset(self):
        """Clear captured usage data. Call before each new LLM invocation."""
        self._usage_data = None

    @property
    def last_usage(self) -> Optional[dict]:
        """Return the most recently captured usage data, or None."""
        return self._usage_data

    @staticmethod
    def parse_usage_metadata(usage: dict) -> Optional[dict]:
        """Parse a LangChain usage_metadata dict into a standard format.

        This is the shared extraction logic used by both the callback handler
        (for single calls) and _run_batch (for batch calls), so parsing
        stays in one place.

        Args:
            usage: A dict with keys like 'input_tokens', 'output_tokens',
                   and optionally 'input_token_details', 'output_token_details'.

        Returns:
            A normalized dict with input_tokens, output_tokens, total_tokens,
            and optionally cached_tokens and reasoning_tokens. Returns None
            if the input is None or empty.
        """
        if not usage:
            return None

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        input_details = usage.get("input_token_details") or {}
        output_details = usage.get("output_token_details") or {}
        cached = input_details.get("cache_read", 0)
        reasoning = output_details.get("reasoning", 0)

        # Use provider's total_tokens (includes reasoning) when available
        result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        }
        if cached:
            result["cached_tokens"] = cached
        if reasoning:
            result["reasoning_tokens"] = reasoning

        return result

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Capture token usage from the LLM response.

        Tries two sources:
        1. response.llm_output['token_usage'] (OpenAI/Azure format)
        2. generation.message.usage_metadata (LangChain normalized format)
        """
        # Source 1: llm_output dict (OpenAI/Azure providers)
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage")
            if token_usage:
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                self._usage_data = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
                # OpenAI details are under prompt_tokens_details / completion_tokens_details
                prompt_details = token_usage.get("prompt_tokens_details") or {}
                completion_details = token_usage.get("completion_tokens_details") or {}
                cached = prompt_details.get("cached_tokens", 0)
                reasoning = completion_details.get("reasoning_tokens", 0)
                if cached:
                    self._usage_data["cached_tokens"] = cached
                if reasoning:
                    self._usage_data["reasoning_tokens"] = reasoning
                return

        # Source 2: generation message's usage_metadata (normalized format)
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    usage = getattr(msg, "usage_metadata", None) if msg else None
                    if usage:
                        self._usage_data = self.parse_usage_metadata(usage)
                        return
