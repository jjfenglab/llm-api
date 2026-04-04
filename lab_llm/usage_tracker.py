"""
UsageTracker wrapper for completion functions.

Implements the CompletionFunctionWrapper protocol to provide token usage tracking
for LLM completion calls. Works as a decorator around any CompletionFunction.
"""

from functools import wraps
from typing import Optional, List, Dict
from litellm import ModelResponse

from .types import CompletionFunction, CompletionFunctionWrapper


class UsageTracker(CompletionFunctionWrapper):
    """
    A wrapper that tracks token usage from completion function calls.

    Accumulates token usage data across multiple calls and provides methods
    to access the most recent usage and total cumulative usage.

    Example usage:
        completion = UsageTracker()(litellm.completion)

        # Make some calls
        response1 = completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])
        response2 = completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi again"}])

        # Access usage data
        last_usage = completion.last_usage()  # Usage from response2
        total_usage = completion.total_usage()  # Cumulative usage from all calls

    Can be composed with other wrappers:
        completion = UsageTracker()(
            CachingCompletion("./cache.db")(
                litellm.completion
            )
        )
    """

    def __init__(self):
        """Initialize the usage tracker."""
        self._usage_data: List[Dict] = []

    def __call__(self, func: CompletionFunction) -> CompletionFunction:
        """
        Implements CompletionFunctionWrapper protocol.

        Args:
            func: The completion function to wrap

        Returns:
            Wrapped completion function with usage tracking
        """
        @wraps(func)
        def wrapped_completion(model: str, messages: List = None, **kwargs) -> ModelResponse:
            if messages is None:
                messages = []

            # Call the wrapped function
            response = func(model, messages=messages, **kwargs)

            # Extract and store usage data
            usage_data = self._extract_usage_from_response(response)
            if usage_data:
                self._usage_data.append(usage_data)

            return response

        return wrapped_completion

    def last_usage(self) -> Optional[Dict]:
        """
        Return the most recently captured usage data, or None if no calls made.

        Returns:
            Dict with keys: input_tokens, output_tokens, total_tokens,
            and optionally cached_tokens and reasoning_tokens
        """
        if not self._usage_data:
            return None
        return self._usage_data[-1]

    def total_usage(self) -> Optional[Dict]:
        """
        Return cumulative usage data across all calls, or None if no calls made.

        Returns:
            Dict with keys: input_tokens, output_tokens, total_tokens,
            and optionally cached_tokens and reasoning_tokens (all summed)
        """
        if not self._usage_data:
            return None

        # Sum all usage data
        total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        optional_fields = {"cached_tokens", "reasoning_tokens"}
        for field in optional_fields:
            total[field] = 0

        for usage in self._usage_data:
            total["input_tokens"] += usage.get("input_tokens", 0)
            total["output_tokens"] += usage.get("output_tokens", 0)
            total["total_tokens"] += usage.get("total_tokens", 0)

            for field in optional_fields:
                if field in usage:
                    total[field] += usage[field]

        # Remove optional fields if they're zero
        for field in optional_fields:
            if total[field] == 0:
                del total[field]

        return total

    def _extract_usage_from_response(self, response: ModelResponse) -> Optional[Dict]:
        """
        Extract usage data from a ModelResponse.

        Uses the same logic as the original UsageCallbackHandler.

        Args:
            response: The ModelResponse to extract usage from

        Returns:
            Usage dict or None if no usage data found
        """
        # Source 1: response.usage (direct usage attribute on ModelResponse)
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0) or getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0) or getattr(usage, 'output_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', 0)

            result = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or (input_tokens + output_tokens),
            }

            # Check for detailed token info (OpenAI format)
            if hasattr(usage, 'prompt_tokens_details'):
                prompt_details = usage.prompt_tokens_details or {}
                cached = getattr(prompt_details, 'cached_tokens', 0) if prompt_details else 0
                if cached:
                    result["cached_tokens"] = cached

            if hasattr(usage, 'completion_tokens_details'):
                completion_details = usage.completion_tokens_details or {}
                reasoning = getattr(completion_details, 'reasoning_tokens', 0) if completion_details else 0
                if reasoning:
                    result["reasoning_tokens"] = reasoning

            return result

        # Source 2: Check choices for usage_metadata (LangChain format)
        if hasattr(response, 'choices') and response.choices:
            for choice in response.choices:
                if hasattr(choice, 'message') and hasattr(choice.message, 'usage_metadata'):
                    usage = choice.message.usage_metadata
                    if usage:
                        return self._parse_usage_metadata(usage)

        return None

    def _parse_usage_metadata(self, usage: Dict) -> Optional[Dict]:
        """
        Parse a LangChain usage_metadata dict into a standard format.

        This is adapted from the original UsageCallbackHandler.parse_usage_metadata method.

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