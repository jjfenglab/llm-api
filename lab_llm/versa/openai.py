import os
from typing import Callable, Any, Optional
from ..constants import VersaOpenAI, VERSA_API_VERSION
from ..types import CompletionFunction
from ..parameter_wrappers import ModelRamp
import logging
from functools import wraps

VersaOpenAIModelRamp = ModelRamp([
    VersaOpenAI.GPT4_O_2024_11,
    VersaOpenAI.GPT5_NANO_2025_08,
    VersaOpenAI.GPT5_MINI_2025_08,
    VersaOpenAI.GPT5_2025_08,
    VersaOpenAI.O_1_2024_12
])

def versa_openai_completion(completion_func: Optional[CompletionFunction] = None, api_key: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> CompletionFunction:
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