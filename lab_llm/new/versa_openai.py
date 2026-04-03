import os
from typing import Callable, Any, Optional
from .types import CompletionFunction
from .parameter_wrappers import ModelRamp
import logging
from functools import wraps

class VersaOpenAIModels:
    GPT4_O_2024_08 = "gpt-4o-2024-08-06"
    GPT4_O_MINI_2024_07 = "gpt-4o-mini-2024-07-18"
    GPT4_O_2024_11 = "gpt-4o-2024-11-20"
    GPT4_O_MINI_2024_11 = "gpt-4o-mini-2024-11-20"
    GPT5_2025_08 = "gpt-5-2025-08-07"
    GPT5_MINI_2025_08 = "gpt-5-mini-2025-08-07"
    GPT5_NANO_2025_08 = "gpt-5-nano-2025-08-07"
    O_1_2024_12 = "o1-2024-12-17"
    O_4_MINI_2025_04 = "o4-mini-2025-04-16"

DefaultVersaModelRamp = ModelRamp([
    VersaOpenAIModels.GPT4_O_2024_11,
    VersaOpenAIModels.GPT5_NANO_2025_08,
    VersaOpenAIModels.GPT5_MINI_2025_08,
    VersaOpenAIModels.GPT5_2025_08,
    VersaOpenAIModels.O_1_2024_12
])

VERSA_API_VERSION = "2024-12-01-preview"

def versa_openai_completion(completion_func: Optional[CompletionFunction] = None, api_key: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> CompletionFunction:
    """
    Wrapper around litellm.completion that configures Azure OpenAI parameters from environment variables.

    Maps environment variables:
    - VERSA_ENDPOINT -> azure_endpoint (with /openai/ appended)
    - VERSA_API_KEY -> api_key
    - Uses VERSA_API_VERSION (defaults to "2024-10-21") -> api_version

    Args:
        completion_func: Function to wrap

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
        return completion_func("azure/" + model, **{**kwargs, **azure_kwargs})
    return wrapper