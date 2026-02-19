"""
Feng Lab LLM API - A unified interface for multiple LLM providers.

This library provides a consistent API for interacting with various Large Language
Model providers including OpenAI, AWS Bedrock, and Azure OpenAI (Versa).

Example usage:
    from lab_llm import LLMApi, LLMCache, DuckDBHandler
    from lab_llm.constants import OpenAi, LLMModel

    db_handle = DuckDBHandler("./cache.db")
    cache = LLMCache(db_handle)
    model = LLMModel(name=OpenAi.GPT4_O_MINI)
    api = LLMApi(cache=cache, seed=42, model_type=model)
    response = api.get_output("Hello, world!")
"""

from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.error_tracker import ErrorTracker
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.dataset import TextDataset, ImageDataset
from lab_llm.constants import (
    LLMModel,
    OpenAi,
    VersaOpenAi,
    Claude,
    Meta,
    Cohere,
    Qwen,
    LocalOpenAi,
    REASONING_MODELS,
    is_reasoning_model,
    is_local_openai,
    list_available_models,
    parse_model_string,
)

__version__ = "0.1.5"

__all__ = [
    # Core classes
    "LLMApi",
    "LLMCache",
    "DuckDBHandler",
    "ErrorTracker",
    "ErrorCallbackHandler",
    # Dataset classes
    "TextDataset",
    "ImageDataset",
    # Model configuration
    "LLMModel",
    # Model enums
    "OpenAi",
    "VersaOpenAi",
    "Claude",
    "Meta",
    "Cohere",
    "Qwen",
    "LocalOpenAi",
    # Reasoning model utilities
    "REASONING_MODELS",
    "is_reasoning_model",
    "is_local_openai",
    # Model helpers
    "list_available_models",
    "parse_model_string",
]
