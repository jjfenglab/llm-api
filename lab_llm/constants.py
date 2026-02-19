from enum import Enum
from typing import Union

from pydantic import BaseModel

VERSA_API_VERSION = "2024-10-21"
AWS_REGION = "us-west-2"


class VersaOpenAi(str, Enum):
    GPT4_O_2024_08 = "gpt-4o-2024-08-06"
    GPT4_O_MINI_2024_07 = "gpt-4o-mini-2024-07-18"
    GPT4_O_2024_11 = "gpt-4o-2024-11-20"
    GPT4_O_MINI_2024_11 = "gpt-4o-mini-2024-11-20"
    GPT5_2025_08 = "gpt-5-2025-08-07"
    GPT5_MINI_2025_08 = "gpt-5-mini-2025-08-07"
    GPT5_NANO_2025_08 = "gpt-5-nano-2025-08-07"


class OpenAi(str, Enum):
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_O = "gpt-4o"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"


class Cohere(str, Enum):
    COMMAND_R = "cohere-command-r"
    COMMAND = "cohere-command"
    COMMAND_LIGHT = "cohere-command-light"


class Claude(str, Enum):
    HAIKU_4_5 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    SONNET_4 = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    OPUS_4_5 = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    HAIKU_3 = "claude-haiku-3"
    HAIKU_3_5 = "anthropic.claude-3-5-haiku-20241022-v1:0"
    SONNET_4_5 = "anthropic.claude-sonnet-4-5-20250929-v1:0"


class Meta(str, Enum):
    LLAMA_3_2_90B = "meta-llama/Meta-Llama-3.2-90B-Instruct"
    LLAMA_3_3_70B = "us.meta.llama3-3-70b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"


class Qwen(str, Enum):
    QWEN_3_235 = "qwen.qwen3-235b-a22b-2507-v1:0"


class LocalOpenAi(str, Enum):
    """Sentinel enum for models served via a local OpenAI-compatible endpoint.
    The actual model name is passed separately via local_model_name."""
    LOCAL = "local"


BedrockModels = Union[Cohere, Claude, Meta, Qwen]
Models = Union[VersaOpenAi, BedrockModels, OpenAi, LocalOpenAi]


class LLMModel(BaseModel):
    name: Models


BEDROCK_MAPPINGS = {
    Cohere.COMMAND_R: "cohere.command-r-v1:0",
    Cohere.COMMAND: "cohere.command-text-v14",
    Cohere.COMMAND_LIGHT: "cohere.command-light-text-v14",
    Claude.HAIKU_3: "anthropic.claude-3-haiku-20240307-v1:0",
    Claude.HAIKU_3_5: "anthropic.claude-3-5-haiku-20241022-v1:0",
    Claude.HAIKU_4_5: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    Claude.SONNET_4: "us.anthropic.claude-sonnet-4-20250514-v1:0",
    Claude.OPUS_4_5: "us.anthropic.claude-opus-4-5-20251101-v1:0",
    Claude.SONNET_4_5: "anthropic.claude-sonnet-4-5-20250929-v1:0",
    Meta.LLAMA_3_2_90B: "us.meta.llama3-2-90b-instruct-v1:0",
    Meta.LLAMA_3_3_70B: "us.meta.llama3-3-70b-instruct-v1:0",
    Meta.LLAMA_3_2_11B: "us.meta.llama3-2-11b-instruct-v1:0",
    Qwen.QWEN_3_235: "qwen.qwen3-235b-a22b-2507-v1:0",
}
is_bedrock = lambda x: isinstance(x, BedrockModels)
is_meta = lambda x: isinstance(x, Meta)
is_versa = lambda x: isinstance(x, VersaOpenAi)
is_openai = lambda x: isinstance(x, OpenAi)
is_local_openai = lambda x: isinstance(x, LocalOpenAi)

# Reasoning models support reasoning_effort and verbosity parameters
REASONING_MODELS = {
    OpenAi.GPT5,
    OpenAi.GPT5_MINI,
    OpenAi.GPT5_NANO,
    VersaOpenAi.GPT5_2025_08,
    VersaOpenAi.GPT5_MINI_2025_08,
    VersaOpenAi.GPT5_NANO_2025_08,
}
is_reasoning_model = lambda x: x in REASONING_MODELS


def get_all_model_enums():
    """Get all model enum classes."""
    return [VersaOpenAi, OpenAi, Cohere, Claude, Meta, Qwen, LocalOpenAi]


def list_available_models() -> list:
    """List all available model string values.

    Returns:
        List of model string values (e.g., 'gpt-4o-2024-08-06')
    """
    models = []
    for enum_class in get_all_model_enums():
        models.extend([member.value for member in enum_class])
    return models


def _build_model_mapping() -> dict:
    """Build mapping from model strings to enum members.

    Maps both enum values (e.g., 'gpt-4o-2024-08-06') and enum names
    (e.g., 'GPT4_O_2024_08') to their corresponding enum members.
    """
    mapping = {}
    for enum_class in get_all_model_enums():
        for member in enum_class:
            mapping[member.value] = member  # e.g., 'gpt-4o-2024-08-06' -> VersaOpenAi.GPT4_O_2024_08
            mapping[member.name] = member   # e.g., 'GPT4_O_2024_08' -> VersaOpenAi.GPT4_O_2024_08
    return mapping


MODEL_MAPPING = _build_model_mapping()


def parse_model_string(model_str: str) -> LLMModel:
    """Parse model string to LLMModel instance.

    Args:
        model_str: Model string - either enum value (e.g., 'gpt-4o-2024-08-06')
                   or enum name (e.g., 'GPT4_O_2024_08')

    Returns:
        LLMModel instance

    Raises:
        ValueError: If model string is not recognized
    """
    assert model_str in MODEL_MAPPING, f"Unsupported model: {model_str}. Available models: {list_available_models()}"
    return LLMModel(name=MODEL_MAPPING[model_str])
