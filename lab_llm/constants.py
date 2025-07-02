from enum import Enum
from typing import Union

from pydantic import BaseModel

VERSA_ENDPOINT = "https://unified-api.ucsf.edu/general/openai/deployments/<model_name>/chat/completions?api-version=2024-10-21"
VERSA_API_VERSION = "2024-10-21"
AWS_REGION = "us-west-2"


class VersaOpenAi(str, Enum):
    GPT4_O_2024_08 = "gpt-4o-2024-08-06"
    GPT4_O_MINI_2024_07 = "gpt-4o-mini-2024-07-18"


class OpenAi(str, Enum):
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_O = "gpt-4o"


class Cohere(str, Enum):
    COMMAND_R = "cohere-command-r"
    COMMAND = "cohere-command"
    COMMAND_LIGHT = "cohere-command-light"


class Claude(str, Enum):
    HAIKU_3 = "claude-haiku-3"
    HAIKU_3_5 = "anthropic.claude-3-5-haiku-20241022-v1:0"
    SONNET = "claude-sonnet"


class Anthropic(str, Enum):
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"


class Meta(str, Enum):
    LLAMA_3_2_90B = "meta-llama/Meta-Llama-3.2-90B-Instruct"
    LLAMA_3_3_70B = "us.meta.llama3-3-70b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"


BedrockModels = Union[Cohere, Claude, Meta]
Models = Union[VersaOpenAi, BedrockModels, OpenAi, Anthropic]


class LLMModel(BaseModel):
    name: Models


BEDROCK_MAPPINGS = {
    Cohere.COMMAND_R: "cohere.command-r-v1:0",
    Cohere.COMMAND: "cohere.command-text-v14",
    Cohere.COMMAND_LIGHT: "cohere.command-light-text-v14",
    Claude.HAIKU_3: "anthropic.claude-3-haiku-20240307-v1:0",
    Claude.HAIKU_3_5: "anthropic.claude-3-5-haiku-20241022-v1:0",
    Claude.SONNET: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    Meta.LLAMA_3_2_90B: "us.meta.llama3-2-90b-instruct-v1:0",
    Meta.LLAMA_3_3_70B: "us.meta.llama3-3-70b-instruct-v1:0",
    Meta.LLAMA_3_2_11B: "us.meta.llama3-2-11b-instruct-v1:0",
}
is_bedrock = lambda x: isinstance(x, BedrockModels)
is_meta = lambda x: isinstance(x, Meta)
is_versa = lambda x: isinstance(x, VersaOpenAi)
is_openai = lambda x: isinstance(x, OpenAi)
is_anthropic = lambda x: isinstance(x, Anthropic)


# keeping for backward compatibility with LLM-CBM code
def convert_to_llm_type(model_name: str) -> LLMModel:
    """Convert a string model name back to LLMModel."""
    # Check all enum classes for the model name
    all_enums = [VersaOpenAi, OpenAi, Cohere, Claude, Anthropic, Meta]

    for enum_class in all_enums:
        for enum_value in enum_class:
            if enum_value.value == model_name:
                return LLMModel(name=enum_value)

    raise ValueError(f"Unknown model name: {model_name}")
