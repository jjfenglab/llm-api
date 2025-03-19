from pydantic import BaseModel
from enum import Enum
from typing import Union

VERSA_ENDPOINT = "https://unified-api.ucsf.edu/general/openai/deployments/<model_name>/chat/completions?api-version=2024-10-21"
VERSA_API_VERSION = "2024-10-21"
AWS_REGION='us-west-2'

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

class Meta(str, Enum):
    LLAMA_3_2_90B = "meta-llama/Meta-Llama-3.2-90B-Instruct"
    LLAMA_3_3_70B = "us.meta.llama3-3-70b-instruct-v1:0"
    LLAMA_3_2_11B = "us.meta.llama3-2-11b-instruct-v1:0"

BedrockModels = Union[Cohere, Claude, Meta]
Models = Union[VersaOpenAi, BedrockModels, OpenAi]

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
