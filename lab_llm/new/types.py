from typing import TypedDict, Literal, Optional, Any, List, Union, Protocol
from collections.abc import Sequence
from litellm import ModelResponse

class ToolCallFunction(TypedDict):
    """Follows OpenAI ChatCompletionMessageFunctionToolCall.function."""
    arguments: str
    name: str

class ToolCall(TypedDict):
    """Follows OpenAI ChatCompletionMessageFunctionToolCall."""
    id: str
    function: ToolCallFunction
    type: Literal["function"]

class ContentPart(TypedDict):
    """Follows OpenAI ChatCompletionContentPart."""
    type: Literal["text", "image", "input_audio", "file"]
    text: Optional[str]
    image_url: Optional[str]
    input_audio: Optional[Any]
    file: Optional[Any]

class FunctionDefinition(TypedDict):
    """Standard function tool definition, follows OpenAI FunctionDefinition."""
    name: str
    description: str
    parameters: dict # JSON Schema
    strict: bool = False

class FunctionTool(TypedDict):
    """Follows OpenAI ChatCompletionFunctionTool."""
    function: FunctionDefinition
    type: Literal["function"]

class CompletionFunction(Protocol):
    """Protocol for litellm.completion function signature."""

    def __call__(
        self,
        model: str,
        messages: List = [],
        # Optional OpenAI params
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop: Any = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        # openai v1.0+ new params
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        safety_identifier: Optional[str] = None,
        deployment_id: Any = None,
        # soon to be deprecated params by OpenAI
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        # set api_base, api_version, api_key
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,
        # Optional liteLLM function params
        **kwargs: Any,
    ) -> ModelResponse:
        ...

class CompletionFunctionWrapper(Protocol):
    """Protocol for a wrapper that can decorate a completion function."""
    def __call__(self, func: CompletionFunction) -> CompletionFunction:
        ...