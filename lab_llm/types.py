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

class FunctionToolDict(TypedDict):
    """Follows OpenAI ChatCompletionFunctionTool."""
    function: FunctionDefinition
    type: Literal["function"]

class MessageDict(TypedDict):
    """Standard message format, similar to LiteLLM's Message but as a TypedDict."""
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str]
    structured_content: Optional[Any]
    tool_calls: Optional[List[ToolCall]]

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

class CompletionKwargs(TypedDict):
    # Optional OpenAI params
    timeout: Optional[Union[float, int]]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    stream_options: Optional[dict]
    stop: Any
    max_completion_tokens: Optional[int]
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[dict]
    user: Optional[str]
    # openai v1.0+ new params
    response_format: Optional[dict]
    seed: Optional[int]
    tools: Optional[List]
    tool_choice: Optional[str]
    parallel_tool_calls: Optional[bool]
    logprobs: Optional[bool]
    top_logprobs: Optional[int]
    safety_identifier: Optional[str]
    deployment_id: Any
    # soon to be deprecated params by OpenAI
    functions: Optional[List]
    function_call: Optional[str]
    # set api_base, api_version, api_key
    base_url: Optional[str]
    api_version: Optional[str]
    api_key: Optional[str]
    model_list: Optional[list]

class Tool(Protocol):
    """
    A type that is able to produce a JSON schema of its inputs and outputs
    for an LLM call, and that can be called directly with a set of kwargs
    to produce outputs.
    """
    @property
    def name(self) -> str:
        ...

    def to_json_schema(self) -> FunctionToolDict:
        """
        Convert the tool's inputs to a JSON schema dictionary.

        Returns:
            JSON schema dict representing the tool's input parameters
        """
        ...
    
    def __call__(self, **kwargs) -> Any:
        """
        Run the tool.
        """
        ...