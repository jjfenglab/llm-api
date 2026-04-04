import asyncio
import json
from typing import Any, List, Optional, Union, Callable, Unpack
from collections.abc import Sequence
import logging
from functools import partial

from litellm import ModelResponse, Message
from pydantic import BaseModel

from .types import CompletionFunction, CompletionFunctionWrapper, FunctionTool, ToolCall, CompletionKwargs
from .utils import make_function_tool, normalize_messages, normalize_tools
from .parameter_wrappers import ModelRamp, ModelDefault
from .caching_completion import CachingCompletion
from .error_tracker import ErrorTracker
from .usage_tracker import UsageTracker

class ToolExecutionError(Exception):
    pass

logger = logging.getLogger(__name__)

def wrap_completion_function(func: CompletionFunction,
                             cache: Optional[CachingCompletion] = None,
                             model_ramp: Optional[ModelRamp] = None,
                             default_model_name: Optional[str] = None,
                             error_tracker: Optional[ErrorTracker] = None,
                             usage_tracker: Optional[UsageTracker] = None) -> CompletionFunction:
    """
    Sets up a completion function with optional wrappers to extend
    its functionality.
    """
    if cache is not None:
        func = cache(func)
    if default_model_name is not None:
        assert model_ramp is None, "Cannot use both model_ramp and default_model_name"
        func = ModelDefault(default_model_name)(func)
    if model_ramp is not None:
        func = model_ramp(func)
    if error_tracker is not None:
        func = error_tracker(func)
    if usage_tracker is not None:
        func = usage_tracker(func)
    return func


class LLMApi:
    def __init__(self, completion_function: CompletionFunction):
        self.completion = completion_function

    def _execute_tool_call(self, tool_call: ToolCall, tools: List[Union[Callable, FunctionTool]]) -> str:
        func_name = tool_call["function"]["name"]

        # Find the matching tool
        for tool in tools:
            if callable(tool) and not isinstance(tool, dict):
                if tool.__name__ == func_name:
                    args = json.loads(tool_call["function"]["arguments"])
                    return str(tool(**args))
            elif isinstance(tool, dict) and tool.get("function", {}).get("name") == func_name:
                # For FunctionTool dicts, we can't execute them directly
                raise ToolExecutionError(f"Cannot execute FunctionTool dict for {func_name}. Only callable functions can be executed.")

        raise ToolExecutionError(f"Tool {func_name} not found in provided tools")

    def run(
        self,
        messages: Union[str, List[Union[str, dict, Message]]],
        tools: Optional[List[Union[Callable, FunctionTool]]] = None,
        max_tool_calls: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Unpack[CompletionKwargs]
    ) -> Any:
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)

        tool_call_count = 0

        while True:
            response = self.completion(
                model,
                messages=normalized_messages,
                tools=normalized_tools,
                **kwargs
            )

            message = response.choices[0].message

            normalized_messages.append(message.model_dump() if hasattr(message, 'model_dump') else dict(message))

            tool_calls = getattr(message, 'tool_calls', None)
            if not tool_calls:
                # No more tool calls, return the final result
                content = message.content
                response_format = kwargs.get('response_format')

                if response_format and hasattr(response_format, '__annotations__'):
                    # It's a Pydantic model class
                    try:
                        return response_format.model_validate_json(content)
                    except Exception:
                        # Fallback to string if parsing fails
                        return content
                elif response_format:
                    # It's a dict schema, return as string (let caller handle)
                    return content
                else:
                    return content

            # Execute tool calls
            if max_tool_calls is not None and tool_call_count >= max_tool_calls:
                raise ToolExecutionError(f"Maximum tool calls ({max_tool_calls}) exceeded")

            for tool_call in tool_calls:
                logger.debug("Executing tool: %s", tool_call)
                try:
                    result = self._execute_tool_call(tool_call, tools or [])
                    normalized_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
                    tool_call_count += 1
                except Exception as e:
                    logger.warning("Error executing tool %s: %s", tool_call, e)
                    normalized_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Error executing tool: {str(e)}"
                    })
                    tool_call_count += 1

    async def _run_single_async(
        self,
        messages: Union[str, List[Union[str, dict, Message]]],
        tools: Optional[List[Union[Callable, FunctionTool]]] = None,
        max_tool_calls: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Unpack[CompletionKwargs]
    ) -> Any:
        # Run the synchronous version in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self.run, 
                messages,
                max_tool_calls=max_tool_calls,
                model=model,
                **kwargs
            )
        )

    async def run_batch(
        self,
        messages_list: List[Union[str, List[Union[str, dict, Message]]]],
        tools: Optional[List[Union[Callable, FunctionTool]]] = None,
        max_tool_calls: Optional[int] = None,
        max_parallel_jobs: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs: Unpack[CompletionKwargs]
    ) -> List[Any]:
        if max_parallel_jobs is None:
            # Use asyncio.gather for unlimited parallelism
            tasks = [
                self._run_single_async(messages, tools, max_tool_calls, model=model, **kwargs)
                for messages in messages_list
            ]
            return await asyncio.gather(*tasks)
        else:
            # Use asyncio.Semaphore with Queue for limited parallelism
            semaphore = asyncio.Semaphore(max_parallel_jobs)
            results = [None] * len(messages_list)

            async def process_item(index: int, messages: Union[str, List[Union[str, dict, Message]]]):
                async with semaphore:
                    results[index] = await self._run_single_async(messages, tools, max_tool_calls, model=model, **kwargs)

            tasks = [
                process_item(i, messages)
                for i, messages in enumerate(messages_list)
            ]
            await asyncio.gather(*tasks)
            return results