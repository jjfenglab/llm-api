import os
import asyncio
import concurrent.futures
import json
from typing_extensions import Unpack
from typing import Any, List, Optional, Union, Callable
from collections.abc import Sequence, Mapping
import logging
from functools import partial

from litellm import ModelResponse, Message
from pydantic import BaseModel

from .types import CompletionFunction, CompletionFunctionWrapper, FunctionToolDict, ToolCall, CompletionKwargs, Tool
from .utils import normalize_messages, normalize_tools
from .parameter_wrappers import ModelRamp, DefaultParameters
from .caching_completion import CachingCompletion
from .error_tracker import ErrorTracker
from .usage_tracker import UsageTracker

class ToolExecutionError(Exception):
    pass

logger = logging.getLogger(__name__)

def _default_max_parallel_jobs() -> int:
    """CPU count available to this process: respects scheduler affinity on 3.13+ (os.process_cpu_count), so a slurm allocation doesn't default to the whole node."""
    return getattr(os, "process_cpu_count", os.cpu_count)() or 4

def wrap_completion_function(func: CompletionFunction,
                             cache: Optional[CachingCompletion] = None,
                             model_ramp: Optional[ModelRamp] = None,
                             error_tracker: Optional[ErrorTracker] = None,
                             usage_tracker: Optional[UsageTracker] = None,
                             model: Optional[str] = None,
                             **defaults: Unpack[CompletionKwargs]) -> CompletionFunction:
    """
    Sets up a completion function with optional wrappers to extend
    its functionality.
    """
    if cache is not None:
        func = cache(func)
    assert (model_ramp is None or model is None), "Cannot use both model_ramp and default model"
    if model is not None or defaults:
        func = DefaultParameters(model=model, **defaults)(func)
    if model_ramp is not None:
        func = model_ramp(func)
    if error_tracker is not None:
        func = error_tracker(func)
    if usage_tracker is not None:
        func = usage_tracker(func)
    return func


class LLMApi:
    """
    Usage:
    
    ```
    api = LLMApi(wrap_completion_function(litellm.completion, cache=..., model=Claude.SONNET_4), track_usage=True)
    response = api.run("What is the capital of France?")
    print(response, api.usage_tracker.last_usage())
    ```
    """
    def __init__(self, completion_function: CompletionFunction, track_usage: bool = True):
        if track_usage:
            self.usage_tracker = UsageTracker()
            completion_function = self.usage_tracker(completion_function)
        else:
            self.usage_tracker = None
        self.completion = completion_function

    def _execute_tool_call(self, tool_call: ToolCall, tools: Mapping[str, Tool]) -> str:
        func_name = tool_call["function"]["name"]

        if func_name not in tools:
            raise ToolExecutionError(f"Tool {func_name} not found in provided tools")
        
        args = json.loads(tool_call["function"]["arguments"])
        return str(tools[func_name](**args))

    def run(
        self,
        messages: Union[str, List[Union[str, dict, Message]]],
        tools: Optional[List[Union[Callable, Tool, FunctionToolDict]]] = None,
        max_tool_calls: Optional[int] = None,
        model: Optional[str] = None,
        strict_response_format: bool = False,
        **kwargs: Unpack[CompletionKwargs]
    ) -> Any:
        """
        Run a single completion request, executing tool calls in a loop until the
        model returns a final response with no pending tool calls.

        Parameters
        ----------
        messages : str or list of (str | dict | Message)
            The conversation input. A plain string is treated as a single user
            message. A list may contain strings, raw message dicts, or litellm
            ``Message`` objects.
        tools : list of (Callable | Tool | FunctionToolDict), optional
            Tools the model may invoke. Callables are introspected to build a
            JSON schema; ``Tool``/``FunctionToolDict`` objects are used as-is.
        max_tool_calls : int, optional
            Maximum number of tool calls to execute before stopping. Once the
            limit is reached, tools are omitted from subsequent requests so the
            model is forced to produce a final answer. If ``None``, tool calls
            are unlimited.
        model : str, optional
            Model identifier to pass to the underlying completion function (e.g.
            ``"gpt-4o"``). Overrides any model set on the completion function.
        strict_response_format : bool, optional
            If True, raise instead of falling back to the raw content string
            when a Pydantic ``response_format`` fails to parse.
        **kwargs : CompletionKwargs
            Additional keyword arguments forwarded directly to the completion
            function (e.g. ``temperature``, ``max_tokens``, ``response_format``).

        Returns
        -------
        Any
            The model's final response. If ``response_format`` is a Pydantic
            model class, returns a validated instance of that class. Otherwise
            returns the raw content string.
        """
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)

        tool_call_count = 0

        while True:
            if max_tool_calls is not None and tool_call_count >= max_tool_calls:
                tools_for_call = None
            else:
                tools_for_call = [tool.to_json_schema() for tool in normalized_tools.values()]

            response = self.completion(
                model,
                messages=normalized_messages,
                tools=tools_for_call,
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
                        if strict_response_format:
                            raise
                        # Fallback to string if parsing fails
                        return content
                elif response_format:
                    # It's a dict schema, return as string (let caller handle)
                    return content
                else:
                    return content

            for tool_call in tool_calls:
                logger.debug("Executing tool: %s", tool_call)
                try:
                    result = self._execute_tool_call(tool_call, normalized_tools)
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
        tools: Optional[List[Union[Callable, FunctionToolDict]]] = None,
        max_tool_calls: Optional[int] = None,
        model: Optional[str] = None,
        strict_response_format: bool = False,
        executor: Optional[concurrent.futures.ThreadPoolExecutor] = None,
        **kwargs: Unpack[CompletionKwargs]
    ) -> Any:
        # Run the synchronous version in an executor to avoid blocking.
        # executor=None falls back to the asyncio default executor, whose
        # min(32, cpu_count + 4) thread cap silently limits concurrency;
        # run_batch always passes a dedicated executor for this reason.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            partial(
                self.run,
                messages,
                tools=tools,
                max_tool_calls=max_tool_calls,
                model=model,
                strict_response_format=strict_response_format,
                **kwargs
            )
        )

    async def run_batch(
        self,
        messages_list: List[Union[str, List[Union[str, dict, Message]]]],
        tools: Optional[List[Union[Callable, FunctionToolDict]]] = None,
        max_tool_calls: Optional[int] = None,
        max_parallel_jobs: Optional[int] = None,
        sleep_interval: Optional[float] = None,
        model: Optional[str] = None,
        return_exceptions: bool = False,
        strict_response_format: bool = False,
        **kwargs: Unpack[CompletionKwargs]
    ) -> List[Any]:
        """
        Run multiple completion requests concurrently, preserving input order in
        the returned results.

        Parameters
        ----------
        messages_list : list of (str | list of (str | dict | Message))
            One entry per request. Each entry follows the same format accepted
            by ``run()``: a plain string or a list of message objects.
        tools : list of (Callable | FunctionToolDict), optional
            Tools available to the model for every request in the batch. See
            ``run()`` for accepted formats.
        max_tool_calls : int, optional
            Maximum tool calls per individual request. Passed unchanged to each
            ``run()`` invocation.
        max_parallel_jobs : int, optional
            Maximum number of requests to run concurrently; each unit is one
            worker thread issuing a blocking ``run()`` call, so values in the
            low hundreds are fine for I/O-bound API calls. Must be in
            [1, 512]; raises ``ValueError`` otherwise. Defaults to the CPU
            count available to the process when ``None``. Sizing rule of
            thumb against provider rate limits: sustainable concurrency ≈
            (requests/min limit) / 60 × mean request latency in seconds —
            per-deployment Versa limits are in the lab wiki's rate-limit
            table. Raising this raises the real request rate, so pair large
            values with ``num_retries`` (see below) and/or ``sleep_interval``.
        sleep_interval : float, optional
            Seconds to sleep after each completed request (within a worker).
            Useful for rate-limiting. No sleep is applied when ``None``.
        model : str, optional
            Model identifier forwarded to each ``run()`` call.
        return_exceptions : bool, optional
            If True, a failed request does not abort the batch: its exception
            is returned in place at the corresponding index, preserving the
            order of ``messages_list``. If False (default), the first exception
            propagates and cancels the batch.
        strict_response_format : bool, optional
            Forwarded to each ``run()`` call; see ``run()``.
        **kwargs : CompletionKwargs
            Additional keyword arguments forwarded to every ``run()`` call.
            In particular ``num_retries=N`` is forwarded to litellm, which
            retries rate-limit (429) and transient connection errors with
            exponential backoff; lab_llm itself performs no retries, so at
            high ``max_parallel_jobs`` a 429 without ``num_retries`` either
            aborts the batch or (with ``return_exceptions=True``) surfaces as
            an exception in the results.

        Returns
        -------
        list of Any
            Results in the same order as ``messages_list``. Each element is the
            return value of the corresponding ``run()`` call (a string or a
            parsed Pydantic model instance), or the raised exception when
            ``return_exceptions`` is True.
        """
        if max_parallel_jobs is None:
            max_parallel_jobs = _default_max_parallel_jobs()
        if not 1 <= max_parallel_jobs <= 512:
            raise ValueError(
                f"max_parallel_jobs={max_parallel_jobs} is outside [1, 512]; "
                "each unit is one worker thread, so larger values are almost "
                "certainly a typo or the wrong knob for rate limiting"
            )

        # Dedicated executor sized to the requested parallelism: the asyncio
        # default executor is capped at min(32, cpu_count + 4) threads, which
        # would silently limit true concurrency below max_parallel_jobs.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_parallel_jobs, thread_name_prefix="lab_llm_batch"
        )
        # The semaphore bounds in-flight submissions so sleep_interval throttles
        # dispatch and a failed batch leaves few queued futures to cancel.
        semaphore = asyncio.Semaphore(max_parallel_jobs)
        results = [None] * len(messages_list)

        num_prompts = len(messages_list)

        async def process_item(index: int, messages: Union[str, List[Union[str, dict, Message]]]):
            async with semaphore:
                logger.info(f"Processing prompt {index} of {num_prompts}")
                try:
                    results[index] = await self._run_single_async(
                        messages, tools, max_tool_calls, model=model,
                        strict_response_format=strict_response_format,
                        executor=executor, **kwargs
                    )
                except Exception as e:
                    if not return_exceptions:
                        raise
                    results[index] = e
                logger.info(f"Received response for prompt {index} of {num_prompts}")
                if sleep_interval is not None:
                    await asyncio.sleep(sleep_interval)

        tasks = [
            process_item(i, messages)
            for i, messages in enumerate(messages_list)
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            # On success all work is done and this is instant; on a propagated
            # exception it cancels queued-but-unstarted calls while already
            # running ones finish in background threads and are discarded.
            executor.shutdown(wait=False, cancel_futures=True)
        return results