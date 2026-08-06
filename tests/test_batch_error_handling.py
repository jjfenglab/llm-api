"""
Unit tests for run_batch partial-failure handling and strict structured output.
"""

import asyncio
import pytest
from typing import List

from litellm import ModelResponse
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from litellm.types.utils import Usage
from pydantic import BaseModel

from lab_llm.api import LLMApi


class Answer(BaseModel):
    value: int


def make_response(model: str, content: str) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop"
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )


class TestRunBatchReturnExceptions:
    """run_batch partial-failure behavior."""

    @pytest.fixture
    def failing_on_boom_api(self):
        """API whose completion raises for any prompt containing 'boom'."""
        def completion_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            prompt = messages[-1]["content"]
            if "boom" in prompt:
                raise ValueError(f"failed: {prompt}")
            return make_response(model, f"ok: {prompt}")
        return LLMApi(completion_func)

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_default_propagates_exception(self, failing_on_boom_api, max_parallel_jobs):
        messages_list = [["a"], ["boom"], ["c"]]

        with pytest.raises(ValueError, match="failed: boom"):
            asyncio.run(failing_on_boom_api.run_batch(
                messages_list, max_parallel_jobs=max_parallel_jobs
            ))

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_return_exceptions_preserves_order(self, failing_on_boom_api, max_parallel_jobs):
        messages_list = [["a"], ["boom one"], ["c"], ["boom two"], ["e"]]

        results = asyncio.run(failing_on_boom_api.run_batch(
            messages_list, max_parallel_jobs=max_parallel_jobs, return_exceptions=True
        ))

        assert len(results) == len(messages_list)
        assert results[0] == "ok: a"
        assert results[2] == "ok: c"
        assert results[4] == "ok: e"

        assert isinstance(results[1], ValueError)
        assert str(results[1]) == "failed: boom one"
        assert isinstance(results[3], ValueError)
        assert str(results[3]) == "failed: boom two"

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_return_exceptions_all_failures(self, failing_on_boom_api, max_parallel_jobs):
        results = asyncio.run(failing_on_boom_api.run_batch(
            [["boom 1"], ["boom 2"]], max_parallel_jobs=max_parallel_jobs, return_exceptions=True
        ))

        assert len(results) == 2
        assert all(isinstance(r, ValueError) for r in results)


class TestStrictResponseFormat:
    """Structured output parse-failure behavior on run()."""

    @pytest.fixture
    def bad_json_api(self):
        def completion_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            return make_response(model, "this is not json")
        return LLMApi(completion_func)

    @pytest.fixture
    def good_json_api(self):
        def completion_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            return make_response(model, '{"value": 42}')
        return LLMApi(completion_func)

    def test_valid_json_parses(self, good_json_api):
        result = good_json_api.run("hello", response_format=Answer)
        assert isinstance(result, Answer)
        assert result.value == 42

    def test_non_strict_returns_raw_string(self, bad_json_api):
        result = bad_json_api.run("hello", response_format=Answer)
        assert result == "this is not json"

    def test_strict_raises(self, bad_json_api):
        with pytest.raises(Exception):
            bad_json_api.run("hello", response_format=Answer, strict_response_format=True)

    def test_strict_does_not_affect_valid_json(self, good_json_api):
        result = good_json_api.run("hello", response_format=Answer, strict_response_format=True)
        assert result == Answer(value=42)


class TestFlagsThroughRunBatch:
    """Both flags composed through run_batch."""

    @pytest.fixture
    def mixed_json_api(self):
        """Returns valid JSON unless the prompt contains 'bad'."""
        def completion_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            prompt = messages[-1]["content"]
            content = "not json" if "bad" in prompt else '{"value": 7}'
            return make_response(model, content)
        return LLMApi(completion_func)

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_strict_failures_returned_in_place(self, mixed_json_api, max_parallel_jobs):
        results = asyncio.run(mixed_json_api.run_batch(
            [["good"], ["bad"], ["good"]],
            max_parallel_jobs=max_parallel_jobs,
            return_exceptions=True,
            strict_response_format=True,
            response_format=Answer,
        ))

        assert results[0] == Answer(value=7)
        assert isinstance(results[1], Exception)
        assert results[2] == Answer(value=7)

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_non_strict_falls_back_to_string(self, mixed_json_api, max_parallel_jobs):
        results = asyncio.run(mixed_json_api.run_batch(
            [["good"], ["bad"]],
            max_parallel_jobs=max_parallel_jobs,
            response_format=Answer,
        ))

        assert results[0] == Answer(value=7)
        assert results[1] == "not json"

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_strict_alone_propagates_from_batch(self, mixed_json_api, max_parallel_jobs):
        with pytest.raises(Exception):
            asyncio.run(mixed_json_api.run_batch(
                [["good"], ["bad"]],
                max_parallel_jobs=max_parallel_jobs,
                strict_response_format=True,
                response_format=Answer,
            ))


class TestFlagsNotForwardedToCompletion:
    """Neither flag may leak into the completion function's kwargs."""

    @pytest.fixture
    def recording(self):
        recorded = []

        def completion_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            recorded.append(kwargs)
            return make_response(model, "hi")

        return LLMApi(completion_func), recorded

    def test_run_does_not_forward_strict_flag(self, recording):
        api, recorded = recording
        api.run("hello", strict_response_format=True, temperature=0.1)

        assert len(recorded) == 1
        assert "strict_response_format" not in recorded[0]
        assert recorded[0]["temperature"] == 0.1

    @pytest.mark.parametrize("max_parallel_jobs", [None, 2])
    def test_run_batch_does_not_forward_flags(self, recording, max_parallel_jobs):
        api, recorded = recording
        asyncio.run(api.run_batch(
            [["a"], ["b"]],
            max_parallel_jobs=max_parallel_jobs,
            return_exceptions=True,
            strict_response_format=True,
            temperature=0.1,
        ))

        assert len(recorded) == 2
        for kwargs in recorded:
            assert "return_exceptions" not in kwargs
            assert "strict_response_format" not in kwargs
            assert kwargs["temperature"] == 0.1
