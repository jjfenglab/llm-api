"""
Tests that a local OpenAI-compatible endpoint can be reached via the standard
LLMApi + wrap_completion_function flow. We mock litellm.completion and verify
that api_base, api_key, and the openai/-prefixed model name are forwarded.
"""

from unittest.mock import Mock

from litellm import ModelResponse

from lab_llm import LLMApi, wrap_completion_function


def _mock_completion_response(content: str = "hi") -> Mock:
    response = Mock(spec=ModelResponse)
    message = Mock()
    message.content = content
    message.tool_calls = None
    message.model_dump = lambda: {"role": "assistant", "content": content}
    choice = Mock()
    choice.message = message
    response.choices = [choice]
    response.usage = None
    return response


def test_default_parameters_forwards_local_endpoint_config():
    """api_base/api_key set via wrap_completion_function reach the completion call."""
    mock_completion = Mock(return_value=_mock_completion_response())

    api = LLMApi(
        wrap_completion_function(
            mock_completion,
            api_base="http://localhost:8000/v1",
            api_key="not-needed",
        ),
        track_usage=False,
    )

    result = api.run("hello", model="openai/Qwen/Qwen3-32B")

    assert result == "hi"
    mock_completion.assert_called_once()
    _, kwargs = mock_completion.call_args
    assert mock_completion.call_args.args[0] == "openai/Qwen/Qwen3-32B"
    assert kwargs["api_base"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "not-needed"


def test_per_call_overrides_default_endpoint():
    """A per-call api_base overrides the default set at wrap time."""
    mock_completion = Mock(return_value=_mock_completion_response())

    api = LLMApi(
        wrap_completion_function(
            mock_completion,
            api_base="http://localhost:8000/v1",
            api_key="not-needed",
        ),
        track_usage=False,
    )

    api.run("hello", model="openai/some-model", api_base="http://otherhost:9000/v1")

    _, kwargs = mock_completion.call_args
    assert kwargs["api_base"] == "http://otherhost:9000/v1"
