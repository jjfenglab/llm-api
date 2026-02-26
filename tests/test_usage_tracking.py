"""
Tests for token usage tracking feature.

Verifies that:
1. UsageCallbackHandler captures token usage from LLM responses
2. track_usage=True prints and logs usage for single and batch calls
3. track_usage=False produces no usage output
4. Cache hits report zero tokens
5. Verbosity levels produce correct output formats
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from lab_llm.constants import LLMModel, OpenAi
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache
from lab_llm.usage_callback_handler import UsageCallbackHandler


# --- UsageCallbackHandler unit tests ---


class TestUsageCallbackHandler:
    """Test the callback handler in isolation."""

    def test_initial_state_is_none(self):
        handler = UsageCallbackHandler()
        assert handler.last_usage is None

    def test_reset_clears_usage(self):
        handler = UsageCallbackHandler()
        handler._usage_data = {"input_tokens": 10}
        handler.reset()
        assert handler.last_usage is None

    def test_captures_from_llm_output_token_usage(self):
        """Test extraction from OpenAI/Azure llm_output format."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        mock_response.generations = []

        handler.on_llm_end(mock_response)

        assert handler.last_usage is not None
        assert handler.last_usage["input_tokens"] == 100
        assert handler.last_usage["output_tokens"] == 50
        assert handler.last_usage["total_tokens"] == 150

    def test_source1_captures_cached_and_reasoning(self):
        """Test Source 1 (llm_output) extracts detailed token breakdowns."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
                "prompt_tokens_details": {"cached_tokens": 300},
                "completion_tokens_details": {"reasoning_tokens": 80},
            }
        }
        mock_response.generations = []

        handler.on_llm_end(mock_response)

        assert handler.last_usage["cached_tokens"] == 300
        assert handler.last_usage["reasoning_tokens"] == 80

    def test_source1_falls_through_when_no_token_usage(self):
        """Test that llm_output without token_usage falls through to Source 2."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = {"model_name": "gpt-4o-mini"}

        mock_message = MagicMock()
        mock_message.usage_metadata = {
            "input_tokens": 42,
            "output_tokens": 10,
        }
        mock_gen = MagicMock()
        mock_gen.message = mock_message
        mock_response.generations = [[mock_gen]]

        handler.on_llm_end(mock_response)

        assert handler.last_usage is not None
        assert handler.last_usage["input_tokens"] == 42
        assert handler.last_usage["output_tokens"] == 10

    def test_captures_from_generation_usage_metadata(self):
        """Test extraction from LangChain normalized usage_metadata."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = None

        mock_message = MagicMock()
        mock_message.usage_metadata = {
            "input_tokens": 200,
            "output_tokens": 80,
        }
        mock_gen = MagicMock()
        mock_gen.message = mock_message
        mock_response.generations = [[mock_gen]]

        handler.on_llm_end(mock_response)

        assert handler.last_usage is not None
        assert handler.last_usage["input_tokens"] == 200
        assert handler.last_usage["output_tokens"] == 80
        assert handler.last_usage["total_tokens"] == 280

    def test_captures_detailed_breakdowns(self):
        """Test extraction of cached and reasoning token details."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = None

        mock_message = MagicMock()
        mock_message.usage_metadata = {
            "input_tokens": 300,
            "output_tokens": 100,
            "input_token_details": {"cache_read": 200},
            "output_token_details": {"reasoning": 40},
        }
        mock_gen = MagicMock()
        mock_gen.message = mock_message
        mock_response.generations = [[mock_gen]]

        handler.on_llm_end(mock_response)

        assert handler.last_usage["cached_tokens"] == 200
        assert handler.last_usage["reasoning_tokens"] == 40

    def test_no_usage_metadata_leaves_none(self):
        """Test that empty responses don't set usage data."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = None
        mock_response.generations = []

        handler.on_llm_end(mock_response)

        assert handler.last_usage is None

    def test_llm_output_preferred_over_generation(self):
        """Test that llm_output is checked first (short-circuits)."""
        handler = UsageCallbackHandler()
        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        }
        # Even with generation data present, llm_output should be used
        mock_message = MagicMock()
        mock_message.usage_metadata = {"input_tokens": 999, "output_tokens": 999}
        mock_gen = MagicMock()
        mock_gen.message = mock_message
        mock_response.generations = [[mock_gen]]

        handler.on_llm_end(mock_response)

        assert handler.last_usage["input_tokens"] == 10
        assert handler.last_usage["output_tokens"] == 5


class TestParseUsageMetadata:
    """Test the shared parse_usage_metadata static method."""

    def test_returns_none_for_empty_input(self):
        assert UsageCallbackHandler.parse_usage_metadata(None) is None
        assert UsageCallbackHandler.parse_usage_metadata({}) is None

    def test_parses_basic_tokens(self):
        result = UsageCallbackHandler.parse_usage_metadata({
            "input_tokens": 100,
            "output_tokens": 50,
        })
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_parses_detailed_breakdowns(self):
        result = UsageCallbackHandler.parse_usage_metadata({
            "input_tokens": 300,
            "output_tokens": 100,
            "input_token_details": {"cache_read": 200},
            "output_token_details": {"reasoning": 40},
        })
        assert result["cached_tokens"] == 200
        assert result["reasoning_tokens"] == 40

    def test_omits_zero_details(self):
        result = UsageCallbackHandler.parse_usage_metadata({
            "input_tokens": 100,
            "output_tokens": 50,
        })
        assert "cached_tokens" not in result
        assert "reasoning_tokens" not in result


# --- Fixtures ---


@pytest.fixture
def mock_cache():
    cache = MagicMock(spec=LLMCache)
    cache.get_response.return_value = (False, None)
    cache.get_responses.return_value = MagicMock()
    return cache


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_error_handler(mock_logger):
    return ErrorCallbackHandler(mock_logger)


def _make_api(mock_cache, mock_error_handler, mock_logger, track_usage=True):
    model = LLMModel(name=OpenAi.GPT4_O_MINI)
    return LLMApi(
        cache=mock_cache,
        seed=42,
        model_type=model,
        error_handler=mock_error_handler,
        logging=mock_logger,
        timeout=60,
        return_exceptions=True,
        track_usage=track_usage,
    )


@pytest.fixture
def api_with_tracking(mock_cache, mock_error_handler, mock_logger):
    return _make_api(mock_cache, mock_error_handler, mock_logger, track_usage=True)


@pytest.fixture
def api_without_tracking(mock_cache, mock_error_handler, mock_logger):
    return _make_api(mock_cache, mock_error_handler, mock_logger, track_usage=False)



# --- LLMApi integration tests (mocked) ---


class TestTrackUsageInit:
    """Test that track_usage parameter wires up correctly."""

    def test_track_usage_creates_handler(self, api_with_tracking):
        assert api_with_tracking.track_usage is True
        assert api_with_tracking.usage_handler is not None
        assert isinstance(api_with_tracking.usage_handler, UsageCallbackHandler)

    def test_no_tracking_has_no_handler(self, api_without_tracking):
        assert api_without_tracking.track_usage is False
        assert api_without_tracking.usage_handler is None

    def test_callbacks_include_usage_handler(self, api_with_tracking):
        callbacks = api_with_tracking._get_callbacks()
        assert len(callbacks) == 2
        assert isinstance(callbacks[1], UsageCallbackHandler)

    def test_callbacks_without_tracking(self, api_without_tracking):
        callbacks = api_without_tracking._get_callbacks()
        assert len(callbacks) == 1


class TestSingleCallUsageTracking:
    """Test usage reporting for get_output()."""

    def test_prints_usage_on_api_call(self, api_with_tracking, mock_cache, capsys):
        """When track_usage=True and API call succeeds, usage is printed."""
        # Simulate the usage handler having captured data
        api_with_tracking.usage_handler._usage_data = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        with (
            patch.object(api_with_tracking, "get_client") as mock_get_client,
            patch.object(api_with_tracking, "_serialize_llm_response", return_value="response"),
            # Patch usage_handler.reset so our test data isn't cleared
            patch.object(api_with_tracking.usage_handler, "reset"),
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock()
            mock_get_client.return_value = mock_llm

            api_with_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "Token usage: 100 input / 50 output" in captured.out

    def test_end_to_end_callback_flow(self, api_with_tracking, mock_cache, capsys):
        """Full flow: reset -> invoke -> on_llm_end -> report without bypassing anything."""
        # Build a mock LLM whose invoke triggers the real callback handler
        handler = api_with_tracking.usage_handler

        with (
            patch.object(api_with_tracking, "get_client") as mock_get_client,
            patch.object(api_with_tracking, "_serialize_llm_response", return_value="response"),
        ):
            mock_llm = MagicMock()

            def fake_invoke(messages):
                # Simulate LangChain calling the callback after the LLM call
                mock_result = MagicMock()
                mock_result.llm_output = {
                    "token_usage": {
                        "prompt_tokens": 75,
                        "completion_tokens": 25,
                        "total_tokens": 100,
                    }
                }
                mock_result.generations = []
                handler.on_llm_end(mock_result)
                return MagicMock(content="response")

            mock_llm.invoke.side_effect = fake_invoke
            mock_get_client.return_value = mock_llm

            api_with_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "Token usage: 75 input / 25 output" in captured.out

    def test_no_usage_printed_on_invoke_failure(self, api_with_tracking, mock_cache, capsys):
        """When llm.invoke() raises, no usage is printed (on_llm_end never fires)."""
        with (
            patch.object(api_with_tracking, "get_client") as mock_get_client,
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("Connection timeout")
            mock_get_client.return_value = mock_llm

            with pytest.raises(Exception, match="Connection timeout"):
                api_with_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "Token usage" not in captured.out

    def test_prints_cached_on_cache_hit(self, api_with_tracking, mock_cache, capsys):
        """When response comes from cache, prints (cached) indicator."""
        mock_cache.get_response.return_value = (True, "Cached response")

        api_with_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "Token usage: 0 input / 0 output (cached)" in captured.out

    def test_no_print_when_tracking_disabled(self, api_without_tracking, mock_cache, capsys):
        """When track_usage=False, nothing is printed."""
        mock_cache.get_response.return_value = (True, "Cached response")

        api_without_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "Token usage" not in captured.out

    def test_shows_details_when_present(self, api_with_tracking, mock_cache, capsys):
        """Cached and reasoning token breakdowns shown when present."""
        api_with_tracking.usage_handler._usage_data = {
            "input_tokens": 300,
            "output_tokens": 100,
            "total_tokens": 400,
            "cached_tokens": 200,
            "reasoning_tokens": 40,
        }

        with (
            patch.object(api_with_tracking, "get_client") as mock_get_client,
            patch.object(api_with_tracking, "_serialize_llm_response", return_value="response"),
            patch.object(api_with_tracking.usage_handler, "reset"),
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock()
            mock_get_client.return_value = mock_llm

            api_with_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "300 input (200 cached)" in captured.out
        assert "100 output (40 reasoning)" in captured.out

    def test_logs_usage(self, api_with_tracking, mock_cache, mock_logger):
        """Usage is also logged via self.logging."""
        mock_cache.get_response.return_value = (True, "Cached response")

        api_with_tracking.get_output("test prompt")

        # Find the usage-related log call
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("Token usage: 0 input / 0 output (cached)" in c for c in info_calls)

    def test_comma_formatting_large_numbers(self, api_with_tracking, mock_cache, capsys):
        """Token counts should be comma-formatted for readability."""
        api_with_tracking.usage_handler._usage_data = {
            "input_tokens": 12345,
            "output_tokens": 6789,
            "total_tokens": 19134,
        }

        with (
            patch.object(api_with_tracking, "get_client") as mock_get_client,
            patch.object(api_with_tracking, "_serialize_llm_response", return_value="response"),
            patch.object(api_with_tracking.usage_handler, "reset"),
        ):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock()
            mock_get_client.return_value = mock_llm

            api_with_tracking.get_output("test prompt")

        captured = capsys.readouterr()
        assert "12,345 input" in captured.out
        assert "6,789 output" in captured.out


class TestBatchUsageTracking:
    """Test usage reporting for _run_batch()."""

    @pytest.mark.asyncio
    async def test_prints_batch_usage(self, api_with_tracking, mock_cache, capsys):
        """Batch calls print aggregate usage."""
        mock_responses = [
            MagicMock(
                content="Response 1",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_token_details": None,
                    "output_token_details": None,
                },
            ),
            MagicMock(
                content="Response 2",
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 60,
                    "input_token_details": None,
                    "output_token_details": None,
                },
            ),
        ]

        with patch.object(
            api_with_tracking, "_serialize_llm_response", side_effect=["Response 1", "Response 2"]
        ):
            mock_llm = MagicMock()
            mock_llm.abatch = AsyncMock(return_value=mock_responses)

            await api_with_tracking._run_batch(
                prompts_to_run=["p1", "p2"],
                llm=mock_llm,
                max_new_tokens=100,
                temperature=0,
            )

        captured = capsys.readouterr()
        assert "Batch token usage:" in captured.out
        assert "220 input" in captured.out
        assert "110 output" in captured.out
        assert "2 queries" in captured.out

    @pytest.mark.asyncio
    async def test_no_batch_print_when_tracking_disabled(
        self, api_without_tracking, mock_cache, capsys
    ):
        """Batch calls with track_usage=False produce no print output."""
        mock_responses = [
            MagicMock(
                content="Response",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "input_token_details": None,
                    "output_token_details": None,
                },
            ),
        ]

        with patch.object(
            api_without_tracking, "_serialize_llm_response", return_value="Response"
        ):
            mock_llm = MagicMock()
            mock_llm.abatch = AsyncMock(return_value=mock_responses)

            await api_without_tracking._run_batch(
                prompts_to_run=["p1"],
                llm=mock_llm,
                max_new_tokens=100,
                temperature=0,
            )

        captured = capsys.readouterr()
        assert "Batch token usage:" not in captured.out

    @pytest.mark.asyncio
    async def test_batch_shows_details_when_present(self, api_with_tracking, mock_cache, capsys):
        """Batch shows cached and reasoning breakdown when present."""
        mock_responses = [
            MagicMock(
                content="Response",
                usage_metadata={
                    "input_tokens": 500,
                    "output_tokens": 200,
                    "input_token_details": {"cache_read": 300},
                    "output_token_details": {"reasoning": 80},
                },
            ),
        ]

        with patch.object(
            api_with_tracking, "_serialize_llm_response", return_value="Response"
        ):
            mock_llm = MagicMock()
            mock_llm.abatch = AsyncMock(return_value=mock_responses)

            await api_with_tracking._run_batch(
                prompts_to_run=["p1"],
                llm=mock_llm,
                max_new_tokens=100,
                temperature=0,
            )

        captured = capsys.readouterr()
        assert "300 cached" in captured.out
        assert "80 reasoning" in captured.out


# Helper for async tests
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
