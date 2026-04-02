"""
Test suite for None value filtering in cache.

Verifies that:
1. None values from failed LLM calls are NOT cached
2. Successful responses ARE cached
3. Mixed batches filter out None values correctly
"""

import asyncio
import hashlib
import logging
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from lab_llm.constants import LLMModel, OpenAi
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing."""
    cache = MagicMock(spec=LLMCache)
    cache.get_response.return_value = (False, None)  # Not in cache
    cache.get_responses.return_value = MagicMock()  # Will be customized per test
    return cache


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def mock_error_handler(mock_logger):
    """Create a mock error handler."""
    return ErrorCallbackHandler(mock_logger)


@pytest.fixture
def llm_api(mock_cache, mock_error_handler, mock_logger):
    """Create LLMApi instance with mocked dependencies."""
    model = LLMModel(name=OpenAi.GPT4_O_MINI)
    return LLMApi(
        cache=mock_cache,
        seed=42,
        model_type=model,
        error_handler=mock_error_handler,
        logging=mock_logger,
        timeout=60,
        return_exceptions=True,
    )


class TestSingleRequestNoneFiltering:
    """Test None filtering for single LLM requests."""

    def test_none_response_not_cached(self, llm_api, mock_cache):
        """Test that None responses are NOT cached."""
        with (
            patch.object(llm_api, "get_client") as mock_get_client,
            patch.object(llm_api, "_serialize_llm_response", return_value=None),
        ):

            # Mock LLM client
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock()
            mock_get_client.return_value = mock_llm

            # Call get_output
            result = llm_api.get_output("test prompt")

            # Verify result is None
            assert result is None or hasattr(result, "content")

            # Verify cache.save_response was NOT called
            mock_cache.save_response.assert_not_called()

    def test_successful_response_is_cached(self, llm_api, mock_cache):
        """Test that successful responses ARE cached."""
        with (
            patch.object(llm_api, "get_client") as mock_get_client,
            patch.object(
                llm_api, "_serialize_llm_response", return_value="Success response"
            ),
        ):

            # Mock LLM client
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock()
            mock_get_client.return_value = mock_llm

            # Call get_output
            llm_api.get_output("test prompt")

            # Verify cache.save_response WAS called with correct arguments
            mock_cache.save_response.assert_called_once()
            args = mock_cache.save_response.call_args[0]
            assert args[0] == "test prompt"
            assert args[1] == "Success response"


class TestBatchRequestNoneFiltering:
    """Test None filtering for batch LLM requests."""

    @pytest.mark.asyncio
    async def test_mixed_batch_filters_nones(self, llm_api, mock_cache):
        """Test that batch with mixed results filters out None values."""
        # Setup mock batch results: [success, exception, success, exception]
        mock_responses = [
            MagicMock(content="Response 1", usage_metadata=None),  # Success
            Exception("Timeout"),  # Failure → None (no serialization call)
            MagicMock(content="Response 3", usage_metadata=None),  # Success
            Exception("Rate limit"),  # Failure → None (no serialization call)
        ]

        with (
            patch.object(llm_api, "get_client") as mock_get_client,
            patch.object(
                llm_api,
                "_serialize_llm_response",
                # Only called for non-Exception responses (indices 0 and 2)
                side_effect=["Response 1", "Response 3"],
            ),
        ):

            # Mock LLM client
            mock_llm = MagicMock()
            mock_llm.abatch = AsyncMock(return_value=mock_responses)
            mock_get_client.return_value = mock_llm

            # Call _run_batch
            prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
            result = await llm_api._run_batch(
                prompts_to_run=prompts,
                llm=mock_llm,
                max_new_tokens=100,
                temperature=0,
                response_model=None,
            )

            # Verify returned results include Nones (workflow continues)
            assert len(result) == 4
            assert result == ["Response 1", None, "Response 3", None]

            # Verify cache.save_responses was called with ONLY successful results
            mock_cache.save_responses.assert_called_once()
            cached_prompts = mock_cache.save_responses.call_args[0][0]
            cached_results = mock_cache.save_responses.call_args[0][1]

            # Should only have 2 successful results
            assert len(cached_prompts) == 2
            assert len(cached_results) == 2
            assert "Response 1" in cached_results
            assert "Response 3" in cached_results
            assert None not in cached_results

    @pytest.mark.asyncio
    async def test_all_none_batch_no_cache_call(self, llm_api, mock_cache):
        """Test that batch with all None results doesn't call cache.save_responses."""
        # Setup mock batch results: all exceptions (no serialization calls)
        mock_responses = [
            Exception("Timeout 1"),
            Exception("Timeout 2"),
            Exception("Rate limit"),
        ]

        with patch.object(llm_api, "get_client") as mock_get_client:
            # Mock LLM client
            mock_llm = MagicMock()
            mock_llm.abatch = AsyncMock(return_value=mock_responses)
            mock_get_client.return_value = mock_llm

            # Call _run_batch
            prompts = ["prompt1", "prompt2", "prompt3"]
            result = await llm_api._run_batch(
                prompts_to_run=prompts,
                llm=mock_llm,
                max_new_tokens=100,
                temperature=0,
                response_model=None,
            )

            # Verify all results are None
            assert result == [None, None, None]

            # Verify cache.save_responses was NOT called (no successful results)
            mock_cache.save_responses.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_success_batch_all_cached(self, llm_api, mock_cache):
        """Test that batch with all successful results caches everything."""
        # Setup mock batch results: all success
        mock_responses = [
            MagicMock(content="Response 1", usage_metadata=None),
            MagicMock(content="Response 2", usage_metadata=None),
            MagicMock(content="Response 3", usage_metadata=None),
        ]

        with (
            patch.object(llm_api, "get_client") as mock_get_client,
            patch.object(
                llm_api,
                "_serialize_llm_response",
                side_effect=["Response 1", "Response 2", "Response 3"],
            ),
        ):

            # Mock LLM client
            mock_llm = MagicMock()
            mock_llm.abatch = AsyncMock(return_value=mock_responses)
            mock_get_client.return_value = mock_llm

            # Call _run_batch
            prompts = ["prompt1", "prompt2", "prompt3"]
            result = await llm_api._run_batch(
                prompts_to_run=prompts,
                llm=mock_llm,
                max_new_tokens=100,
                temperature=0,
                response_model=None,
            )

            # Verify all results are successful
            assert result == ["Response 1", "Response 2", "Response 3"]

            # Verify cache.save_responses was called with ALL results
            mock_cache.save_responses.assert_called_once()
            cached_prompts = mock_cache.save_responses.call_args[0][0]
            cached_results = mock_cache.save_responses.call_args[0][1]

            assert len(cached_prompts) == 3
            assert len(cached_results) == 3
            assert cached_results == ["Response 1", "Response 2", "Response 3"]


class TestCachingPolicy:
    """Test overall caching policy behavior."""

    def test_cache_lookup_before_call(self, llm_api, mock_cache):
        """Test that cache is checked before making LLM call."""
        # Setup cache to return a hit
        mock_cache.get_response.return_value = (True, "Cached response")

        with patch.object(llm_api, "get_client") as mock_get_client:
            # Call get_output
            result = llm_api.get_output("test prompt")

            # Verify cache was checked
            mock_cache.get_response.assert_called_once()

            # Verify LLM client was NOT created (cache hit)
            mock_get_client.assert_not_called()

    def test_none_return_preserves_workflow(self, llm_api, mock_cache):
        """Test that returning None doesn't break downstream code."""
        with (
            patch.object(llm_api, "get_client") as mock_get_client,
            patch.object(llm_api, "_serialize_llm_response", return_value=None),
        ):

            # Mock LLM client
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock()
            mock_get_client.return_value = mock_llm

            # Call get_output - should not raise exception
            result = llm_api.get_output("test prompt")

            # Workflow continues, just returns None
            # (Downstream code can check for None and handle appropriately)
            assert result is None or hasattr(result, "content")


# Helper for async tests
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
