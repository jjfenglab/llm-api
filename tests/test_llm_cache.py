"""
Tests for LLMCache functionality.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from lab_llm.constants import LLMModel
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.llm_cache import LLMCache


class TestLLMCacheBasicOperations:
    """Test basic cache operations."""

    def test_cache_initialization(self, cache):
        """Test that cache initializes properly."""
        assert cache is not None
        assert hasattr(cache, "db_handler")
        assert hasattr(cache, "logger")

    def test_empty_cache_get_response(self, cache, openai_model):
        """Test getting response from empty cache."""
        found, response = cache.get_response("test prompt", openai_model, 42, 100, 0.0)
        assert not found
        assert response is None

    def test_empty_cache_get_responses(self, cache, openai_model):
        """Test getting responses from empty cache."""
        texts = ["prompt1", "prompt2", "prompt3"]
        df = cache.get_responses(texts, openai_model, 42, 100, 0.0)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["prompt", "llm_output", "found_in_cache"]
        assert df["prompt"].tolist() == texts
        assert df["llm_output"].isna().all()
        assert not df["found_in_cache"].any()

    def test_save_and_retrieve_single_response(self, cache, openai_model):
        """Test saving and retrieving a single response."""
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."

        # Save response
        cache.save_response(prompt, response, openai_model, 42, 100, 0.0)

        # Retrieve response
        found, retrieved_response = cache.get_response(
            prompt, openai_model, 42, 100, 0.0
        )

        assert found
        assert retrieved_response == response

    def test_save_and_retrieve_multiple_responses(self, cache, openai_model):
        """Test saving and retrieving multiple responses."""
        prompts = ["prompt1", "prompt2", "prompt3"]
        responses = ["response1", "response2", "response3"]

        # Save responses
        cache.save_responses(prompts, responses, openai_model, 42, 100, 0.0)

        # Retrieve responses
        df = cache.get_responses(prompts, openai_model, 42, 100, 0.0)

        assert len(df) == 3
        assert df["found_in_cache"].all()
        assert df["llm_output"].tolist() == responses


class TestLLMCacheParameterSensitivity:
    """Test that cache is sensitive to different parameters."""

    def test_different_models(self, cache, openai_model, anthropic_model):
        """Test that different models are cached separately."""
        prompt = "test prompt"
        response1 = "response from openai"
        response2 = "response from anthropic"

        # Save responses for different models
        cache.save_response(prompt, response1, openai_model, 42, 100, 0.0)
        cache.save_response(prompt, response2, anthropic_model, 42, 100, 0.0)

        # Retrieve responses
        found1, retrieved1 = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        found2, retrieved2 = cache.get_response(prompt, anthropic_model, 42, 100, 0.0)

        assert found1 and found2
        assert retrieved1 == response1
        assert retrieved2 == response2

    def test_different_seeds(self, cache, openai_model):
        """Test that different seeds are cached separately."""
        prompt = "test prompt"
        response1 = "response with seed 42"
        response2 = "response with seed 123"

        # Save responses with different seeds
        cache.save_response(prompt, response1, openai_model, 42, 100, 0.0)
        cache.save_response(prompt, response2, openai_model, 123, 100, 0.0)

        # Retrieve responses
        found1, retrieved1 = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        found2, retrieved2 = cache.get_response(prompt, openai_model, 123, 100, 0.0)

        assert found1 and found2
        assert retrieved1 == response1
        assert retrieved2 == response2

    def test_different_temperatures(self, cache, openai_model):
        """Test that different temperatures are cached separately."""
        prompt = "test prompt"
        response1 = "response with temp 0.0"
        response2 = "response with temp 0.7"

        # Save responses with different temperatures
        cache.save_response(prompt, response1, openai_model, 42, 100, 0.0)
        cache.save_response(prompt, response2, openai_model, 42, 100, 0.7)

        # Retrieve responses
        found1, retrieved1 = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        found2, retrieved2 = cache.get_response(prompt, openai_model, 42, 100, 0.7)

        assert found1 and found2
        assert retrieved1 == response1
        assert retrieved2 == response2

    def test_different_max_tokens(self, cache, openai_model):
        """Test that different max_tokens are cached separately."""
        prompt = "test prompt"
        response1 = "response with 100 tokens"
        response2 = "response with 200 tokens"

        # Save responses with different max_tokens
        cache.save_response(prompt, response1, openai_model, 42, 100, 0.0)
        cache.save_response(prompt, response2, openai_model, 42, 200, 0.0)

        # Retrieve responses
        found1, retrieved1 = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        found2, retrieved2 = cache.get_response(prompt, openai_model, 42, 200, 0.0)

        assert found1 and found2
        assert retrieved1 == response1
        assert retrieved2 == response2


class TestLLMCacheConsistency:
    """Test consistency between single and batch operations."""

    def test_get_response_vs_get_responses_consistency(self, cache, openai_model):
        """Test that get_response and get_responses return consistent results."""
        prompt = "consistency test prompt"
        response = "consistency test response"

        # Save response
        cache.save_response(prompt, response, openai_model, 42, 100, 0.0)

        # Get single response
        found_single, response_single = cache.get_response(
            prompt, openai_model, 42, 100, 0.0
        )

        # Get batch response
        df = cache.get_responses([prompt], openai_model, 42, 100, 0.0)
        found_batch = df.iloc[0]["found_in_cache"]
        response_batch = df.iloc[0]["llm_output"]

        # Should be consistent
        assert found_single == found_batch
        assert response_single == response_batch

    def test_partial_cache_hits(self, cache, openai_model):
        """Test behavior when only some prompts are in cache."""
        prompts = ["cached_prompt", "uncached_prompt1", "uncached_prompt2"]
        cached_response = "this is cached"

        # Save only first prompt
        cache.save_response(prompts[0], cached_response, openai_model, 42, 100, 0.0)

        # Get all prompts
        df = cache.get_responses(prompts, openai_model, 42, 100, 0.0)

        assert len(df) == 3
        assert df.iloc[0]["found_in_cache"]
        assert df.iloc[0]["llm_output"] == cached_response
        assert not df.iloc[1]["found_in_cache"]
        assert pd.isna(df.iloc[1]["llm_output"])
        assert not df.iloc[2]["found_in_cache"]
        assert pd.isna(df.iloc[2]["llm_output"])


class TestLLMCacheEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self, cache, openai_model):
        """Test handling of empty prompts."""
        empty_prompt = ""
        response = "response to empty prompt"

        # Should be able to save and retrieve empty prompt
        cache.save_response(empty_prompt, response, openai_model, 42, 100, 0.0)
        found, retrieved = cache.get_response(empty_prompt, openai_model, 42, 100, 0.0)

        assert found
        assert retrieved == response

    def test_whitespace_normalization(self, cache, openai_model):
        """Test that prompts with different whitespace are treated as same."""
        prompt1 = "test prompt"
        prompt2 = "  test prompt  "  # with leading/trailing whitespace
        response = "test response"

        # Save with first prompt
        cache.save_response(prompt1, response, openai_model, 42, 100, 0.0)

        # Retrieve with second prompt (should find due to normalization)
        found, retrieved = cache.get_response(prompt2, openai_model, 42, 100, 0.0)

        assert found
        assert retrieved == response

    def test_none_seed_handling(self, cache, openai_model):
        """Test handling of None seed values."""
        prompt = "test prompt with none seed"
        response = "test response"

        cache.save_response(prompt, response, openai_model, None, 100, 0.0)
        found, retrieved = cache.get_response(prompt, openai_model, None, 100, 0.0)

        assert found
        assert retrieved == response

    def test_empty_batch_operations(self, cache, openai_model):
        """Test batch operations with empty lists."""
        # Empty get_responses
        df = cache.get_responses([], openai_model, 42, 100, 0.0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

        # Empty save_responses
        cache.save_responses(
            [], [], openai_model, 42, 100, 0.0
        )  # Should not raise error


class TestLLMCacheErrorHandling:
    """Test error handling and resilience."""

    def test_database_error_handling(self, temp_db_path, openai_model):
        """Test that cache handles database errors gracefully."""
        # Create a working cache first
        handler = DuckDBHandler(temp_db_path, read_only=False, max_retries=3)
        cache = LLMCache(handler)
        handler.close_connection()

        # Replace handler with mock that fails
        mock_handler = Mock()
        mock_handler.execute_with_retry.side_effect = Exception("Database error")
        cache.db_handler = mock_handler

        # Should not raise exception, should return False/None
        found, response = cache.get_response("test", openai_model, 42, 100, 0.0)
        assert not found
        assert response is None

    @patch("lab_llm.llm_cache.logging.getLogger")
    def test_error_logging(self, mock_logger, temp_db_path, openai_model):
        """Test that errors are properly logged."""
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        # Create a working cache first
        handler = DuckDBHandler(temp_db_path, read_only=False, max_retries=3)
        cache = LLMCache(handler)
        handler.close_connection()

        # Replace handler with mock that raises errors
        mock_handler = Mock()
        mock_handler.execute_with_retry.side_effect = Exception("Test error")
        cache.db_handler = mock_handler

        # Should log error when operation fails
        cache.get_response("test", openai_model, 42, 100, 0.0)
        mock_logger_instance.error.assert_called()


class TestLLMCacheHashing:
    """Test hash computation functionality."""

    def test_compute_hash_consistency(self, cache):
        """Test that hash computation is consistent."""
        text = "test text for hashing"
        hash1 = cache.compute_hash(text)
        hash2 = cache.compute_hash(text)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

    def test_different_texts_different_hashes(self, cache):
        """Test that different texts produce different hashes."""
        text1 = "first text"
        text2 = "second text"

        hash1 = cache.compute_hash(text1)
        hash2 = cache.compute_hash(text2)

        assert hash1 != hash2

    def test_whitespace_normalization_in_hash(self, cache):
        """Test that whitespace is normalized in hash computation."""
        text1 = "test text"
        text2 = "  test text  "

        hash1 = cache.compute_hash(text1)
        hash2 = cache.compute_hash(text2)

        assert hash1 == hash2  # Should be same due to strip()


class TestLLMCacheTemperatureNormalization:
    """Test temperature normalization between int and float values."""

    def test_temperature_normalization_single_operations(self, cache, openai_model):
        """Test that int and float temperatures work interchangeably for single operations."""
        prompt = "test temperature normalization"
        response = "normalized response"

        # Save with float temperature
        cache.save_response(prompt, response, openai_model, 42, 100, 0.0)

        # Retrieve with int temperature
        found_int, response_int = cache.get_response(prompt, openai_model, 42, 100, 0)
        assert found_int
        assert response_int == response

        # Retrieve with float temperature
        found_float, response_float = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        assert found_float
        assert response_float == response

    def test_temperature_normalization_reverse_order(self, cache, openai_model):
        """Test saving with int and retrieving with float."""
        prompt = "test reverse normalization"
        response = "reverse response"

        # Save with int temperature
        cache.save_response(prompt, response, openai_model, 42, 100, 0)

        # Retrieve with float temperature
        found_float, response_float = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        assert found_float
        assert response_float == response

        # Retrieve with int temperature
        found_int, response_int = cache.get_response(prompt, openai_model, 42, 100, 0)
        assert found_int
        assert response_int == response

    def test_temperature_normalization_batch_operations(self, cache, openai_model):
        """Test that temperature normalization works for batch operations."""
        prompts = ["prompt1", "prompt2"]
        responses = ["response1", "response2"]

        # Save responses with float temperature
        cache.save_responses(prompts, responses, openai_model, 42, 100, 0.0)

        # Retrieve with int temperature
        df_int = cache.get_responses(prompts, openai_model, 42, 100, 0)
        assert df_int["found_in_cache"].all()
        assert df_int["llm_output"].tolist() == responses

        # Retrieve with float temperature
        df_float = cache.get_responses(prompts, openai_model, 42, 100, 0.0)
        assert df_float["found_in_cache"].all()
        assert df_float["llm_output"].tolist() == responses

    def test_temperature_normalization_mixed_operations(self, cache, openai_model):
        """Test mixing save/retrieve operations with different temperature types."""
        # Save one with int, one with float
        cache.save_response("int_prompt", "int_response", openai_model, 42, 100, 0)
        cache.save_response("float_prompt", "float_response", openai_model, 42, 100, 0.0)

        # Retrieve both with int temperature
        df_int = cache.get_responses(["int_prompt", "float_prompt"], openai_model, 42, 100, 0)
        assert df_int["found_in_cache"].all()
        assert df_int.iloc[0]["llm_output"] == "int_response"
        assert df_int.iloc[1]["llm_output"] == "float_response"

        # Retrieve both with float temperature
        df_float = cache.get_responses(["int_prompt", "float_prompt"], openai_model, 42, 100, 0.0)
        assert df_float["found_in_cache"].all()
        assert df_float.iloc[0]["llm_output"] == "int_response"
        assert df_float.iloc[1]["llm_output"] == "float_response"

    def test_temperature_normalization_different_values(self, cache, openai_model):
        """Test that different temperature values are still cached separately."""
        prompt = "different temps"

        # Save with different temperatures
        cache.save_response(prompt, "temp_0", openai_model, 42, 100, 0.0)
        cache.save_response(prompt, "temp_0_5", openai_model, 42, 100, 0.5)
        cache.save_response(prompt, "temp_1", openai_model, 42, 100, 1.0)

        # Retrieve each (testing with both int and float where applicable)
        found_0_int, response_0_int = cache.get_response(prompt, openai_model, 42, 100, 0)
        found_0_float, response_0_float = cache.get_response(prompt, openai_model, 42, 100, 0.0)
        found_05, response_05 = cache.get_response(prompt, openai_model, 42, 100, 0.5)
        found_1_int, response_1_int = cache.get_response(prompt, openai_model, 42, 100, 1)
        found_1_float, response_1_float = cache.get_response(prompt, openai_model, 42, 100, 1.0)

        # All should be found
        assert all([found_0_int, found_0_float, found_05, found_1_int, found_1_float])

        # Responses should match saved values
        assert response_0_int == "temp_0"
        assert response_0_float == "temp_0"
        assert response_05 == "temp_0_5"
        assert response_1_int == "temp_1"
        assert response_1_float == "temp_1"
