"""
Tests for LLMApi functionality.
"""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel
from torch.utils.data import Dataset

from lab_llm.constants import Anthropic, LLMModel, OpenAi
from lab_llm.llm_api import LLMApi


class SimpleResponse(BaseModel):
    """Simple response model for testing."""

    answer: str
    confidence: float


class MockDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Return (prompt, backup_prompt) tuple as expected by get_outputs
        return self.prompts[idx], f"backup_{self.prompts[idx]}"


class TestLLMApiInitialization:
    """Test LLMApi initialization and basic properties."""

    def test_api_initialization(self, cache, error_handler, openai_model, mock_logger):
        """Test that LLMApi initializes properly."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
            timeout=60,
            return_exceptions=False,
        )

        assert api is not None
        assert api.cache == cache
        assert api.seed == 42
        assert api.model_type == openai_model
        assert api.error_handler == error_handler
        assert api.timeout == 60
        assert not api.return_exceptions
        assert api.is_api

    def test_api_default_parameters(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test that default parameters work correctly."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        assert api.timeout == 60
        assert not api.return_exceptions


class TestLLMApiClientCreation:
    """Test LLM client creation for different providers."""

    @patch.dict(os.environ, {"OPENAI_ACCESS_TOKEN": "test_token"})
    def test_get_openai_client(self, cache, error_handler, openai_model, mock_logger):
        """Test OpenAI client creation."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        with patch("lab_llm.llm_api.ChatOpenAI") as mock_openai:
            client = api.get_client(max_new_tokens=100, temperature=0.7)

            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            assert call_args[1]["api_key"] == "test_token"
            assert call_args[1]["model_name"] == "gpt-4o-mini"
            assert call_args[1]["max_tokens"] == 100
            assert call_args[1]["temperature"] == 0.7
            assert call_args[1]["seed"] == 42

    @patch.dict(os.environ, {"ANTHROPIC_ACCESS_KEY": "test_token"})
    def test_get_anthropic_client(
        self, cache, error_handler, anthropic_model, mock_logger
    ):
        """Test Anthropic client creation."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=anthropic_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        with patch("lab_llm.llm_api.ChatAnthropic") as mock_anthropic:
            client = api.get_client(max_new_tokens=200, temperature=0.5)

            mock_anthropic.assert_called_once()
            call_args = mock_anthropic.call_args
            assert call_args[1]["api_key"] == "test_token"
            assert call_args[1]["model_name"] == "claude-3-5-sonnet-20241022"
            assert call_args[1]["max_tokens"] == 200
            assert call_args[1]["temperature"] == 0.5

    def test_get_client_with_rate_limiter(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test client creation with rate limiting."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        with (
            patch("lab_llm.llm_api.ChatOpenAI") as mock_openai,
            patch("lab_llm.llm_api.InMemoryRateLimiter") as mock_rate_limiter,
            patch.dict(os.environ, {"OPENAI_ACCESS_TOKEN": "test_token"}),
        ):

            api.get_client(requests_per_second=5)

            mock_rate_limiter.assert_called_once_with(
                requests_per_second=5, check_every_n_seconds=0.1, max_bucket_size=10
            )


class TestLLMApiCacheIntegration:
    """Test integration with cache system."""

    def test_get_output_cache_hit(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test get_output when response is in cache."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Pre-populate cache
        cache.save_response(
            "test prompt", "cached response", openai_model, 42, 100, 0.0
        )

        # Should get cached response without calling LLM
        with patch.object(api, "get_client") as mock_get_client:
            result = api.get_output("test prompt", max_new_tokens=100, temperature=0.0)

            # Client should not be called for cache hit
            mock_get_client.assert_not_called()
            assert result == "cached response"

    def test_get_output_cache_miss_mock_llm(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test get_output when response is not in cache."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "new response"

        mock_client = Mock()
        mock_client.invoke.return_value = mock_response

        with patch.object(api, "get_client", return_value=mock_client):
            result = api.get_output("new prompt", max_new_tokens=100, temperature=0.0)

            assert result == mock_response
            mock_client.invoke.assert_called_once()

            # Verify response was cached
            found, cached = cache.get_response("new prompt", openai_model, 42, 100, 0.0)
            assert found
            assert cached == "new response"

    def test_get_output_with_response_model(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test get_output with structured response model."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Mock structured response
        mock_response = SimpleResponse(answer="test answer", confidence=0.95)

        mock_client = Mock()
        mock_client.with_structured_output.return_value = mock_client
        mock_client.invoke.return_value = mock_response

        with patch.object(api, "get_client", return_value=mock_client):
            result = api.get_output(
                "test prompt",
                response_model=SimpleResponse,
                max_new_tokens=100,
                temperature=0.0,
            )

            assert result == mock_response
            mock_client.with_structured_output.assert_called_once_with(SimpleResponse)


class TestLLMApiAsyncOperations:
    """Test async batch operations."""

    @pytest.mark.asyncio
    async def test_get_outputs_with_mock_dataset(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test get_outputs with mocked LLM responses."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Create test dataset
        prompts = ["prompt1", "prompt2", "prompt3"]
        dataset = MockDataset(prompts)

        # Mock client and track batch calls
        call_count = 0

        def mock_abatch_side_effect(batch_prompts, **kwargs):
            nonlocal call_count
            responses = []
            for prompt_messages in batch_prompts:
                # Extract the actual prompt from the message structure
                prompt_content = prompt_messages[1].content  # HumanMessage content
                if "prompt1" in prompt_content:
                    response = Mock()
                    response.content = "response1"
                    responses.append(response)
                elif "prompt2" in prompt_content:
                    response = Mock()
                    response.content = "response2"
                    responses.append(response)
                elif "prompt3" in prompt_content:
                    response = Mock()
                    response.content = "response3"
                    responses.append(response)
            call_count += 1
            return responses

        mock_client = Mock()
        mock_client.abatch = AsyncMock(side_effect=mock_abatch_side_effect)

        with patch.object(api, "get_client", return_value=mock_client):
            results = await api.get_outputs(dataset, batch_size=2)

            assert len(results) == 3
            # Results should contain all three responses, though order may vary due to batching
            assert "response1" in results
            assert "response2" in results
            assert "response3" in results

    @pytest.mark.asyncio
    async def test_get_outputs_basic_functionality(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test basic get_outputs functionality with simple mocking."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        prompts = ["test_prompt"]
        dataset = MockDataset(prompts)

        # Mock the entire batch processing method to avoid complex logic
        expected_results = ["test_response"]

        with patch.object(
            api, "_process_single_batch", return_value=expected_results
        ) as mock_process:
            results = await api.get_outputs(dataset, batch_size=1)

            assert len(results) == 1
            assert results[0] == "test_response"
            mock_process.assert_called()

    @pytest.mark.asyncio
    async def test_get_outputs_with_cached_responses(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test get_outputs when some responses are cached."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Pre-populate cache for first prompt
        cache.save_response("prompt1", "cached_response1", openai_model, 42, 4000, 0.0)

        prompts = ["prompt1", "prompt2"]
        dataset = MockDataset(prompts)

        # Mock response for uncached prompt
        mock_response = Mock()
        mock_response.content = "new_response2"

        mock_client = Mock()
        mock_client.abatch = AsyncMock(return_value=[mock_response])

        with patch.object(api, "get_client", return_value=mock_client):
            results = await api.get_outputs(dataset, batch_size=2)

            assert len(results) == 2
            assert "cached_response1" in results
            assert "new_response2" in results

    @pytest.mark.asyncio
    async def test_get_outputs_with_response_model(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test get_outputs with structured response model."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        prompts = ["prompt1"]
        dataset = MockDataset(prompts)

        # Mock structured response
        mock_response = SimpleResponse(answer="test", confidence=0.9)

        mock_client = Mock()
        mock_client.with_structured_output.return_value = mock_client
        mock_client.abatch = AsyncMock(return_value=[mock_response])

        with patch.object(api, "get_client", return_value=mock_client):
            results = await api.get_outputs(dataset, response_model=SimpleResponse)

            assert len(results) == 1
            assert isinstance(results[0], SimpleResponse)
            assert results[0].answer == "test"
            assert results[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_get_outputs_error_handling(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test error handling in batch operations."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        prompts = ["prompt1"]
        dataset = MockDataset(prompts)

        mock_client = Mock()
        mock_client.abatch = AsyncMock(side_effect=Exception("API Error"))

        with patch.object(api, "get_client", return_value=mock_client):
            with pytest.raises(ValueError, match="Error with LLM batch query"):
                await api.get_outputs(dataset, max_retries=1)


class TestLLMApiSerialization:
    """Test response serialization."""

    def test_serialize_llm_response_string(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test serialization of string responses."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        mock_response = Mock()
        mock_response.content = "test response"

        result = api._serialize_llm_response(mock_response)
        assert result == "test response"

    def test_serialize_llm_response_with_model(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test serialization of structured responses."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        mock_response = SimpleResponse(answer="test", confidence=0.8)

        result = api._serialize_llm_response(
            mock_response, response_model=SimpleResponse
        )
        # Should return JSON string
        assert isinstance(result, str)
        assert "test" in result
        assert "0.8" in result

    def test_serialize_llm_response_error_with_exceptions(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test serialization error handling when return_exceptions=True."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
            return_exceptions=True,
        )

        # Mock response that will fail serialization
        mock_response = Mock()
        mock_response.model_dump_json.side_effect = Exception("Serialization error")

        result = api._serialize_llm_response(
            mock_response, response_model=SimpleResponse
        )
        assert result is None

    def test_serialize_llm_response_error_without_exceptions(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test serialization error handling when return_exceptions=False."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
            return_exceptions=False,
        )

        # Mock response that will fail serialization
        mock_response = Mock()
        mock_response.model_dump_json.side_effect = Exception("Serialization error")

        with pytest.raises(Exception, match="Was unable to serialize llm response"):
            api._serialize_llm_response(mock_response, response_model=SimpleResponse)


class TestLLMApiImageHandling:
    """Test image encoding functionality."""

    def test_encode_images_single_image(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test encoding single image."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            image_path = f.name

        try:
            batch_data = [([{"type": "text", "text": "describe"}], image_path)]
            result = api._encode_images(batch_data)

            assert len(result) == 1
            assert len(result[0]) == 2  # text + image
            assert result[0][1]["type"] == "image_url"
            assert "data:image/jpeg;base64," in result[0][1]["image_url"]["url"]
        finally:
            os.unlink(image_path)

    def test_encode_images_multiple_images(
        self, cache, error_handler, openai_model, mock_logger
    ):
        """Test encoding multiple images with + separator."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=openai_model,
            error_handler=error_handler,
            logging=mock_logger,
        )

        # Create temporary image files
        image_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(f"fake image data {i}".encode())
                image_paths.append(f.name)

        try:
            batch_data = [
                ([{"type": "text", "text": "describe"}], "+".join(image_paths))
            ]
            result = api._encode_images(batch_data)

            assert len(result) == 1
            assert len(result[0]) == 3  # text + 2 images
            assert result[0][1]["type"] == "image_url"
            assert result[0][2]["type"] == "image_url"
        finally:
            for path in image_paths:
                os.unlink(path)
