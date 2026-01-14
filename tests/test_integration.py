"""
Integration tests for lab_llm.

These tests verify:
1. Library imports work correctly
2. API initialization works
3. Versa endpoint configuration is properly enforced
4. Real API calls work (when credentials are available)

Run with: pytest tests/test_integration.py -v
"""

import os
import logging
import pytest
from unittest.mock import patch

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestImports:
    """Test that all public imports work correctly."""

    def test_import_main_classes(self):
        """Test importing main classes from lab_llm."""
        from lab_llm import LLMApi, LLMCache, DuckDBHandler

        assert LLMApi is not None
        assert LLMCache is not None
        assert DuckDBHandler is not None

    def test_import_error_handling(self):
        """Test importing error handling classes."""
        from lab_llm import ErrorTracker, ErrorCallbackHandler

        assert ErrorTracker is not None
        assert ErrorCallbackHandler is not None

    def test_import_datasets(self):
        """Test importing dataset classes."""
        from lab_llm import TextDataset, ImageDataset

        assert TextDataset is not None
        assert ImageDataset is not None

    def test_import_model_enums(self):
        """Test importing model enums."""
        from lab_llm import (
            LLMModel,
            OpenAi,
            VersaOpenAi,
            Claude,
            Meta,
            Cohere,
            Qwen,
        )

        assert LLMModel is not None
        assert OpenAi is not None
        assert VersaOpenAi is not None
        assert Claude is not None
        assert Meta is not None
        assert Cohere is not None
        assert Qwen is not None

    def test_import_reasoning_utilities(self):
        """Test importing reasoning model utilities."""
        from lab_llm import REASONING_MODELS, is_reasoning_model

        assert REASONING_MODELS is not None
        assert callable(is_reasoning_model)

    def test_version_defined(self):
        """Test that version is defined."""
        from lab_llm import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)


class TestApiInitialization:
    """Test API initialization and configuration."""

    def test_create_duckdb_handler(self, tmp_path):
        """Test creating a DuckDB handler with a temp path."""
        from lab_llm import DuckDBHandler

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))

        assert handler is not None

    def test_create_cache(self, tmp_path):
        """Test creating an LLM cache."""
        from lab_llm import DuckDBHandler, LLMCache

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)

        assert cache is not None

    def test_create_error_tracker(self, tmp_path):
        """Test creating an error tracker."""
        from lab_llm import ErrorTracker

        log_path = tmp_path / "errors.jsonl"
        tracker = ErrorTracker(str(log_path))

        assert tracker is not None

    def test_create_llm_api(self, tmp_path):
        """Test creating an LLMApi instance."""
        from lab_llm import (
            LLMApi,
            LLMCache,
            DuckDBHandler,
            ErrorCallbackHandler,
            LLMModel,
            OpenAi,
        )

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)
        model = LLMModel(name=OpenAi.GPT4_O_MINI)
        error_handler = ErrorCallbackHandler(logger)

        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=model,
            error_handler=error_handler,
            logging=logger,
        )

        assert api is not None


class TestVersaEndpointConfiguration:
    """Test Versa endpoint configuration requirements."""

    def test_versa_requires_endpoint_env_var(self, tmp_path):
        """Test that Versa models raise error when VERSA_ENDPOINT is not set."""
        from lab_llm import (
            LLMApi,
            LLMCache,
            DuckDBHandler,
            ErrorCallbackHandler,
            LLMModel,
            VersaOpenAi,
        )

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)
        model = LLMModel(name=VersaOpenAi.GPT4_O_2024_08)
        error_handler = ErrorCallbackHandler(logger)

        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=model,
            error_handler=error_handler,
            logging=logger,
        )

        # Remove VERSA_ENDPOINT if it exists
        with patch.dict(os.environ, {}, clear=False):
            # Temporarily remove VERSA_ENDPOINT
            env_backup = os.environ.pop("VERSA_ENDPOINT", None)

            try:
                # Attempting to get client should raise ValueError
                with pytest.raises(ValueError) as exc_info:
                    api.get_client(max_new_tokens=100, temperature=0)

                assert "VERSA_ENDPOINT" in str(exc_info.value)
                assert "environment variable" in str(exc_info.value).lower()
            finally:
                # Restore VERSA_ENDPOINT if it was set
                if env_backup is not None:
                    os.environ["VERSA_ENDPOINT"] = env_backup

    def test_versa_works_with_endpoint_set(self, tmp_path):
        """Test that Versa models work when VERSA_ENDPOINT is set."""
        from lab_llm import (
            LLMApi,
            LLMCache,
            DuckDBHandler,
            ErrorCallbackHandler,
            LLMModel,
            VersaOpenAi,
        )

        # Skip if no Versa credentials
        if not os.environ.get("VERSA_ENDPOINT") or not os.environ.get("VERSA_API_KEY"):
            pytest.skip("VERSA_ENDPOINT and VERSA_API_KEY required for this test")

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)
        model = LLMModel(name=VersaOpenAi.GPT4_O_2024_08)
        error_handler = ErrorCallbackHandler(logger)

        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=model,
            error_handler=error_handler,
            logging=logger,
        )

        # Should not raise an error when getting client
        client = api.get_client(max_new_tokens=100, temperature=0)
        assert client is not None


class TestTextDataset:
    """Test TextDataset functionality."""

    def test_create_text_dataset(self):
        """Test creating a TextDataset."""
        from lab_llm import TextDataset

        prompts = ["Hello", "World"]
        dataset = TextDataset(prompts)

        assert len(dataset) == 2

    def test_text_dataset_iteration(self):
        """Test iterating over TextDataset."""
        from lab_llm import TextDataset

        prompts = ["Hello", "World", "Test"]
        dataset = TextDataset(prompts)

        items = [dataset[i] for i in range(len(dataset))]
        assert len(items) == 3


# Optional integration tests that require API credentials
class TestOpenAIIntegration:
    """Integration tests for OpenAI models (requires OPENAI_ACCESS_TOKEN)."""

    @pytest.fixture
    def api(self, tmp_path):
        """Create an LLMApi instance for OpenAI."""
        from lab_llm import (
            LLMApi,
            LLMCache,
            DuckDBHandler,
            ErrorCallbackHandler,
            LLMModel,
            OpenAi,
        )

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)
        model = LLMModel(name=OpenAi.GPT4_O_MINI)
        error_handler = ErrorCallbackHandler(logger)

        return LLMApi(
            cache=cache,
            seed=42,
            model_type=model,
            error_handler=error_handler,
            logging=logger,
            timeout=60,
        )

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_ACCESS_TOKEN"),
        reason="OPENAI_ACCESS_TOKEN required"
    )
    def test_single_prompt(self, api):
        """Test a single prompt with OpenAI."""
        response = api.get_output("What is 2 + 2? Answer with just the number.")

        assert response is not None
        # Response could be a string or an AIMessage object
        response_text = str(response.content) if hasattr(response, 'content') else str(response)
        assert "4" in response_text

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_ACCESS_TOKEN"),
        reason="OPENAI_ACCESS_TOKEN required"
    )
    def test_batch_prompts(self, api):
        """Test batch prompts with OpenAI."""
        import asyncio
        from lab_llm import TextDataset

        prompts = ["What is 1+1?", "What is 2+2?"]
        dataset = TextDataset(prompts)

        responses = asyncio.run(api.get_outputs(dataset, batch_size=2))

        assert len(responses) == 2
        assert all(r is not None for r in responses)


class TestVersaIntegration:
    """Integration tests for Versa/Azure OpenAI models."""

    @pytest.fixture
    def api(self, tmp_path):
        """Create an LLMApi instance for Versa."""
        from lab_llm import (
            LLMApi,
            LLMCache,
            DuckDBHandler,
            ErrorCallbackHandler,
            LLMModel,
            VersaOpenAi,
        )

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)
        model = LLMModel(name=VersaOpenAi.GPT4_O_2024_08)
        error_handler = ErrorCallbackHandler(logger)

        return LLMApi(
            cache=cache,
            seed=42,
            model_type=model,
            error_handler=error_handler,
            logging=logger,
            timeout=60,
        )

    @pytest.mark.skipif(
        not (os.environ.get("VERSA_ENDPOINT") and os.environ.get("VERSA_API_KEY")),
        reason="VERSA_ENDPOINT and VERSA_API_KEY required"
    )
    def test_single_prompt_versa(self, api):
        """Test a single prompt with Versa."""
        response = api.get_output("What is 2 + 2? Answer with just the number.")

        assert response is not None
        # Response could be a string or an AIMessage object
        response_text = str(response.content) if hasattr(response, 'content') else str(response)
        assert "4" in response_text


class TestBedrockIntegration:
    """Integration tests for AWS Bedrock models."""

    @pytest.fixture
    def api(self, tmp_path):
        """Create an LLMApi instance for Bedrock Claude."""
        from lab_llm import (
            LLMApi,
            LLMCache,
            DuckDBHandler,
            ErrorCallbackHandler,
            LLMModel,
            Claude,
        )

        db_path = tmp_path / "test_cache.db"
        handler = DuckDBHandler(str(db_path))
        cache = LLMCache(handler)
        model = LLMModel(name=Claude.HAIKU_3_5)
        error_handler = ErrorCallbackHandler(logger)

        return LLMApi(
            cache=cache,
            seed=42,
            model_type=model,
            error_handler=error_handler,
            logging=logger,
            timeout=60,
        )

    @pytest.mark.skipif(
        not (os.environ.get("BEDROCK_ACCESS_KEY") and os.environ.get("BEDROCK_ACCESS_KEY_SECRET")),
        reason="BEDROCK_ACCESS_KEY and BEDROCK_ACCESS_KEY_SECRET required"
    )
    def test_single_prompt_bedrock(self, api):
        """Test a single prompt with Bedrock Claude."""
        response = api.get_output("What is 2 + 2? Answer with just the number.")

        assert response is not None
        # Response could be a string or an AIMessage object
        response_text = str(response.content) if hasattr(response, 'content') else str(response)
        assert "4" in response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
