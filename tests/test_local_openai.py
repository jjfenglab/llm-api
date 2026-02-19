"""
Tests for local OpenAI-compatible endpoint support.

Run with: pytest tests/test_local_openai.py -v
"""

import os
import logging
from unittest.mock import patch

import pytest

from lab_llm import (
    LLMApi,
    LLMCache,
    DuckDBHandler,
    ErrorCallbackHandler,
    LLMModel,
    LocalOpenAi,
    OpenAi,
    Meta,
    is_local_openai,
)
from lab_llm.constants import is_meta, parse_model_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "test_cache.db"
    handler = DuckDBHandler(str(db_path))
    return LLMCache(handler)


@pytest.fixture
def error_handler():
    return ErrorCallbackHandler(logger)


@pytest.fixture
def local_model():
    return LLMModel(name=LocalOpenAi.LOCAL)


class TestLocalOpenAiEnum:
    """Test LocalOpenAi enum and predicates."""

    def test_local_enum_value(self):
        assert LocalOpenAi.LOCAL.value == "local"

    def test_is_local_openai_predicate(self):
        assert is_local_openai(LocalOpenAi.LOCAL) is True

    def test_is_local_openai_false_for_openai(self):
        assert is_local_openai(OpenAi.GPT4_O_MINI) is False

    def test_is_meta_false_for_local(self):
        assert is_meta(LocalOpenAi.LOCAL) is False

    def test_parse_model_string_local(self):
        model = parse_model_string("local")
        assert model.name == LocalOpenAi.LOCAL

    def test_parse_model_string_local_by_name(self):
        model = parse_model_string("LOCAL")
        assert model.name == LocalOpenAi.LOCAL


class TestLocalOpenAiClient:
    """Test that get_client() returns ChatOpenAI with correct params for local models."""

    def test_get_client_returns_chat_openai(self, cache, error_handler, local_model):
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            base_url="http://localhost:8000/v1",
            local_model_name="Qwen/Qwen3-32B",
        )

        client = api.get_client(max_new_tokens=4000, temperature=0)

        from langchain_openai import ChatOpenAI
        assert isinstance(client, ChatOpenAI)

    def test_get_client_base_url(self, cache, error_handler, local_model):
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            base_url="http://myserver:9000/v1",
            local_model_name="test-model",
        )

        client = api.get_client()
        assert str(client.openai_api_base) == "http://myserver:9000/v1"

    def test_get_client_model_name(self, cache, error_handler, local_model):
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            base_url="http://localhost:8000/v1",
            local_model_name="meta-llama/Llama-3.3-70B-Instruct",
        )

        client = api.get_client()
        assert client.model_name == "meta-llama/Llama-3.3-70B-Instruct"

    def test_get_client_env_var_fallback_base_url(self, cache, error_handler, local_model):
        """base_url falls back to LOCAL_LLM_BASE_URL env var."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            local_model_name="test-model",
        )

        with patch.dict(os.environ, {"LOCAL_LLM_BASE_URL": "http://envserver:7000/v1"}):
            client = api.get_client()
            assert str(client.openai_api_base) == "http://envserver:7000/v1"

    def test_get_client_env_var_fallback_api_key(self, cache, error_handler, local_model):
        """api_key falls back to LOCAL_LLM_API_KEY env var."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            base_url="http://localhost:8000/v1",
            local_model_name="test-model",
        )

        with patch.dict(os.environ, {"LOCAL_LLM_API_KEY": "my-secret-key"}):
            client = api.get_client()
            assert client.openai_api_key.get_secret_value() == "my-secret-key"

    def test_get_client_default_api_key(self, cache, error_handler, local_model):
        """api_key defaults to 'not-needed' when env var is unset."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            base_url="http://localhost:8000/v1",
            local_model_name="test-model",
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCAL_LLM_API_KEY", None)
            client = api.get_client()
            assert client.openai_api_key.get_secret_value() == "not-needed"

    def test_get_client_default_base_url(self, cache, error_handler, local_model):
        """base_url defaults to http://localhost:8000/v1 when nothing is set."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            local_model_name="test-model",
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCAL_LLM_BASE_URL", None)
            client = api.get_client()
            assert str(client.openai_api_base) == "http://localhost:8000/v1"

    def test_get_client_default_model_name(self, cache, error_handler, local_model):
        """model name defaults to 'default' when local_model_name is not set."""
        api = LLMApi(
            cache=cache,
            seed=42,
            model_type=local_model,
            error_handler=error_handler,
            logging=logger,
            base_url="http://localhost:8000/v1",
        )

        client = api.get_client()
        assert client.model_name == "default"


class TestStructuredOutputSupport:
    """Test _supports_native_structured_output() behavior."""

    def test_openai_supports_structured_output(self, cache, error_handler):
        model = LLMModel(name=OpenAi.GPT4_O_MINI)
        api = LLMApi(
            cache=cache, seed=42, model_type=model,
            error_handler=error_handler, logging=logger,
        )
        assert api._supports_native_structured_output() is True

    def test_meta_does_not_support_structured_output(self, cache, error_handler):
        model = LLMModel(name=Meta.LLAMA_3_3_70B)
        api = LLMApi(
            cache=cache, seed=42, model_type=model,
            error_handler=error_handler, logging=logger,
        )
        assert api._supports_native_structured_output() is False

    def test_local_defaults_to_supports_structured_output(self, cache, error_handler, local_model):
        api = LLMApi(
            cache=cache, seed=42, model_type=local_model,
            error_handler=error_handler, logging=logger,
            base_url="http://localhost:8000/v1",
        )
        assert api._supports_native_structured_output() is True

    def test_local_can_disable_structured_output(self, cache, error_handler, local_model):
        api = LLMApi(
            cache=cache, seed=42, model_type=local_model,
            error_handler=error_handler, logging=logger,
            base_url="http://localhost:8000/v1",
            native_structured_output=False,
        )
        assert api._supports_native_structured_output() is False
