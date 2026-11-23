"""
Unit tests for CachingCompletion wrapper.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from litellm import ModelResponse, Message
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from litellm.types.utils import Usage

from lab_llm.caching_completion import CachingCompletion


class TestCachingCompletion:
    """Test suite for CachingCompletion wrapper."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.TemporaryDirectory() as tempdir:
            yield Path(tempdir)

    @pytest.fixture
    def mock_completion_func(self):
        """Mock completion function that returns a simple response."""
        def mock_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            return ModelResponse(
                id="chatcmpl-test",
                object="chat.completion",
                created=1234567890,
                model=model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content="Test response"
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15
                )
            )
        return mock_func

    @pytest.fixture
    def caching_completion(self, temp_db_path):
        """Create a CachingCompletion instance with temporary database."""
        return CachingCompletion(temp_db_path / "test.db")

    def test_init_creates_database_and_tables(self, temp_db_path):
        """Test that initialization creates database and required tables."""
        db_file = temp_db_path / "test.db"

        # Database shouldn't exist initially
        assert not os.path.exists(db_file)

        # Create CachingCompletion
        cache = CachingCompletion(db_file)

        # Database file should now exist
        assert os.path.exists(db_file)

        # Verify table exists via SQLAlchemy inspector
        from sqlalchemy import inspect as sa_inspect
        insp = sa_inspect(cache.engine)
        assert 'completion_cache' in insp.get_table_names()

    def test_init_idempotent(self, temp_db_path):
        """Test that creating a second CachingCompletion on the same path doesn't error."""
        db_file = temp_db_path / "test.db"
        cache1 = CachingCompletion(db_file)
        # Should not raise even though table already exists
        cache2 = CachingCompletion(db_file)
        from sqlalchemy import inspect as sa_inspect
        assert 'completion_cache' in sa_inspect(cache2.engine).get_table_names()

    def test_connection_string_detection(self, caching_completion):
        """Test that connection strings are correctly detected vs plain filepaths."""
        cache = caching_completion

        # Plain filepaths should NOT be detected as connection strings
        assert not cache._is_connection_string("./cache.db")
        assert not cache._is_connection_string("/abs/path/cache.db")
        assert not cache._is_connection_string(Path("cache.db"))
        assert not cache._is_connection_string("cache.db")

        # SQLAlchemy URLs should be detected
        assert cache._is_connection_string("postgresql://localhost/db")
        assert cache._is_connection_string("sqlite:///./cache.db")
        assert cache._is_connection_string("duckdb:///./cache.db")
        assert cache._is_connection_string("mysql+pymysql://user:pass@localhost/db")

    def test_init_with_sqlite_connection_string(self, temp_db_path, mock_completion_func):
        """Test that a SQLAlchemy connection string (SQLite) is accepted and works end-to-end."""
        url = f"sqlite:///{temp_db_path / 'sqlite_test.db'}"
        cache = CachingCompletion(url)

        # Table must exist
        from sqlalchemy import inspect as sa_inspect
        insp = sa_inspect(cache.engine)
        assert 'completion_cache' in insp.get_table_names()

        # Full cache lifecycle
        wrapped = cache(mock_completion_func)
        response = wrapped("gpt-4o", [{"role": "user", "content": "Hello"}])
        assert response.choices[0].message.content == "Test response"

        # Response should have been cached
        cache_key = cache._compute_cache_key("gpt-4o", [{"role": "user", "content": "Hello"}], {})
        assert cache._get_cached_response(cache_key) is not None

    def test_filter_kwargs_for_cache(self, caching_completion):
        """Test kwargs filtering for cache key computation."""
        cache = caching_completion

        # Test basic parameter inclusion/exclusion
        kwargs = {
            'temperature': 0.7,
            'seed': 42,
            'api_key': 'secret',
            'api_base': 'https://api.example.com',
            'timeout': 30,
            'top_p': 0.9
        }

        filtered = cache._filter_kwargs_for_cache(**kwargs)

        # Should include non-api_ parameters
        assert 'temperature' in filtered
        assert 'seed' in filtered
        assert 'timeout' in filtered
        assert 'top_p' in filtered

        # Should exclude api_ parameters
        assert 'api_key' not in filtered
        assert 'api_base' not in filtered

        # Test response_format conversion
        class MockResponseFormat:
            def model_json_schema(self):
                return {"type": "json_object"}

        kwargs_with_response_format = {
            'response_format': MockResponseFormat(),
            'temperature': 0.5
        }

        filtered = cache._filter_kwargs_for_cache(**kwargs_with_response_format)
        assert filtered['response_format'] == {"type": "json_object"}
        assert filtered['temperature'] == 0.5

    def test_compute_cache_key_deterministic(self, caching_completion):
        """Test that cache key computation is deterministic."""
        cache = caching_completion

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"temperature": 0.7, "seed": 42}

        # Compute key multiple times
        key1 = cache._compute_cache_key(model, messages, kwargs)
        key2 = cache._compute_cache_key(model, messages, kwargs)
        key3 = cache._compute_cache_key(model, messages, kwargs)

        # Should be identical
        assert key1 == key2 == key3

        # Should be a valid SHA-256 hash (64 hex characters)
        assert len(key1) == 64
        assert all(c in '0123456789abcdef' for c in key1)

    def test_compute_cache_key_different_inputs(self, caching_completion):
        """Test that different inputs produce different cache keys."""
        cache = caching_completion

        base_model = "gpt-4o"
        base_messages = [{"role": "user", "content": "Hello"}]
        base_kwargs = {"temperature": 0.7}

        base_key = cache._compute_cache_key(base_model, base_messages, base_kwargs)

        # Different model should produce different key
        different_model_key = cache._compute_cache_key("gpt-3.5-turbo", base_messages, base_kwargs)
        assert base_key != different_model_key

        # Different messages should produce different key
        different_messages = [{"role": "user", "content": "Goodbye"}]
        different_messages_key = cache._compute_cache_key(base_model, different_messages, base_kwargs)
        assert base_key != different_messages_key

        # Different kwargs should produce different key
        different_kwargs = {"temperature": 0.5}
        different_kwargs_key = cache._compute_cache_key(base_model, base_messages, different_kwargs)
        assert base_key != different_kwargs_key

    def test_cache_miss_and_storage(self, caching_completion, mock_completion_func):
        """Test cache miss followed by response storage."""
        cache = caching_completion
        wrapped = cache(mock_completion_func)

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        # First call should be cache miss
        response1 = wrapped(model, messages, temperature=0.7)

        # Verify response
        assert response1.model == model
        assert response1.choices[0].message.content == "Test response"

        # Verify response was cached
        cache_key = cache._compute_cache_key(model, messages, {"temperature": 0.7})
        cached_response = cache._get_cached_response(cache_key)
        assert cached_response is not None
        assert cached_response.choices[0].message.content == "Test response"

    def test_cache_hit(self, caching_completion, mock_completion_func):
        """Test cache hit on second identical request."""
        cache = caching_completion

        # Mock the wrapped function to count calls
        call_count = 0
        def counting_mock_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            return mock_completion_func(model, messages, **kwargs)

        wrapped = cache(counting_mock_func)

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"temperature": 0.7, "seed": 42}

        # First call - should hit the mock function
        response1 = wrapped(model, messages, **kwargs)
        assert call_count == 1

        # Second identical call - should hit cache
        response2 = wrapped(model, messages, **kwargs)
        assert call_count == 1  # Mock function should not be called again

        # Responses should be equivalent
        assert response1.choices[0].message.content == response2.choices[0].message.content
        assert response1.model == response2.model

    def test_error_handling_in_cache_operations(self, caching_completion):
        """Test graceful handling of cache operation errors."""
        cache = caching_completion

        # Mock a function that works normally
        def normal_func(model: str, messages: List = None, **kwargs) -> ModelResponse:
            return ModelResponse(
                id="test",
                object="chat.completion",
                created=123,
                model=model,
                choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="OK"), finish_reason="stop")],
                usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            )

        wrapped = cache(normal_func)

        # Corrupt the engine so DB operations fail
        cache.engine.dispose()
        cache.engine = None

        # Should still work despite cache errors (graceful degradation)
        response = wrapped("gpt-4o", [{"role": "user", "content": "Hello"}])
        assert response.choices[0].message.content == "OK"

    def test_api_parameter_exclusion(self, caching_completion):
        """Test that parameters starting with 'api_' are excluded from cache key."""
        cache = caching_completion

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Hello"}]

        # These should produce the same cache key
        kwargs1 = {"temperature": 0.7, "api_key": "key1", "api_base": "base1"}
        kwargs2 = {"temperature": 0.7, "api_key": "key2", "api_base": "base2"}

        key1 = cache._compute_cache_key(model, messages, kwargs1)
        key2 = cache._compute_cache_key(model, messages, kwargs2)

        assert key1 == key2

    def test_serialization_roundtrip(self, temp_db_path, mock_completion_func):
        """Test that response serialization and deserialization preserves data."""
        cache = CachingCompletion(temp_db_path / "test.db", return_original_usage=True)

        # Create a more complex response
        original_response = ModelResponse(
            id="chatcmpl-complex",
            object="chat.completion",
            created=1234567890,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Complex response with metadata"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=15,
                completion_tokens=8,
                total_tokens=23
            )
        )

        # Serialize and deserialize
        serialized = cache._serialize_model_response(original_response)
        deserialized = cache._build_model_response_from_cache(serialized)

        # Check key properties are preserved
        assert deserialized.id == original_response.id
        assert deserialized.model == original_response.model
        assert deserialized.choices[0].message.content == original_response.choices[0].message.content
        assert deserialized.usage.total_tokens == original_response.usage.total_tokens
