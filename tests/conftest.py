"""
Shared pytest fixtures for all test files.
"""

import os
import tempfile
import shutil
from unittest.mock import Mock
import pytest

from lab_llm.llm_cache import LLMCache
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.constants import LLMModel, OpenAi, Anthropic


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_cache.db")
    yield db_path
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def db_handler(temp_db_path):
    """Create a DuckDBHandler instance with temporary database."""
    handler = DuckDBHandler(temp_db_path, read_only=False, max_retries=3)
    yield handler
    handler.close_connection()


@pytest.fixture
def cache(db_handler):
    """Create an LLMCache instance."""
    return LLMCache(db_handler)


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def error_handler(mock_logger):
    """Create an error handler."""
    return ErrorCallbackHandler(mock_logger)


@pytest.fixture
def openai_model():
    """Create an OpenAI model for testing."""
    return LLMModel(name=OpenAi.GPT4_O_MINI)


@pytest.fixture
def anthropic_model():
    """Create an Anthropic model for testing."""
    return LLMModel(name=Anthropic.CLAUDE_3_5_SONNET)