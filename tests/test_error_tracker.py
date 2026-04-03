"""
Tests for the ErrorTracker decorator-style wrapper.

Tests overall behavior of error handling, classification, and logging.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from litellm import ModelResponse

from lab_llm.new.error_tracker import ErrorTracker, ErrorCategory


class MockCompletion:
    """Mock completion function for testing."""

    def __init__(self, should_fail=False, error_type=None):
        self.should_fail = should_fail
        self.error_type = error_type
        self.call_count = 0

    def __call__(self, model: str, messages=None, **kwargs):
        self.call_count += 1
        self.last_model = model
        self.last_messages = messages or []
        self.last_kwargs = kwargs

        if self.should_fail:
            if self.error_type:
                raise self.error_type("Mock error")
            else:
                raise ValueError("Mock error")

        # Return a mock ModelResponse
        return ModelResponse(
            id="test-123",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop"
            }],
            created=1234567890,
            model=model,
            object="chat.completion",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )


def test_error_tracker_success():
    """Test that ErrorTracker doesn't interfere with successful completions."""
    logger = logging.getLogger("test")
    mock_completion = MockCompletion()

    wrapped = ErrorTracker(logger=logger)(mock_completion)

    response = wrapped(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7
    )

    assert mock_completion.call_count == 1
    assert mock_completion.last_model == "gpt-4o"
    assert mock_completion.last_messages == [{"role": "user", "content": "Hello"}]
    assert mock_completion.last_kwargs["temperature"] == 0.7
    assert response.model == "gpt-4o"


def test_error_classification():
    """Test that errors are classified correctly."""
    logger = logging.getLogger("test")
    tracker = ErrorTracker(logger=logger)

    # Test transient errors
    assert tracker.classify_error(TimeoutError()) == ErrorCategory.TRANSIENT
    assert tracker.classify_error(ConnectionError()) == ErrorCategory.TRANSIENT

    # Test permanent errors
    assert tracker.classify_error(ValueError()) == ErrorCategory.PERMANENT
    assert tracker.classify_error(TypeError()) == ErrorCategory.PERMANENT

    # Test user interrupt
    assert tracker.classify_error(KeyboardInterrupt()) == ErrorCategory.USER_INTERRUPT

    # Test unknown error
    assert tracker.classify_error(RuntimeError()) == ErrorCategory.UNKNOWN


def test_error_logging_and_propagation():
    """Test that errors are logged and then re-raised."""
    logger = Mock()
    mock_completion = MockCompletion(should_fail=True, error_type=ValueError)

    wrapped = ErrorTracker(logger=logger)(mock_completion)

    # Error should be caught, logged, and re-raised
    with pytest.raises(ValueError, match="Mock error"):
        wrapped(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )

    # Check that logging occurred
    assert logger.error.call_count >= 3  # error type, message, model
    logger.error.assert_any_call("LLM Error [permanent]: ValueError")
    logger.error.assert_any_call("Message: Mock error")
    logger.error.assert_any_call("Model: gpt-4o")


def test_jsonl_logging():
    """Test that errors are logged to JSONL file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_errors.jsonl"
        logger = logging.getLogger("test")

        mock_completion = MockCompletion(should_fail=True, error_type=TimeoutError)
        wrapped = ErrorTracker(
            logger=logger,
            log_file=str(log_file),
            include_traceback=True
        )(mock_completion)

        # Trigger an error
        with pytest.raises(TimeoutError):
            wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test message"}],
                temperature=0.5,
                max_tokens=100
            )

        # Check that JSONL file was created and contains error
        assert log_file.exists()

        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 1
        error_record = json.loads(lines[0])

        assert error_record["error_type"] == "TimeoutError"
        assert error_record["error_category"] == "transient"
        assert error_record["error_message"] == "Mock error"
        assert error_record["model"] == "gpt-4o"
        assert error_record["temperature"] == 0.5
        assert error_record["max_tokens"] == 100
        assert "prompt_hash" in error_record
        assert "prompt_preview" in error_record
        assert "timestamp" in error_record
        assert "traceback" in error_record

def test_context_filtering():
    """Test that sensitive context is filtered from logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_errors.jsonl"
        logger = logging.getLogger("test")

        mock_completion = MockCompletion(should_fail=True)
        wrapped = ErrorTracker(logger=logger, log_file=str(log_file))(mock_completion)

        # Include various parameters
        with pytest.raises(ValueError):
            wrapped(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.7,
                api_key="secret-key",  # Should be filtered out
                base_url="https://api.openai.com",  # Should be filtered out
                response_format={"type": "json_object"},  # Should be included
                custom_param="should_not_appear"  # Should be filtered out
            )

        # Check logged context
        with open(log_file) as f:
            error_record = json.loads(f.readline())

        assert error_record["temperature"] == 0.7
        assert error_record["response_format"] == {"type": "json_object"}
        assert "api_key" not in error_record
        assert "base_url" not in error_record
        assert "custom_param" not in error_record


def test_composition_with_other_wrappers():
    """Test that ErrorTracker can be composed with other wrappers."""
    logger = logging.getLogger("test")

    # Create a simple wrapper that adds a header
    def add_header_wrapper(func):
        def wrapped(model, messages=None, **kwargs):
            if messages is None:
                messages = []
            # Add a system message
            messages = [{"role": "system", "content": "You are helpful"}] + messages
            return func(model, messages, **kwargs)
        return wrapped

    mock_completion = MockCompletion()

    # Compose: ErrorTracker wraps the header wrapper
    composed = ErrorTracker(logger=logger)(add_header_wrapper(mock_completion))

    response = composed(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )

    # Check that both wrappers worked
    assert len(mock_completion.last_messages) == 2
    assert mock_completion.last_messages[0]["role"] == "system"
    assert mock_completion.last_messages[1]["role"] == "user"
    assert response.model == "gpt-4o"