"""
Tests for the UsageTracker wrapper class.
"""

import pytest
from unittest.mock import Mock
from litellm import ModelResponse

from lab_llm.new.usage_tracker import UsageTracker


class MockUsage:
    """Mock usage object for testing."""
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=None,
                 cached_tokens=0, reasoning_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens or (prompt_tokens + completion_tokens)

        # Mock details objects
        self.prompt_tokens_details = None
        self.completion_tokens_details = None

        if cached_tokens > 0:
            self.prompt_tokens_details = Mock()
            self.prompt_tokens_details.cached_tokens = cached_tokens

        if reasoning_tokens > 0:
            self.completion_tokens_details = Mock()
            self.completion_tokens_details.reasoning_tokens = reasoning_tokens


def create_mock_response(input_tokens=10, output_tokens=20, total_tokens=None,
                        cached_tokens=0, reasoning_tokens=0):
    """Create a mock ModelResponse with usage data."""
    response = Mock(spec=ModelResponse)
    response.usage = MockUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens
    )
    response.choices = []
    return response


def test_usage_tracker_basic_functionality():
    """Test basic usage tracking functionality."""
    # Create mock completion function
    mock_completion = Mock()
    mock_completion.return_value = create_mock_response(input_tokens=10, output_tokens=20)

    # Wrap with usage tracker
    tracker = UsageTracker()
    tracked_completion = tracker(mock_completion)

    # Make a call
    response = tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])

    # Verify the original function was called
    mock_completion.assert_called_once_with(
        "gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )

    # Verify usage tracking
    last_usage = tracker.last_usage()
    assert last_usage is not None
    assert last_usage["input_tokens"] == 10
    assert last_usage["output_tokens"] == 20
    assert last_usage["total_tokens"] == 30

    # Total usage should be the same for single call
    total_usage = tracker.total_usage()
    assert total_usage == last_usage


def test_usage_tracker_cumulative_usage():
    """Test cumulative usage tracking across multiple calls."""
    mock_completion = Mock()

    # Set up different responses for multiple calls
    mock_completion.side_effect = [
        create_mock_response(input_tokens=10, output_tokens=20),
        create_mock_response(input_tokens=15, output_tokens=25),
        create_mock_response(input_tokens=5, output_tokens=10)
    ]

    # Wrap with usage tracker
    tracker = UsageTracker()
    tracked_completion = tracker(mock_completion)

    # Make multiple calls
    tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])
    tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "Hi again"}])
    tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "One more"}])

    # Check last usage (should be from the third call)
    last_usage = tracker.last_usage()
    assert last_usage["input_tokens"] == 5
    assert last_usage["output_tokens"] == 10
    assert last_usage["total_tokens"] == 15

    # Check total usage (cumulative)
    total_usage = tracker.total_usage()
    assert total_usage["input_tokens"] == 30  # 10 + 15 + 5
    assert total_usage["output_tokens"] == 55  # 20 + 25 + 10
    assert total_usage["total_tokens"] == 85   # 30 + 40 + 15


def test_usage_tracker_with_cached_and_reasoning_tokens():
    """Test usage tracking with cached and reasoning tokens."""
    mock_completion = Mock()
    mock_completion.return_value = create_mock_response(
        input_tokens=100,
        output_tokens=50,
        cached_tokens=30,
        reasoning_tokens=20
    )

    # Wrap with usage tracker
    tracker = UsageTracker()
    tracked_completion = tracker(mock_completion)

    # Make a call
    tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])

    # Verify usage includes cached and reasoning tokens
    last_usage = tracker.last_usage()
    assert last_usage["input_tokens"] == 100
    assert last_usage["output_tokens"] == 50
    assert last_usage["total_tokens"] == 150
    assert last_usage["cached_tokens"] == 30
    assert last_usage["reasoning_tokens"] == 20


def test_usage_tracker_no_usage_data():
    """Test behavior when no usage data is available."""
    mock_completion = Mock()

    # Create response with no usage data
    response = Mock(spec=ModelResponse)
    response.usage = None
    response.choices = []
    mock_completion.return_value = response

    # Wrap with usage tracker
    tracker = UsageTracker()
    tracked_completion = tracker(mock_completion)

    # Make a call
    tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])

    # Should return None for both methods
    assert tracker.last_usage() is None
    assert tracker.total_usage() is None


def test_usage_tracker_composition_with_other_wrappers():
    """Test that UsageTracker can be composed with other wrappers."""
    # Mock another wrapper
    mock_wrapper = Mock()
    mock_inner_completion = Mock()
    mock_inner_completion.return_value = create_mock_response(input_tokens=15, output_tokens=30)
    mock_wrapper.return_value = mock_inner_completion

    # Mock base completion function
    mock_base_completion = Mock()

    # Compose: UsageTracker(OtherWrapper(base_completion))
    tracker = UsageTracker()
    wrapped_once = mock_wrapper(mock_base_completion)
    tracked_completion = tracker(wrapped_once)

    # Make a call
    response = tracked_completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])

    # Verify the chain was called
    mock_wrapper.assert_called_once_with(mock_base_completion)
    mock_inner_completion.assert_called_once_with(
        "gpt-4o",
        messages=[{"role": "user", "content": "Hello"}]
    )

    # Verify usage tracking still works
    last_usage = tracker.last_usage()
    assert last_usage["input_tokens"] == 15
    assert last_usage["output_tokens"] == 30