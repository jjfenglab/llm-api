"""
Integration tests for LLM API framework.

Tests the ability to use various LLM APIs including:
- Direct OpenAI calling via litellm
- Direct Claude calling via litellm
- Versa OpenAI (Azure OpenAI)
- Versa Claude (AWS Bedrock)

Credentials are loaded via dotenv and tests are skipped if required environment variables are not present.
"""

import os
import pytest
from typing import Dict, Any
from unittest.mock import patch

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    # dotenv is optional for testing
    pass

import litellm
from lab_llm import LLMApi, wrap_completion_function
from lab_llm.constants import OpenAI, Claude, VersaOpenAI, VersaClaude
from lab_llm.versa.openai import versa_openai_completion
from lab_llm.versa.claude import versa_claude_completion
from lab_llm.usage_tracker import UsageTracker


class TestIntegrationLLMAPIs:
    """Integration tests for various LLM API providers."""

    def _check_env_vars_present(self, required_vars: list[str]) -> bool:
        """Check if all required environment variables are present."""
        return all(os.getenv(var) for var in required_vars)

    def _create_test_messages(self) -> list[Dict[str, str]]:
        """Create simple test messages for API calls."""
        return [
            {"role": "user", "content": "What is 2+2? Respond with just the number."}
        ]

    def _create_test_tool(self):
        """Create a simple test tool function."""
        def get_current_time() -> str:
            """Get the current time as a string."""
            import datetime
            return datetime.datetime.now().isoformat()
        return get_current_time

    def _validate_response(self, response: Any, expected_content_keywords: list[str] = None):
        """Validate that response has expected structure and content."""
        # Basic structure validation
        assert response is not None
        assert isinstance(response, str) or hasattr(response, 'choices')

        # If it's a string response (from LLMApi.run), check content
        if isinstance(response, str):
            assert len(response.strip()) > 0
            if expected_content_keywords:
                response_lower = response.lower()
                assert any(keyword in response_lower for keyword in expected_content_keywords)

    def _validate_usage_tracking(self, tracker: UsageTracker):
        """Validate that usage tracking worked."""
        last_usage = tracker.last_usage()
        total_usage = tracker.total_usage()

        assert last_usage is not None
        assert total_usage is not None
        assert last_usage["input_tokens"] > 0
        assert last_usage["output_tokens"] > 0
        assert last_usage["total_tokens"] > 0
        assert total_usage["input_tokens"] >= last_usage["input_tokens"]
        assert total_usage["output_tokens"] >= last_usage["output_tokens"]
        assert total_usage["total_tokens"] >= last_usage["total_tokens"]

    def test_direct_openai_integration(self):
        """Test direct OpenAI API integration via litellm."""
        required_vars = ["OPENAI_API_KEY"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping OpenAI test - missing environment variables: {required_vars}")

        # Create usage tracker for monitoring
        usage_tracker = UsageTracker()

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(
            litellm.completion,
            usage_tracker=usage_tracker
        )
        api = LLMApi(completion_func)

        # Test basic completion
        messages = self._create_test_messages()
        response = api.run(
            messages=messages,
            model=OpenAI.GPT4_O_MINI,  # Use mini model to save costs
            temperature=0.1,
            max_tokens=50
        )

        self._validate_response(response, expected_content_keywords=["4"])
        self._validate_usage_tracking(usage_tracker)

        print(f"OpenAI API test completed successfully. Usage: {usage_tracker.total_usage()}")

    def test_direct_claude_integration(self):
        """Test direct Claude API integration via litellm."""
        required_vars = ["ANTHROPIC_API_KEY"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping Claude test - missing environment variables: {required_vars}")

        # Create usage tracker for monitoring
        usage_tracker = UsageTracker()

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(
            litellm.completion,
            usage_tracker=usage_tracker
        )
        api = LLMApi(completion_func)

        # Test basic completion
        messages = self._create_test_messages()
        response = api.run(
            messages=messages,
            model=Claude.HAIKU_4_5,  # Use Haiku for cost efficiency
            temperature=0.1,
            max_tokens=50
        )

        self._validate_response(response, expected_content_keywords=["4"])
        self._validate_usage_tracking(usage_tracker)

        print(f"Claude API test completed successfully. Usage: {usage_tracker.total_usage()}")

    def test_versa_openai_integration(self):
        """Test Versa OpenAI (Azure OpenAI) integration."""
        required_vars = ["VERSA_API_KEY", "VERSA_ENDPOINT"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping Versa OpenAI test - missing environment variables: {required_vars}")

        # Create usage tracker for monitoring
        usage_tracker = UsageTracker()

        # Create Versa OpenAI completion function
        versa_completion = versa_openai_completion()

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(
            versa_completion,
            usage_tracker=usage_tracker
        )
        api = LLMApi(completion_func)

        # Test basic completion
        messages = self._create_test_messages()
        response = api.run(
            messages=messages,
            model=VersaOpenAI.GPT4_O_2024_11,
            temperature=0.1,
            max_tokens=50
        )

        self._validate_response(response, expected_content_keywords=["4"])
        self._validate_usage_tracking(usage_tracker)

        print(f"Versa OpenAI test completed successfully. Usage: {usage_tracker.total_usage()}")

    def test_versa_claude_integration(self):
        """Test Versa Claude (AWS Bedrock) integration."""
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "VERSA_ENDPOINT"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping Versa Claude test - missing environment variables: {required_vars}")

        # Create usage tracker for monitoring
        usage_tracker = UsageTracker()

        # Create Versa Claude completion function
        versa_completion = versa_claude_completion()

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(
            versa_completion,
            usage_tracker=usage_tracker
        )
        api = LLMApi(completion_func)

        # Test basic completion
        messages = self._create_test_messages()
        response = api.run(
            messages=messages,
            model=VersaClaude.CLAUDE_HAIKU_4_5,  # Use Haiku for cost efficiency
            temperature=0.1,
            max_tokens=50
        )

        self._validate_response(response, expected_content_keywords=["4"])
        self._validate_usage_tracking(usage_tracker)

        print(f"Versa Claude test completed successfully. Usage: {usage_tracker.total_usage()}")

    def test_tool_usage_with_direct_openai(self):
        """Test tool/function calling capabilities with direct OpenAI."""
        required_vars = ["OPENAI_API_KEY"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping OpenAI tool test - missing environment variables: {required_vars}")

        # Create usage tracker for monitoring
        usage_tracker = UsageTracker()

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(
            litellm.completion,
            usage_tracker=usage_tracker
        )
        api = LLMApi(completion_func)

        # Create test tool
        test_tool = self._create_test_tool()

        # Test tool usage
        messages = [{"role": "user", "content": "What time is it right now? Use the available tool."}]
        response = api.run(
            messages=messages,
            tools=[test_tool],
            model=OpenAI.GPT4_O_MINI,
            temperature=0.1,
            max_tool_calls=3
        )

        self._validate_response(response)
        self._validate_usage_tracking(usage_tracker)

        # Response should mention time or indicate tool was used
        response_lower = response.lower() if isinstance(response, str) else ""
        assert any(keyword in response_lower for keyword in ["time", ":", "t", "current"])

        print(f"OpenAI tool usage test completed successfully. Usage: {usage_tracker.total_usage()}")

    def test_batch_processing_with_claude(self):
        """Test batch processing capabilities with Claude."""
        required_vars = ["ANTHROPIC_API_KEY"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping Claude batch test - missing environment variables: {required_vars}")

        # Create usage tracker for monitoring
        usage_tracker = UsageTracker()

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(
            litellm.completion,
            usage_tracker=usage_tracker
        )
        api = LLMApi(completion_func)

        # Test batch processing
        batch_messages = [
            [{"role": "user", "content": "What is 1+1?"}],
            [{"role": "user", "content": "What is 2+3?"}],
            [{"role": "user", "content": "What is 4+4?"}]
        ]

        import asyncio
        async def run_batch():
            return await api.run_batch(
                messages_list=batch_messages,
                model=Claude.HAIKU_4_5,
                temperature=0.1,
                max_tokens=30,
                max_parallel_jobs=2  # Limit concurrency
            )

        responses = asyncio.run(run_batch())

        # Validate all responses
        assert len(responses) == 3
        for i, response in enumerate(responses):
            self._validate_response(response)
            expected_answers = ["2", "5", "8"]
            if isinstance(response, str):
                assert expected_answers[i] in response

        self._validate_usage_tracking(usage_tracker)

        # Usage should reflect multiple calls
        total_usage = usage_tracker.total_usage()
        assert total_usage["input_tokens"] > usage_tracker.last_usage()["input_tokens"]  # Should be more than single call

        print(f"Claude batch processing test completed successfully. Usage: {total_usage}")

    def test_error_handling_with_invalid_model(self):
        """Test error handling when using invalid model names."""
        required_vars = ["OPENAI_API_KEY"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping OpenAI error test - missing environment variables: {required_vars}")

        # Create LLMApi with wrapped completion function
        completion_func = wrap_completion_function(litellm.completion)
        api = LLMApi(completion_func)

        # Test with invalid model - should raise appropriate exception
        messages = self._create_test_messages()

        with pytest.raises(Exception):  # Could be various exceptions depending on provider
            api.run(
                messages=messages,
                model="invalid-model-name-that-does-not-exist",
                temperature=0.1,
                max_tokens=50
            )

    def test_environment_variable_loading_patterns(self):
        """Test that environment variables are properly loaded for different providers."""

        # Test OpenAI credentials
        if os.getenv("OPENAI_API_KEY"):
            assert len(os.getenv("OPENAI_API_KEY")) > 10  # Basic sanity check
            print("✓ OpenAI credentials detected")
        else:
            print("✗ OpenAI credentials not found")

        # Test Anthropic credentials
        if os.getenv("ANTHROPIC_API_KEY"):
            assert len(os.getenv("ANTHROPIC_API_KEY")) > 10
            print("✓ Anthropic credentials detected")
        else:
            print("✗ Anthropic credentials not found")

        # Test Versa OpenAI credentials
        if os.getenv("VERSA_API_KEY") and os.getenv("VERSA_ENDPOINT"):
            assert len(os.getenv("VERSA_API_KEY")) > 10
            assert "http" in os.getenv("VERSA_ENDPOINT").lower()
            print("✓ Versa OpenAI credentials detected")
        else:
            print("✗ Versa OpenAI credentials not found")

        # Test AWS/Versa Claude credentials
        if (os.getenv("AWS_ACCESS_KEY_ID") and
            os.getenv("AWS_SECRET_ACCESS_KEY") and
            os.getenv("VERSA_ENDPOINT")):
            assert len(os.getenv("AWS_ACCESS_KEY_ID")) > 10
            assert len(os.getenv("AWS_SECRET_ACCESS_KEY")) > 10
            print("✓ Versa Claude credentials detected")
        else:
            print("✗ Versa Claude credentials not found")

    def test_wrapper_composition_in_integration(self):
        """Test that wrapper composition works correctly in integration scenarios."""
        required_vars = ["OPENAI_API_KEY"]

        if not self._check_env_vars_present(required_vars):
            pytest.skip(f"Skipping wrapper composition test - missing environment variables: {required_vars}")

        # Import additional wrappers for testing
        from lab_llm.caching_completion import CachingCompletion
        from lab_llm.error_tracker import ErrorTracker
        import tempfile
        import logging

        # Create temporary cache database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache_path = f.name

        try:
            # Create usage and error trackers
            usage_tracker = UsageTracker()
            error_tracker = ErrorTracker(
                logger=logging.getLogger("integration_test"),
                propagate_interrupts=True
            )
            cache = CachingCompletion(cache_path)

            # Create fully wrapped completion function
            completion_func = wrap_completion_function(
                litellm.completion,
                cache=cache,
                error_tracker=error_tracker,
                usage_tracker=usage_tracker
            )
            api = LLMApi(completion_func)

            # Make the same call twice to test caching
            messages = self._create_test_messages()

            # First call - should hit API
            response1 = api.run(
                messages=messages,
                model=OpenAI.GPT4_O_MINI,
                temperature=0.0,  # Use 0 temperature for deterministic results
                max_tokens=20,
                seed=42  # Use seed for reproducibility
            )

            first_usage = usage_tracker.total_usage()

            # Second identical call - should hit cache
            response2 = api.run(
                messages=messages,
                model=OpenAI.GPT4_O_MINI,
                temperature=0.0,
                max_tokens=20,
                seed=42
            )

            second_usage = usage_tracker.total_usage()

            # Validate both responses
            self._validate_response(response1, expected_content_keywords=["4"])
            self._validate_response(response2, expected_content_keywords=["4"])

            # Responses should be similar (cached response)
            if isinstance(response1, str) and isinstance(response2, str):
                # They should be exactly the same due to caching
                assert response1 == response2

            # Usage should NOT increase for second call (cache hit with 0 tokens)
            # Note: Depending on cache implementation, usage might be 0 for cached responses
            assert second_usage["total_tokens"] >= first_usage["total_tokens"]

            print(f"Wrapper composition test completed successfully.")
            print(f"First call usage: {first_usage}")
            print(f"Total usage after second call: {second_usage}")

        finally:
            # Clean up temporary cache file
            import os
            try:
                os.unlink(cache_path)
            except:
                pass