"""
Example usage of the LLMApi class.

This demonstrates:
1. Simple string message handling
2. Automatic tool execution with max_tool_calls
3. Structured output with Pydantic models
4. Batch processing with run_batch()
"""

import asyncio
from pydantic import BaseModel, Field

from lab_llm import LLMApi, wrap_completion_function, CachingCompletion, ErrorTracker, UsageTracker, ModelRamp
from lab_llm.constants import Claude
import litellm

def weather_lookup(city: str, state: str = "", country: str = "US") -> dict:
    """Look up weather information for a city."""
    print(f"Looking up weather for {city}, {state}, {country}")

    if city.lower() == 'seattle':
        return {
            "temperature": "47°F (8°C)",
            "conditions": "Rainy",
            "humidity": "85%",
            "wind": "10 mph SW"
        }
    elif city.lower() == 'miami':
        return {
            "temperature": "82°F (28°C)",
            "conditions": "Sunny",
            "humidity": "65%",
            "wind": "5 mph E"
        }
    else:
        return {
            "temperature": "Unknown",
            "conditions": "Data not available",
            "humidity": "Unknown",
            "wind": "Unknown"
        }


def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression. Input should be a Python math expression."""
    # Simple safe evaluation for basic math
    result = eval(expression, {}, {})
    return float(result)


class WeatherReport(BaseModel):
    city: str = Field(description="The city name")
    summary: str = Field(description="Brief weather summary")
    recommendation: str = Field(description="What to wear or do")


def main():
    # Set up the completion function with decorators
    usage_tracker = UsageTracker()
    error_tracker = ErrorTracker(
        logger=logging.getLogger(__name__)
    )
    cache = CachingCompletion("./llmapi_cache.db")

    # Create the LLMApi instance
    api = LLMApi(wrap_completion_function(
        litellm.completion,
        cache=cache,
        model_ramp=ModelRamp([Claude.HAIKU_4_5, Claude.SONNET_4_5, Claude.OPUS_4_5], default_size="sm"),
        error_tracker=error_tracker,
        usage_tracker=usage_tracker
    ))

    print("=== Example 1: Simple string message ===")
    result = api.run(
        "What is the capital of France?"
    )
    print(f"Result: {result}\n")

    print("=== Example 2: Tool usage with weather lookup ===")
    result = api.run(
        "What's the weather like in Seattle? Give me a brief summary.",
        tools=[weather_lookup],
        max_tool_calls=3
    )
    print(f"Result: {result}\n")

    print("=== Example 3: Structured output with Pydantic ===")
    result = api.run(
        "Get the weather for Miami and format it nicely",
        tools=[weather_lookup],
        response_format=WeatherReport,
        max_tool_calls=3
    )
    print(f"Structured result: {result}")
    if isinstance(result, WeatherReport):
        print(f"  City: {result.city}")
        print(f"  Summary: {result.summary}")
        print(f"  Recommendation: {result.recommendation}")
    print()

    print("=== Example 4: Multiple tools ===")
    result = api.run(
        "What's 15 * 23 + 47? Then tell me about Seattle weather.",
        tools=[calculate, weather_lookup],
        max_tool_calls=5,
        model="xs"
    )
    print(f"Result: {result}\n")

    print("=== Example 5: Batch processing ===")
    messages_batch = [
        "What is 2 + 2?",
        "What's the weather in Seattle?",
        "Calculate 50 / 2 and tell me the result"
    ]

    async def run_batch_example():
        results = await api.run_batch(
            messages_batch,
            tools=[calculate, weather_lookup],
            max_tool_calls=3,
            max_parallel_jobs=2,  # Limit to 2 concurrent requests
            model="xs"
        )

        for i, (msg, result) in enumerate(zip(messages_batch, results)):
            print(f"Batch {i+1}: {msg}")
            print(f"Result: {result}\n")

    asyncio.run(run_batch_example())

    print("=== Usage Summary ===")
    print(f"Total usage: {usage_tracker.total_usage()}")


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    main()