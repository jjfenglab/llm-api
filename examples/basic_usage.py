"""
Basic usage example for lab_llm.

This script demonstrates how to:
1. Set up the LLM API with caching
2. Make single and batch LLM calls
3. Use structured output with Pydantic
4. Handle errors gracefully

Before running, ensure you have set up your .env file with the appropriate
API credentials. See README.md for details.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from lab_llm import (
    LLMApi,
    LLMCache,
    DuckDBHandler,
    ErrorCallbackHandler,
    TextDataset,
    LLMModel,
    OpenAi,
)

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_llm_api():
    """Initialize the LLM API with caching and error handling."""
    # Use environment variable for cache path, or default to ./cache.db
    cache_path = os.environ.get("LLM_CACHE_PATH", "./cache.db")

    # Initialize database handler and cache
    db_handle = DuckDBHandler(cache_path)
    cache = LLMCache(db_handle)

    # Choose your model
    model = LLMModel(name=OpenAi.GPT4_O_MINI)

    # Create error handler
    error_handler = ErrorCallbackHandler(logger)

    # Create the API instance
    api = LLMApi(
        cache=cache,
        seed=42,  # For reproducibility
        model_type=model,
        error_handler=error_handler,
        logging=logger,
        timeout=120,
    )

    return api


def single_prompt_example(api):
    """Example: Single prompt inference."""
    print("\n" + "=" * 50)
    print("Single Prompt Example")
    print("=" * 50)

    response = api.get_output("What is the capital of France?")
    print(f"Response: {response}")

    return response


def batch_processing_example(api):
    """Example: Batch processing multiple prompts."""
    print("\n" + "=" * 50)
    print("Batch Processing Example")
    print("=" * 50)

    prompts = [
        "What is 2 + 2?",
        "Name three primary colors.",
        "What is the speed of light in m/s?",
    ]

    dataset = TextDataset(prompts)

    # Process in batches
    responses = asyncio.run(api.get_outputs(dataset, batch_size=2))

    for prompt, response in zip(prompts, responses):
        print(f"\nQ: {prompt}")
        print(f"A: {response}")

    return responses


def structured_output_example(api):
    """Example: Using Pydantic models for structured output."""
    print("\n" + "=" * 50)
    print("Structured Output Example")
    print("=" * 50)

    # Define the response schema
    class MathAnswer(BaseModel):
        expression: str
        result: int
        explanation: str

    response = api.get_output(
        "What is 15 + 27? Provide the expression, result, and a brief explanation.",
        response_model=MathAnswer,
    )

    if response:
        print(f"Expression: {response.expression}")
        print(f"Result: {response.result}")
        print(f"Explanation: {response.explanation}")

    return response


def custom_system_prompt_example(api):
    """Example: Using a custom system prompt."""
    print("\n" + "=" * 50)
    print("Custom System Prompt Example")
    print("=" * 50)

    response = api.get_output(
        "Explain photosynthesis",
        system_prompt="You are a biology teacher explaining concepts to a 5th grader. Use simple language and fun analogies.",
    )

    print(f"Response: {response}")

    return response


def main():
    """Run all examples."""
    print("Lab LLM Basic Usage Examples")
    print("=" * 50)

    # Check for API key
    if not os.environ.get("OPENAI_ACCESS_TOKEN"):
        print("Warning: OPENAI_ACCESS_TOKEN not found in environment.")
        print("Please set up your .env file. See README.md for details.")
        return

    # Initialize the API
    api = setup_llm_api()

    # Run examples
    single_prompt_example(api)
    batch_processing_example(api)
    structured_output_example(api)
    custom_system_prompt_example(api)

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
