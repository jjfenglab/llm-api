"""
Basic usage of the LLMApi class to get started using a public LLM API.
"""

from lab_llm import LLMApi, wrap_completion_function, CachingCompletion
from lab_llm.constants import Claude
import litellm
import dotenv

# Assume we have ANTHROPIC_API_KEY in the .env file
dotenv.load_dotenv()

# Create the LLMApi instance by specifying a completion function,
# which we wrap with a DuckDB-based cache to store previous outputs
api = LLMApi(wrap_completion_function(
    litellm.completion,
    cache=CachingCompletion("./llmapi_cache.db"),
    model=Claude.HAIKU_4_5
))

# Example 1: Single message
result = api.run("What is the capital of France?")
print(f"Result: {result}\n")
# Result: The capital of France is Paris.

# Example 2: Multiple messages
result = api.run([
  {"role": "system", "content": "Always speak French to the user."},
  {"role": "user", "content": "What is the capital of France?"},
])
print(f"Result: {result}\n")
# Result: La capitale de la France est Paris.

# Example 3: Structured output
from pydantic import BaseModel, Field

class ResponseStyle(BaseModel):
    nouns: list[str] = Field(description="All nouns in the input sentence")
    verbs: list[str] = Field(description="All verbs in the input sentence")

result: ResponseStyle = api.run("The quick brown fox jumped over the lazy dog", 
                                response_format=ResponseStyle)
print(f"Result: nouns = {result.nouns}, verbs = {result.verbs}\n")
# Result: nouns = ['fox', 'dog'], verbs = ['jumped']