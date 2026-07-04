"""
Integration test for a locally running Ollama server via litellm.

To run this test:

1. Install Ollama (https://ollama.com/download). On macOS: `brew install ollama`.

2. Start the Ollama server. On macOS, either launch the Ollama app or run:
       ollama serve
   The server listens on http://localhost:11434 by default. Leave it running
   in a separate terminal (or as a background service).

3. Pull a small model to test against:
       ollama pull llama3.2:1b

4. Set the env vars pytest looks for:
       export OLLAMA_API_BASE=http://localhost:11434
       export OLLAMA_TEST_MODEL=llama3.2:1b

5. Run the test:
       pytest tests/test_local_ollama.py -v

The test is skipped (not failed) if either env var is unset or if the server
at OLLAMA_API_BASE is unreachable, so it is safe to keep in the normal
test collection.
"""

import os
import urllib.error
import urllib.request

import pytest
from pydantic import BaseModel

import dotenv
dotenv.load_dotenv()

import litellm
from lab_llm import LLMApi, wrap_completion_function
from lab_llm.usage_tracker import UsageTracker


def _ollama_reachable(api_base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{api_base.rstrip('/')}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


@pytest.fixture(scope="module")
def ollama_config():
    api_base = os.getenv("OLLAMA_API_BASE")
    model = os.getenv("OLLAMA_TEST_MODEL")
    if not api_base or not model:
        pytest.skip("Set OLLAMA_API_BASE and OLLAMA_TEST_MODEL to run Ollama integration tests")
    if not _ollama_reachable(api_base):
        pytest.skip(f"Ollama server at {api_base} is not reachable")
    return {"api_base": api_base, "model": f"ollama_chat/{model}"}


def test_ollama_basic_completion(ollama_config):
    """A simple prompt round-trips through the local Ollama server."""
    usage_tracker = UsageTracker()
    api = LLMApi(wrap_completion_function(
        litellm.completion,
        usage_tracker=usage_tracker,
        api_base=ollama_config["api_base"],
    ))

    response = api.run(
        "What is 2+2? Respond with just the number.",
        model=ollama_config["model"],
        temperature=0.0,
        max_tokens=20,
    )
    print("HERE IS ollama's answer", response)

    assert isinstance(response, str)
    assert "4" in response

    last = usage_tracker.last_usage()
    assert last is not None
    assert last["input_tokens"] > 0
    assert last["output_tokens"] > 0


def test_ollama_structured_output(ollama_config):
    """Pydantic response_format works via litellm's Ollama route."""
    class Answer(BaseModel):
        answer: int
        explanation: str

    api = LLMApi(wrap_completion_function(
        litellm.completion,
        api_base=ollama_config["api_base"],
    ))

    result = api.run(
        "What is 2+2? Reply with the numeric answer and a one-sentence explanation.",
        model=ollama_config["model"],
        temperature=0.0,
        response_format=Answer,
    )
    print("HERE IS ollama's answer", result)

    assert isinstance(result, Answer), f"Expected Answer, got {type(result).__name__}: {result!r}"
    assert result.answer == 4
