<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

# Feng Lab LLM API

A unified Python library for inference across multiple Large Language Model providers. Built for research workflows, this library provides a consistent interface for OpenAI, AWS Bedrock, and Azure OpenAI (Versa) models with built-in caching, batch processing, and error tracking.

## Features

- **Multi-provider support**: Unified API for OpenAI, Anthropic, AWS Bedrock Claude, Azure OpenAI, and local OpenAI-compatible servers (vLLM, LM Studio, Ollama)
- **Response caching**: DuckDB-based caching to avoid redundant API calls and reduce costs
- **Batch processing**: Async batch inference with configurable concurrency
- **Structured output**: Pydantic model validation for enforcing response schemas
- **Error tracking**: Comprehensive error classification and JSONL logging for debugging
- **Reasoning model support**: Special handling for GPT-5 reasoning models with effort/verbosity parameters

## Installation

```bash
pip install lab-llm
```

Or install from source:

```bash
git clone https://github.com/jjfenglab/llm-api.git
cd llm-api
pip install -e .
```

## Quick Start

```python
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
    cache=CachingCompletion("./llmapi_cache.db")
))

result = api.run("What is the capital of France?", model=Claude.HAIKU_4_5)
print(f"Result: {result}\n")
# Result: The capital of France is Paris.
```

See the Python scripts in `examples/` for more examples.

## Configuration

Create a `.env` file in your project directory with the required credentials:

```bash
# For OpenAI models
OPENAI_ACCESS_TOKEN=your_openai_api_key

# For Claude models
ANTHROPIC_API_KEY=your_claude_api_key

# For AWS Bedrock models (Claude, Llama, Cohere, Qwen)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2

# For Azure OpenAI / Versa models (see "Using Azure OpenAI" example below)
VERSA_API_KEY=your_versa_api_key
VERSA_ENDPOINT=https://your-endpoint.openai.azure.com/general
# Optional: override the default API version (defaults to 2024-10-21)
# VERSA_API_VERSION=2024-10-21

# For a local OpenAI-compatible server (vLLM, LM Studio, etc.)
# Only needed if you don't pass api_base/api_key explicitly to the completion function.
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=not-needed
```

Load environment variables in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Supported Models

### OpenAI (Direct API)

- `OpenAI.GPT4_O` - GPT-4o
- `OpenAI.GPT4_O_MINI` - GPT-4o Mini
- `OpenAI.GPT5` - GPT-5 (reasoning model)
- `OpenAI.GPT5_MINI` - GPT-5 Mini (reasoning model)
- `OpenAI.GPT5_NANO` - GPT-5 Nano (reasoning model)

### Anthropic (Direct API)

- `Claude.SONNET_4` - Claude Sonnet 4
- `Claude.HAIKU_4_5` - Claude Haiku 4.5
- `Claude.SONNET_4_5` - Claude Sonnet 4.5
- `Claude.OPUS_4_5` - Claude Opus 4.5
- `Claude.HAIKU_4_6` - Claude Haiku 4.6
- `Claude.SONNET_4_6` - Claude Sonnet 4.6
- `Claude.OPUS_4_6` - Claude Opus 4.6

### Azure OpenAI (Versa)

**Note:** Azure models require the `make_versa_openai_completion` wrapper. See [Using Azure OpenAI](#using-azure-openai-versa) for setup.

- `VersaOpenAI.GPT4_O_2024_08` - GPT-4o (August 2024)
- `VersaOpenAI.GPT4_O_MINI_2024_07` - GPT-4o Mini (July 2024)
- `VersaOpenAI.GPT4_1_MINI_2025_04` - GPT-4.1 Mini (April 2025)
- `VersaOpenAI.GPT5_2025_08` - GPT-5 (August 2025)
- And more...

### AWS Bedrock

- `VersaClaude.CLAUDE_SONNET_4` - Claude Sonnet 4
- `VersaClaude.CLAUDE_OPUS_4_1` - Claude Opus 4.1
- `VersaClaude.CLAUDE_HAIKU_4_5` - Claude Haiku 4.5
- `VersaClaude.CLAUDE_OPUS_4_5` - Claude Opus 4.5
- `VersaClaude.CLAUDE_SONNET_4_6` - Claude Sonnet 4.6
- `VersaClaude.CLAUDE_OPUS_4_6` - Claude Opus 4.6

### Local OpenAI-Compatible Endpoints

Any model served by a local OpenAI-compatible server (vLLM, LM Studio, llama.cpp's OpenAI server, etc.) can be used by prefixing the model name with `openai/` and pointing litellm at the server's base URL. Ollama has its own prefix (`ollama/`). See the [Local Model](#using-a-local-model-vllm-lm-studio-ollama) example below.

For the complete list, see [lab_llm/constants.py](lab_llm/constants.py).

## Usage Examples

### Using a Local Model (vLLM, LM Studio, Ollama)

litellm routes any `openai/<model>` string through the OpenAI-compatible protocol, so a locally hosted server (vLLM, LM Studio, llama.cpp's OpenAI server, etc.) works with the same `LLMApi` you'd use for a hosted provider. Set the server's URL via `api_base` (and an `api_key` if the server requires one — most local servers ignore it, but litellm requires the parameter to be present):

```python
from lab_llm import LLMApi, wrap_completion_function, CachingCompletion
import litellm

api = LLMApi(wrap_completion_function(
    litellm.completion,
    cache=CachingCompletion("./llmapi_cache.db"),
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
))

result = api.run(
    "What is the capital of France?",
    model="openai/Qwen/Qwen3-32B",
)
```

You can also set `OPENAI_API_BASE` and `OPENAI_API_KEY` in your `.env` instead of passing them to `wrap_completion_function`. For Ollama, use the `ollama/` prefix (e.g. `model="ollama/llama3.3"`) with `OLLAMA_API_BASE`.

### Using Azure OpenAI (Versa)

Azure OpenAI models require additional configuration. Use the `make_versa_openai_completion` wrapper which reads credentials from environment variables:

```python
from lab_llm import LLMApi, wrap_completion_function, CachingCompletion
from lab_llm.versa.openai import make_versa_openai_completion
from lab_llm.constants import VersaOpenAI
import dotenv

dotenv.load_dotenv()  # Loads VERSA_API_KEY and VERSA_ENDPOINT

# Create Azure-configured completion function
versa_completion = make_versa_openai_completion()

api = LLMApi(wrap_completion_function(
    versa_completion,
    cache=CachingCompletion("./llmapi_cache.db")
))

result = api.run("What is the capital of France?", model=VersaOpenAI.GPT4_1_MINI_2025_04)
print(result)
```

**Required environment variables:**
- `VERSA_API_KEY` - Your Azure OpenAI API key
- `VERSA_ENDPOINT` - Your Azure endpoint URL (e.g., `https://your-resource.openai.azure.com/general`)
- `VERSA_API_VERSION` (optional) - API version, defaults to `2024-10-21`

You can also pass these directly to the wrapper:

```python
versa_completion = make_versa_openai_completion(
    api_key="your-key",
    endpoint="https://your-endpoint.openai.azure.com/general",
    api_version="2024-10-21"
)
```

### Structured Output (Pydantic)

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float
    reasoning: str

api = LLMApi(litellm.completion)

response = api.run(
    "What is 2+2? Provide your confidence level.",
    response_format=Answer,
    model="gpt-4o"
)
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
```

### Tool/Function Calling

```python
from lab_llm import LLMApi, wrap_completion_function
from lab_llm.constants import OpenAI
import litellm

def get_weather(city: str, country: str = "US") -> dict:
    """Get current weather for a city."""
    # Mock implementation
    return {"temperature": "72°F", "conditions": "sunny"}

api = LLMApi(litellm.completion)

result = api.run(
    "What's the weather like in San Francisco?",
    tools=[get_weather],
    model=OpenAI.GPT4_O_MINI
)
print(result)
```

### Error Handling and Tracking

```python
from lab_llm import LLMApi, wrap_completion_function
from lab_llm.error_tracker import ErrorTracker
import logging
import litellm

# Set up error tracking with JSONL logging
logger = logging.getLogger(__name__)
error_tracker = ErrorTracker(
    logger=logger,
    log_file="errors.jsonl",
    include_traceback=True
)

api = LLMApi(wrap_completion_function(
    litellm.completion,
    error_tracker=error_tracker
))

try:
    result = api.run("Hello", model="invalid-model-name")
except Exception as e:
    print(f"Error caught: {e}")
    # Error details logged to errors.jsonl with classification
```

### Model Ramps and Usage Tracking

```python
from lab_llm import LLMApi, wrap_completion_function
from lab_llm.parameter_wrappers import ModelRamp
from lab_llm.usage_tracker import UsageTracker
import litellm

# Usage tracking collects total token counts
usage_tracker = UsageTracker()
# Model ramp allows selecting models by size
model_ramp = ModelRamp([
    "gpt-4o-mini",    # xs, sm
    "gpt-4o",         # md
    "gpt-4-turbo"     # lg, xl
])

api = LLMApi(wrap_completion_function(
    litellm.completion,
    model_ramp=model_ramp,
    usage_tracker=usage_tracker
))

result = api.run("Explain quantum computing", model="lg")
print(f"Response: {result}")
print(f"Usage: {usage_tracker.total_usage()}")
```

## Development

### Running Tests

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Run unit and integration tests (requires API credentials):

```bash
pytest tests/
```

Run unit tests only:

```bash
pytest tests/ --ignore=tests/test_integration.py
```

**Note:** Integration tests require environment variables for API credentials. Copy `.env.example` to `.env` and fill in your credentials. Tests will be automatically skipped if the required environment variables are not present.

### Release Process

1. Update version in `pyproject.toml`
2. Add changes to `CHANGELOG.md`
3. Test installation: `pip install -e .`
4. Create a new release tag

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this library in your research, please cite:

```
@software{feng_lab_llm,
  author = {Feng Lab, UCSF},
  title = {Feng Lab LLM API},
  url = {https://github.com/jjfenglab/llm-api}
}
```
