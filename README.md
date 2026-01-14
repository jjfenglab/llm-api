<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

# Feng Lab LLM API

A unified Python library for inference across multiple Large Language Model providers. Built for research workflows, this library provides a consistent interface for OpenAI, AWS Bedrock, and Azure OpenAI (Versa) models with built-in caching, batch processing, and error tracking.

## Features

- **Multi-provider support**: Unified API for OpenAI, AWS Bedrock (Claude, Llama, Cohere, Qwen), and Azure OpenAI
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
import asyncio
from dotenv import load_dotenv

from lab_llm.llm_api import LLMApi
from lab_llm.constants import OpenAi, LLMModel
from lab_llm.dataset import TextDataset
from lab_llm.llm_cache import LLMCache
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.duckdb_handler import DuckDBHandler

load_dotenv()

# Initialize components
db_handle = DuckDBHandler("./cache.db")
cache = LLMCache(db_handle)
model = LLMModel(name=OpenAi.GPT4_O_MINI)

# Create API instance
api = LLMApi(cache=cache, seed=42, model_type=model)

# Single prompt
response = api.get_output("What is the capital of France?")
print(response)

# Batch processing
dataset = TextDataset(["What is 2+2?", "What is the speed of light?"])
responses = asyncio.run(api.get_outputs(dataset))
print(responses)
```

## Configuration

Create a `.env` file in your project directory with the required credentials:

```bash
# For OpenAI models
OPENAI_ACCESS_TOKEN=your_openai_api_key

# For AWS Bedrock models (Claude, Llama, Cohere, Qwen)
BEDROCK_ACCESS_KEY=your_aws_access_key
BEDROCK_ACCESS_KEY_SECRET=your_aws_secret_key

# For Azure OpenAI / Versa models
VERSA_API_KEY=your_versa_api_key
VERSA_ENDPOINT=https://your-endpoint.openai.azure.com/openai/deployments/<model_name>/chat/completions?api-version=2024-10-21
```

Load environment variables in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Supported Models

### OpenAI (Direct API)
- `OpenAi.GPT4_O` - GPT-4o
- `OpenAi.GPT4_O_MINI` - GPT-4o Mini
- `OpenAi.GPT5` - GPT-5 (reasoning model)
- `OpenAi.GPT5_MINI` - GPT-5 Mini (reasoning model)
- `OpenAi.GPT5_NANO` - GPT-5 Nano (reasoning model)

### Azure OpenAI (Versa)
- `VersaOpenAi.GPT4_O_2024_08` - GPT-4o (August 2024)
- `VersaOpenAi.GPT4_O_MINI_2024_07` - GPT-4o Mini (July 2024)
- `VersaOpenAi.GPT5_2025_08` - GPT-5 (August 2025)
- And more...

### AWS Bedrock
- `Claude.HAIKU_3` - Claude 3 Haiku
- `Claude.HAIKU_3_5` - Claude 3.5 Haiku
- `Claude.SONNET_4_5` - Claude Sonnet 4.5
- `Meta.LLAMA_3_3_70B` - Llama 3.3 70B
- `Meta.LLAMA_3_2_11B` - Llama 3.2 11B
- `Cohere.COMMAND_R` - Command R
- `Qwen.QWEN_3_235` - Qwen 3 235B

For the complete list, see [lab_llm/constants.py](lab_llm/constants.py).

## Usage Examples

### Using Structured Output (Pydantic)

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

response = api.get_output(
    "What is 2+2?",
    response_model=Answer
)
print(response.answer, response.confidence)
```

### Using Reasoning Models

```python
from lab_llm.constants import OpenAi, LLMModel

model = LLMModel(name=OpenAi.GPT5)
api = LLMApi(
    cache=cache,
    model_type=model,
    reasoning_effort="medium",  # low, medium, high
    verbosity="concise"         # concise, detailed
)
```

### Custom System Prompt

```python
response = api.get_output(
    "Analyze this data",
    system_prompt="You are a data scientist specializing in statistical analysis."
)
```

## Error Tracking

lab_llm provides error tracking to help debug failures during research workflows.

### Quick Start

```python
from lab_llm.error_tracker import ErrorTracker
from lab_llm.error_callback_handler import ErrorCallbackHandler

# Create error tracker (logs to JSONL file)
error_tracker = ErrorTracker("study_errors.jsonl")

# Pass to error handler
error_handler = ErrorCallbackHandler(logger, error_tracker=error_tracker)

# Use with LLMApi
llm_api = LLMApi(
    cache=cache,
    error_handler=error_handler,
    # ... other params
)
```

### Analyzing Errors

```python
import pandas as pd
from lab_llm.error_tracker import ErrorTracker

tracker = ErrorTracker("study_errors.jsonl")

# Get error summary
summary = tracker.get_summary()
print(summary)

# Analyze transient errors (should retry)
transient = tracker.get_transient_errors()

# Analyze permanent errors (need fixes)
permanent = tracker.get_permanent_errors()

# Investigate specific prompt
errors = tracker.get_errors_by_prompt(prompt_hash)
```

**Error Categories:**
- **transient**: Timeouts, rate limits, network errors (will retry automatically)
- **permanent**: Validation errors, serialization errors (need prompt/code fixes)
- **user_interrupt**: Keyboard interrupts (stops execution)
- **unknown**: Unclassified errors

For a complete example, see [examples/analyze_failures.ipynb](examples/analyze_failures.ipynb).

## Development

### Running Tests

```bash
pytest tests/
```

### Testing Changes

Use `test_script.py` for manual testing:

```bash
python test_script.py
```

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
