# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `run_batch` now honors `max_parallel_jobs` exactly: worker threads run on a
  dedicated executor instead of the asyncio default executor, whose
  `min(32, cpu_count + 4)` thread cap silently limited true concurrency
  (`max_parallel_jobs=100` previously ran at most 32 wide, 15–16 on typical
  machines). The executor is torn down after every batch; an aborted batch
  cancels queued-but-unstarted requests.

### Changed
- **Behavior change on shared clusters:** `max_parallel_jobs=None` now
  defaults to the CPU count available to the process (`os.process_cpu_count()`
  on Python 3.13+, falling back to `os.cpu_count()`), not the whole node's
  core count. A 2-CPU slurm allocation now defaults to 2 instead of 64 — set
  `max_parallel_jobs` explicitly when you want throughput.
- `run_batch` raises `ValueError` when `max_parallel_jobs` is outside
  [1, 512], instead of silently accepting values that would spawn thousands
  of threads.
- `CachingCompletion` cache keys ignore `num_retries` (transport-only
  parameter), so enabling retries keeps existing cache hits.

### Added
- Rate-limit guidance in the `run_batch` docstring: lab_llm does no retries
  itself; pass `num_retries=N` through to litellm for exponential backoff on
  429s, and size concurrency as ≈ (requests/min limit) / 60 × mean request
  latency (per-deployment Versa limits are in the lab wiki).

## [1.0.2] - 2026-08-29

### Added
- Docstrings for `LLMApi.run`/`run_batch`; default concurrency for
  `run_batch`; boto3 dependency (#21)
- `return_exceptions` option for `run_batch` (per-item failures returned in
  place, order preserved) and `strict_response_format` for `run` (#22)

### Fixed
- `CachingCompletion` uses per-operation cursors on one shared DuckDB
  connection instead of opening separate connections (#21)

### Changed
- README updates (#23); `requirements.txt` refreshed to current pinned
  versions, including duckdb 0.7.0 → 1.5.5 (#24)

## [1.0.1] - 2026-06-29

### Added
- Local OpenAI-compatible endpoint support (e.g. ollama) via litellm, with
  documentation (#20)

### Changed
- Minimum Python set to 3.10 with typing back-compat fixes; new model IDs
  added and Claude Sonnet 4 deprecated (#19)

## [1.0.0] - 2026-05-20

### Changed
- Complete refactor to composable completion-function wrappers:
  `wrap_completion_function` plus `CachingCompletion`, `UsageTracker`,
  `ErrorTracker`, `ModelRamp`, `DefaultParameters`; Versa factories renamed
  to `make_versa_*` (#16)

### Added
- Tool use in `LLMApi.run`: tool-call execution loop with `max_tool_calls`,
  tools built from plain Python functions via introspection (#16)
- Opt-in token usage tracking for LLM queries, with cumulative totals and
  `reset()` (#15, #16)
- PHI-safe Claude routing support (#13)
- pytest-based integration tests and public-release preparation (#12)

## [0.1.5] - 2025-01-14

### Added
- Batch logging information for token usage tracking
- Prevention of caching None/failed responses (#11)
- Reasoning model support with `reasoning_effort` and `verbosity` parameters (#10)

### Fixed
- Only cache successful LLM responses, not failed ones

## [0.1.4] - 2024-12

### Added
- Updated LLM model versions (#9)
- GPT-5 model family support (GPT-5, GPT-5 Mini, GPT-5 Nano)

## [0.1.3] - 2024-11

### Added
- Image support for multimodal inference (#8)
- Base64 image encoding in cache

## [0.1.2] - 2024-10

### Added
- Token counting utility using tiktoken (#3)
- Return exceptions option for batch processing (#2)

### Fixed
- Handle None values from cache properly (#5)
- Catch LangChain validation errors (#4)

## [0.1.1] - 2024-09

### Added
- Error tracking system with JSONL logging
- Error classification (transient, permanent, user_interrupt, unknown)
- ErrorTracker class for analyzing failures
- ErrorCallbackHandler for LangChain integration

## [0.1.0] - 2024-08

### Added
- Initial release
- Multi-provider LLM support (OpenAI, AWS Bedrock, Azure OpenAI)
- DuckDB-based response caching
- Async batch processing with DataLoader
- Structured output with Pydantic validation
- Support for Claude, Llama, Cohere, and GPT models
