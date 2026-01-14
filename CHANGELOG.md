# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
