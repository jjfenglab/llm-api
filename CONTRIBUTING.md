# Contributing to Feng Lab LLM API

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/jjfenglab/llm-api.git
   cd llm-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

## Running Tests

```bash
pytest tests/
```

For manual testing:
```bash
python test_script.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Add docstrings to public functions and classes
- Keep functions focused and concise

## Pull Request Process

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear messages:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```

3. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

4. Push your branch and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

5. In the PR description:
   - Describe what changes were made
   - Reference any related issues
   - Note any breaking changes

## Adding New Models

To add support for a new LLM:

1. Add the model enum to `lab_llm/constants.py`
2. Update the model initialization logic in `lab_llm/llm_api.py`
3. Add any required mappings (e.g., for Bedrock models)
4. Update documentation in README.md
5. Add tests if applicable

## Reporting Issues

When reporting issues, please include:

- Python version
- Package version (`pip show lab-llm`)
- Minimal code to reproduce the issue
- Full error traceback
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or discussions about the project.
