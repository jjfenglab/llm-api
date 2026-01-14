"""
Pytest configuration for lab_llm tests.

This file is automatically loaded by pytest and configures:
- Environment variable loading from .env
- Common fixtures
- Test markers
"""

import os
import sys

import pytest
from dotenv import load_dotenv

# Add the project root to the path so we can import lab_llm
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file at the project root
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require API credentials)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
