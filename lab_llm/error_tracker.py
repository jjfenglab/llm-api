"""
Error tracking system for LLM API failures.
"""

import hashlib
import json
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from openai import PermissionDeniedError


class ErrorCategory(Enum):
    """Classification of errors for debugging and retry logic"""

    TRANSIENT = "transient"  # Timeout, rate limit, network - should retry
    PERMANENT = "permanent"  # Validation, serialization - needs prompt fix
    USER_INTERRUPT = "user_interrupt"  # Ctrl+C, SIGTERM - stop execution
    UNKNOWN = "unknown"  # Unclassified errors


class ErrorTracker:
    """
    Error tracking where each error is appended as a JSON line to a file

    Usage:
        tracker = ErrorTracker("study_errors.jsonl")
        tracker.log_error(
            error=TimeoutError("Request timed out"),
            prompt_hash="abc123...",
            context={"model": "gpt-4", "timeout": 60}
        )
    """

    # Error classification based on exception hierarchy
    TRANSIENT_BASES = (TimeoutError, ConnectionError, PermissionDeniedError)
    PERMANENT_BASES = (ValueError, TypeError)

    def __init__(self, log_file: str = "llm_errors.jsonl"):
        """
        Initialize error tracker with JSONL log file.

        Args:
            log_file: Path to JSONL file for error logging
        """
        self.log_file = Path(log_file)

    def classify_error(self, error: Exception) -> ErrorCategory:
        """
        Classify error into category for retry logic and debugging.

        Uses exception hierarchy instead of hardcoded type names
        to be more maintainable as libraries update their exceptions.

        Args:
            error: The exception to classify

        Returns:
            ErrorCategory enum value
        """
        # Check for user interrupts first
        if isinstance(error, KeyboardInterrupt):
            return ErrorCategory.USER_INTERRUPT

        # Check if error has retry attribute
        if hasattr(error, "should_retry") and error.should_retry:
            return ErrorCategory.TRANSIENT

        # Classify by exception hierarchy
        if isinstance(error, self.TRANSIENT_BASES):
            return ErrorCategory.TRANSIENT
        elif isinstance(error, self.PERMANENT_BASES):
            return ErrorCategory.PERMANENT

        # Check error type name for common patterns
        error_name = type(error).__name__
        if any(
            keyword in error_name.lower()
            for keyword in ["timeout", "ratelimit", "connection"]
        ):
            return ErrorCategory.TRANSIENT
        elif any(
            keyword in error_name.lower()
            for keyword in ["validation", "serialization", "auth"]
        ):
            return ErrorCategory.PERMANENT

        # Default to unknown (won't retry by default)
        return ErrorCategory.UNKNOWN

    def log_error(
        self,
        error: Exception,
        prompt: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        context: Optional[Dict] = None,
        include_traceback: bool = False,
    ):
        """
        Log an error to the JSONL file.

        Args:
            error: The exception that occurred
            prompt: The full prompt text (will be hashed and truncated)
            prompt_hash: Pre-computed prompt hash (optional)
            context: Additional context (model, timeout, etc.)
            include_traceback: Whether to include full traceback (default: False for size)
        """
        # Compute prompt hash if not provided
        if prompt_hash is None and prompt is not None:
            prompt_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

        # Classify the error
        category = self.classify_error(error)

        # Build error record
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_category": category.value,
            "error_message": str(error),
            "prompt_hash": prompt_hash,
        }

        # Add prompt preview for debugging (first 200 chars)
        if prompt is not None:
            error_record["prompt_preview"] = prompt[:200]

        # Add context fields
        if context:
            error_record.update(context)

        # Optionally add full traceback
        if include_traceback:
            error_record["traceback"] = traceback.format_exc()

        # Append to JSONL file (one JSON object per line)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(error_record) + "\n")

    def load_errors(self) -> pd.DataFrame:
        """
        Load all errors from JSONL file into a pandas DataFrame.

        Returns:
            DataFrame with all error records
        """
        if not self.log_file.exists():
            return pd.DataFrame()

        # Read JSONL file
        records = []
        with open(self.log_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    records.append(json.loads(line))

        return pd.DataFrame(records)

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of errors.

        Returns:
            DataFrame with error counts by category and type
        """
        df = self.load_errors()
        if df.empty:
            return pd.DataFrame()

        summary = (
            df.groupby(["error_category", "error_type"])
            .agg({"prompt_hash": "count", "timestamp": ["min", "max"]})
            .reset_index()
        )

        summary.columns = ["category", "type", "count", "first_seen", "last_seen"]
        return summary.sort_values("count", ascending=False)

    def get_transient_errors(self) -> pd.DataFrame:
        """
        Get all transient errors (timeouts, rate limits, etc.).

        These are errors that should be retried.

        Returns:
            DataFrame with transient errors only
        """
        df = self.load_errors()
        if df.empty:
            return df

        return df[df["error_category"] == ErrorCategory.TRANSIENT.value]

    def get_permanent_errors(self) -> pd.DataFrame:
        """
        Get all permanent errors (validation, serialization, etc.).

        These are errors that need prompt or code fixes.

        Returns:
            DataFrame with permanent errors and occurrence counts
        """
        df = self.load_errors()
        if df.empty:
            return df

        permanent = df[df["error_category"] == ErrorCategory.PERMANENT.value]

        # Group by prompt hash to see which prompts consistently fail
        if not permanent.empty:
            grouped = (
                permanent.groupby(["prompt_hash", "error_type", "error_message"])
                .agg({"timestamp": "count", "prompt_preview": "first"})
                .reset_index()
            )
            grouped.columns = [
                "prompt_hash",
                "error_type",
                "error_message",
                "count",
                "prompt_preview",
            ]
            return grouped.sort_values("count", ascending=False)

        return permanent

    def get_errors_by_prompt(self, prompt_hash: str) -> pd.DataFrame:
        """
        Get all errors for a specific prompt (for debugging).

        Args:
            prompt_hash: The SHA256 hash of the prompt

        Returns:
            DataFrame with all errors for this prompt
        """
        df = self.load_errors()
        if df.empty:
            return df

        return df[df["prompt_hash"] == prompt_hash].sort_values("timestamp")

    def clear(self):
        """Clear the error log (use with caution!)"""
        if self.log_file.exists():
            self.log_file.unlink()
            self.log_file.unlink()
