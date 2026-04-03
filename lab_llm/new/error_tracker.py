"""
ErrorTracker wrapper for litellm.completion with error handling and logging.

Combines the functionality of the original ErrorTracker and ErrorCallbackHandler
into a decorator-style wrapper that catches errors, classifies them, and handles
them by logging to console and/or saving to JSONL file.
"""

import hashlib
import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

from litellm import ModelResponse
from openai import PermissionDeniedError
from litellm.exceptions import PermissionDeniedError as LiteLLMPermissionDenied, APIError as LiteLLMAPIError

from .types import CompletionFunction, CompletionFunctionWrapper


class ErrorCategory(Enum):
    """Classification of errors for debugging and retry logic"""

    TRANSIENT = "transient"  # Timeout, rate limit, network - should retry
    PERMANENT = "permanent"  # Validation, serialization - needs prompt fix
    USER_INTERRUPT = "user_interrupt"  # Ctrl+C, SIGTERM - stop execution
    UNKNOWN = "unknown"  # Unclassified errors


class ErrorTracker(CompletionFunctionWrapper):
    """
    A wrapper that adds error handling and logging to completion functions.

    Provides:
    - Error classification (transient/permanent/user_interrupt)
    - Optional JSONL-based error logging for analysis
    - Structured logging with error categories
    - User interrupt propagation
    - Preserves original exception types and behavior

    Example usage:
        completion = ErrorTracker(
            logger=logging.getLogger(__name__),
            log_file="errors.jsonl"
        )(litellm.completion)

        response = completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )

    Can be composed with other wrappers:
        completion = ErrorTracker(logger)(
            CachingCompletion("./cache.db")(
                litellm.completion
            )
        )
    """

    # Error classification based on exception hierarchy
    TRANSIENT_BASES = (TimeoutError, ConnectionError, PermissionDeniedError, LiteLLMPermissionDenied, LiteLLMAPIError)
    PERMANENT_BASES = (ValueError, TypeError)

    def __init__(
        self,
        logger: logging.Logger,
        log_file: Optional[str] = None,
        propagate_interrupts: bool = True,
        include_traceback: bool = False,
    ):
        """
        Initialize the error tracking wrapper.

        Args:
            logger: Python logger for console/file logging
            log_file: Optional path to JSONL file for error logging
            propagate_interrupts: If True, re-raise KeyboardInterrupt (default: True)
            include_traceback: Whether to include full traceback in JSONL logs
        """
        self.logger = logger
        self.log_file = Path(log_file) if log_file else None
        self.propagate_interrupts = propagate_interrupts
        self.include_traceback = include_traceback

    def __call__(self, func: CompletionFunction) -> CompletionFunction:
        """
        Implements CompletionFunctionWrapper protocol.

        Args:
            func: The completion function to wrap

        Returns:
            Wrapped completion function with error handling
        """
        @wraps(func)
        def wrapped_completion(model: str, messages: List = None, **kwargs) -> ModelResponse:
            try:
                return func(model, messages=messages, **kwargs)
            except Exception as error:
                self._handle_error(error, model, messages, **kwargs)
                # Re-raise the original error to preserve exception handling behavior
                raise

        return wrapped_completion

    def _handle_error(self, error: Exception, model: str, messages: List, **kwargs):
        """
        Handle an error by classifying and logging it.

        Args:
            error: The exception that occurred
            model: Model name used in the request
            messages: Messages sent to the model
            **kwargs: Additional context from the completion call
        """
        # Classify the error
        category = self.classify_error(error)

        # Log to JSONL file if enabled
        if self.log_file:
            self._log_error_to_file(error, category, model, messages, **kwargs)

        # Enhanced structured logging with category
        self.logger.error(f"LLM Error [{category.value}]: {type(error).__name__}")
        self.logger.error(f"Message: {str(error)}")
        self.logger.error(f"Model: {model}")
        if kwargs:
            self.logger.error(f"Additional context: {kwargs}")

        # Propagate user interrupts to stop execution
        if self.propagate_interrupts and isinstance(error, KeyboardInterrupt):
            self.logger.warning("User interrupt detected - stopping execution")
            # Don't re-raise here, let the wrapper function handle it

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

    def _log_error_to_file(
        self,
        error: Exception,
        category: ErrorCategory,
        model: str,
        messages: List,
        **kwargs
    ):
        """
        Log an error to the JSONL file.

        Args:
            error: The exception that occurred
            category: The classified error category
            model: Model name used in the request
            messages: Messages sent to the model
            **kwargs: Additional context from the completion call
        """
        try:
            # Compute prompt hash from messages
            prompt_text = self._messages_to_string(messages)
            prompt_hash = hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()

            # Build error record
            error_record = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_category": category.value,
                "error_message": str(error),
                "prompt_hash": prompt_hash,
                "model": model,
            }

            # Add prompt preview for debugging (first 200 chars)
            if prompt_text:
                error_record["prompt_preview"] = prompt_text[:200]

            # Add relevant context fields (filter out sensitive/large data)
            context = self._filter_context_for_logging(**kwargs)
            if context:
                error_record.update(context)

            # Optionally add full traceback
            if self.include_traceback:
                error_record["traceback"] = traceback.format_exc()

            # Append to JSONL file (one JSON object per line)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(error_record) + "\n")

        except Exception as log_error:
            # Don't let logging errors crash the application
            self.logger.warning(f"Failed to log error to file: {log_error}")

    def _messages_to_string(self, messages: List) -> str:
        """
        Convert messages list to a string representation for hashing.

        Args:
            messages: List of message objects (dicts or pydantic models)

        Returns:
            String representation of messages
        """
        try:
            # Convert messages to a consistent format
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_messages.append(msg)
                else:  # Pydantic Message object
                    if hasattr(msg, 'model_dump'):
                        formatted_messages.append(msg.model_dump())
                    else:
                        # Fallback - try to convert to dict
                        formatted_messages.append(dict(msg))

            # Convert to JSON string for consistent hashing
            return json.dumps(formatted_messages, sort_keys=True, separators=(',', ':'))
        except Exception:
            # Fallback to string representation
            return str(messages)

    def _filter_context_for_logging(self, **kwargs) -> Dict:
        """
        Filter kwargs to include only relevant context for logging.

        Removes sensitive or large data that shouldn't be logged.

        Args:
            **kwargs: All keyword arguments from completion call

        Returns:
            Filtered context dict
        """
        context = {}

        # Include relevant parameters for debugging
        relevant_params = {
            'temperature', 'top_p', 'max_tokens', 'max_completion_tokens',
            'timeout', 'n', 'stream', 'response_format', 'tools', 'tool_choice'
        }

        for key, value in kwargs.items():
            if key in relevant_params:
                # Handle response_format conversion if it's a pydantic model
                if key == 'response_format' and value is not None:
                    if isinstance(value, dict):
                        context[key] = value
                    elif hasattr(value, 'model_json_schema'):
                        context[key] = value.model_json_schema()
                    else:
                        context[key] = str(value)
                else:
                    context[key] = value

        return context