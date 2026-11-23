"""
CachingCompletion wrapper for litellm.completion with SQLAlchemy-backed caching.

Implements the CompletionFunctionWrapper protocol to provide transparent caching
of LLM completion calls. Works as a decorator around any CompletionFunction.
Supports any SQLAlchemy-compatible database (DuckDB, PostgreSQL, SQLite, MySQL, etc.).
"""

import os
import json
import datetime
import hashlib
import logging
from functools import wraps
from typing import Optional, List, Any
from pathlib import Path

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Index
from sqlalchemy import select, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from litellm import ModelResponse

from .types import CompletionFunction, CompletionFunctionWrapper
from .parameter_wrappers import MODEL_SIZES


logger = logging.getLogger(__name__)


class CachingCompletion(CompletionFunctionWrapper):
    """
    A wrapper that adds caching to completion functions using SQLAlchemy storage.

    Supports any SQLAlchemy-compatible database backend. Plain filepaths are
    automatically converted to DuckDB URLs for backwards compatibility.

    Example usage with a plain filepath (DuckDB, backwards compatible):
        completion = CachingCompletion("./cache.db")(litellm.completion)
        response = completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )

    Example usage with a SQLAlchemy connection string:
        completion = CachingCompletion("postgresql://user:pass@host/dbname")(litellm.completion)
        completion = CachingCompletion("sqlite:///./cache.db")(litellm.completion)

    Can be composed with other wrappers. Note that this should typically be
    added directly around the original completion function, to ensure that
    the parameters added by other wrappers are saved.

        completion = ModelRamp(["gpt-4o-mini", "gpt-5"])(
            CachingCompletion("./cache.db")(
                make_versa_openai_completion()
            )
        )
    """

    _metadata = MetaData()
    _cache_table = Table(
        'completion_cache', _metadata,
        Column('id', Integer, primary_key=True, autoincrement=True),
        Column('cache_key', String, nullable=False, unique=True),
        Column('model', String, nullable=False),
        Column('response_json', Text, nullable=False),
        Column('created_at', DateTime, default=func.now()),
    )
    _index = Index('idx_cache_key', _cache_table.c.cache_key)

    def __init__(self, db_path: str | Path, return_original_usage: bool = False):
        """
        Initialize the caching wrapper.

        Args:
            db_path: Path to the database. Accepts either:
                - A plain filepath (str or pathlib.Path), e.g. ``"./cache.db"`` or
                  ``Path("/data/cache.db")``. The file is opened as a DuckDB database
                  for backwards compatibility with existing cache files.
                - A SQLAlchemy connection string, e.g.
                  ``"duckdb:///./cache.db"``,
                  ``"postgresql://user:pass@localhost/dbname"``,
                  ``"sqlite:///./cache.db"``, or
                  ``"mysql+pymysql://user:pass@localhost/dbname"``.
                  Any URL containing ``://`` is treated as a SQLAlchemy URL and
                  passed directly to ``create_engine()``.

                The required table (``completion_cache``) is created automatically
                if it does not already exist.
            return_original_usage: Whether or not to return original token usage counts
                in responses that are loaded from cache (default is to set the token
                usage to zero, reflecting the actual cost of the query).
        """
        self.return_original_usage = return_original_usage

        if self._is_connection_string(db_path):
            url = str(db_path)
        else:
            # Plain filepath — ensure parent directory exists, then build DuckDB URL
            self._ensure_db_dir_exists(db_path)
            url = self._filepath_to_url(db_path)

        self.engine = create_engine(url)
        # Always create tables if they don't exist (idempotent)
        self._metadata.create_all(self.engine, checkfirst=True)

    def __del__(self):
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()

    def __call__(self, func: CompletionFunction) -> CompletionFunction:
        """
        Implements CompletionFunctionWrapper protocol.

        Args:
            func: The completion function to wrap

        Returns:
            Wrapped completion function with caching
        """
        @wraps(func)
        def wrapped_completion(model: str, messages: List = None, **kwargs) -> ModelResponse:
            if model is None:
                raise ValueError(f"model cannot be None for CachingCompletion. Ensure that if you are routing models with a decorator, this happens within the completion function passed to CachingCompletion.")

            # Check if we should cache this request
            if not self._should_cache(**kwargs):
                return func(model, messages=messages, **kwargs)

            # Compute cache key
            try:
                cache_key = self._compute_cache_key(model, messages, kwargs)
            except Exception as e:
                logger.warning(f"Failed to compute cache key: {e}")
                return func(model, messages=messages, **kwargs)

            # Try to get from cache
            try:
                cached_response = self._get_cached_response(cache_key)
                if cached_response is not None:
                    logger.debug(f"Cache hit for model {model}, key: {cache_key[:12]}...")
                    return cached_response
            except Exception as e:
                logger.warning(f"Failed to get cached response: {e}")

            # Cache miss - call the wrapped function
            logger.debug(f"Cache miss for model {model}, key: {cache_key[:12]}...")
            response = func(model, messages=messages, **kwargs)

            # Cache the response if successful
            if response:
                try:
                    self._cache_response(cache_key, model, response)
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")

            return response

        return wrapped_completion

    def _is_connection_string(self, db_path: str | Path) -> bool:
        """Return True if db_path looks like a SQLAlchemy URL (contains '://')."""
        return '://' in str(db_path)

    def _filepath_to_url(self, db_path: str | Path) -> str:
        """Convert a plain filepath to a DuckDB SQLAlchemy URL."""
        path = Path(db_path).resolve()
        return f"duckdb:///{path}"

    def _ensure_db_dir_exists(self, db_path: str | Path):
        """Ensure the parent directory of a file-based database path exists."""
        db_dir = os.path.dirname(Path(db_path))
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

    def _should_cache(self, **kwargs) -> bool:
        """
        Determine if a request should be cached.

        Returns False for:
        - Streaming requests (stream=True)
        - Multiple completions (n > 1)
        """
        # Skip caching for streaming requests
        if kwargs.get('stream', False):
            return False

        # Skip caching for multiple completions
        if kwargs.get('n', 1) > 1:
            return False

        return True

    def _convert_messages_to_dicts(self, messages: List) -> List[dict]:
        """
        Convert pydantic Message objects to dicts for serialization.

        Args:
            messages: List of message objects (dicts or pydantic models)

        Returns:
            List of message dicts
        """
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
            else:  # Pydantic Message object
                if hasattr(msg, 'model_dump'):
                    result.append(msg.model_dump())
                else:
                    # Fallback - try to convert to dict
                    result.append(dict(msg))
        return result

    def _filter_kwargs_for_cache(self, **kwargs) -> dict:
        """
        Filter kwargs to include only cache-relevant parameters.

        Removes:
        - Any parameter starting with "api_"
        - Transport-only parameters that don't affect the response
          (num_retries), so enabling retries doesn't fork the key space

        Converts:
        - response_format from pydantic model to dict if necessary

        Args:
            **kwargs: All keyword arguments

        Returns:
            Filtered kwargs dict
        """
        filtered = {}

        for key, value in kwargs.items():
            # Skip any parameter starting with "api_"
            if key.startswith('api_') or key == 'num_retries':
                continue

            # Handle response_format conversion
            if key == 'response_format' and value is not None:
                if isinstance(value, dict):
                    filtered[key] = value
                else:
                    # Pydantic model
                    filtered[key] = value.model_json_schema()

            else:
                filtered[key] = value

        return filtered

    def _compute_cache_key(self, model: str, messages: List, kwargs: dict) -> str:
        """
        Compute deterministic cache key from request parameters.

        Args:
            model: Model name
            messages: Message list
            kwargs: Keyword arguments

        Returns:
            SHA-256 hash as hex string
        """
        # Convert messages to dicts
        messages_dicts = self._convert_messages_to_dicts(messages)

        # Filter kwargs for cache key
        filtered_kwargs = self._filter_kwargs_for_cache(**kwargs)

        # Build cache key dict
        cache_key_dict = {
            'model': model,
            'messages': messages_dicts,
            **filtered_kwargs
        }

        # Convert to deterministic JSON string
        cache_key_json = json.dumps(cache_key_dict, sort_keys=True, separators=(',', ':'))

        # Compute SHA-256 hash
        return hashlib.sha256(cache_key_json.encode('utf-8')).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[ModelResponse]:
        """
        Retrieve cached response by cache key.

        Args:
            cache_key: Cache key to look up

        Returns:
            ModelResponse if found, None otherwise
        """
        stmt = select(self._cache_table.c.response_json).where(
            self._cache_table.c.cache_key == cache_key
        )
        with self.engine.connect() as conn:
            result = conn.execute(stmt).fetchone()

        if result is None:
            return None

        return self._build_model_response_from_cache(result[0])

    def _cache_response(self, cache_key: str, model: str, response: ModelResponse):
        """
        Store response in cache.

        Args:
            cache_key: Cache key for this response
            model: Model name used
            response: ModelResponse to cache
        """
        response_json = self._serialize_model_response(response)
        stmt = insert(self._cache_table).values(
            cache_key=cache_key,
            model=model,
            response_json=response_json,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
        except IntegrityError:
            pass  # Duplicate key — expected; mirrors original ON CONFLICT DO NOTHING

    def _serialize_model_response(self, response: ModelResponse) -> str:
        """
        Serialize ModelResponse to JSON string.

        Args:
            response: ModelResponse to serialize

        Returns:
            JSON string representation
        """
        # Convert ModelResponse to dict
        if hasattr(response, 'model_dump'):
            response_dict = response.model_dump()
        else:
            # Fallback for other response types
            response_dict = dict(response)

        return json.dumps(response_dict, separators=(',', ':'))

    def _build_model_response_from_cache(self, cached_json: str) -> ModelResponse:
        """
        Reconstruct ModelResponse from cached JSON.

        Args:
            cached_json: JSON string from cache

        Returns:
            Reconstructed ModelResponse
        """
        response_dict = json.loads(cached_json)
        response_dict["_cache_hit"] = True

        if not self.return_original_usage:
            # Set the usage statistics to zero
            def to_zero(d: Any) -> dict:
                if isinstance(d, dict):
                    return {k: 0 if isinstance(v, int) else to_zero(v) for k, v in d.items()}
                return d
            response_dict["usage"] = to_zero(response_dict["usage"])

        return ModelResponse(**response_dict)