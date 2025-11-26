"""
Caches LLM responses
"""

import hashlib
import json
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from lab_llm.constants import LLMModel
from lab_llm.duckdb_handler import DuckDBHandler


class LLMCache:
    def __init__(self, db_handler: DuckDBHandler):
        self.db_handler = db_handler
        self._create_cache()

    def get_response(
        self,
        text: str,
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        db = self.db_handler.get_connection()
        text_hash = self.compute_hash(text)
        # Cache key includes all parameters that affect LLM output semantics
        # Excluded: timeout, requests_per_second, error_handler, logging (operational only)
        if reasoning_effort is not None and verbosity is not None:

            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                }
            )
        else:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
        call_params_hash = self.compute_hash(call_params)
        query = """
            SELECT 
                llm_output
            FROM cache 
            WHERE 
                prompt_hash = ? AND 
                call_params_hash = ?
            """
        result = db.execute(query, [text_hash, call_params_hash]).fetchone()

        if result:
            return True, result[-1]
        return False, None

    def save_response(
        self,
        input_text: str,
        llm_output: str,
        model_type: LLMModel,
        seed: Optional[int],
        max_new_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ):
        db = self.db_handler.get_connection()
        input_text_hash = self.compute_hash(input_text)
        # Cache key includes all parameters that affect LLM output semantics
        # Excluded: timeout, requests_per_second, error_handler, logging (operational only)
        # Only include reasoning_effort/verbosity if both are set (for backward compatibility with old cache)
        if reasoning_effort is not None and verbosity is not None:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                }
            )
        else:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
        call_params_hash = self.compute_hash(call_params)

        query = """
            INSERT INTO cache (prompt_hash, call_params_hash, llm_output)
            VALUES (?, ?, ?)
        """
        db.execute(query, [input_text_hash, call_params_hash, llm_output])

    def save_responses(
        self,
        input_texts: str,
        llm_outputs: str,
        model_type: LLMModel,
        seed: Optional[int],
        max_new_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ):
        db = self.db_handler.get_connection()
        input_texts_hash = [self.compute_hash(input_text) for input_text in input_texts]
        # Cache key includes all parameters that affect LLM output semantics
        # Excluded: timeout, requests_per_second, error_handler, logging (operational only)
        # Only include reasoning_effort/verbosity if both are set (for backward compatibility with old cache)
        if reasoning_effort is not None and verbosity is not None:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                }
            )
        else:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
        call_params_hash = self.compute_hash(call_params)
        insert_data = []
        for input_hash, llm_output in zip(input_texts_hash, llm_outputs):
            insert_data.append((input_hash, call_params_hash, llm_output))
        query = """
            INSERT INTO cache (prompt_hash, call_params_hash, llm_output)
            VALUES (?, ?, ?)
        """
        db.executemany(query, insert_data)

    def get_responses(
        self,
        texts: [str],
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        output_type: BaseModel = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> pd.DataFrame:
        db = self.db_handler.get_connection()
        texts_hash = [self.compute_hash(text) for text in texts]
        text_df = pd.DataFrame({"prompt": texts, "prompt_hash": texts_hash})

        # Cache key includes all parameters that affect LLM output semantics
        # Excluded: timeout, requests_per_second, error_handler, logging (operational only)
        # Only include reasoning_effort/verbosity if both are set (for backward compatibility with old cache)
        if reasoning_effort is not None and verbosity is not None:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                    "verbosity": verbosity,
                }
            )
        else:
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
        call_params_hash = self.compute_hash(call_params)
        texts_hash_string = ",".join(["'%s'" % text for text in texts_hash])

        query = f"""
            WITH ranked_outputs AS (
                SELECT
                    prompt_hash,
                    llm_output,
                    row_number() OVER (PARTITION BY prompt_hash ORDER BY created_at DESC) AS rn
                FROM cache
                WHERE
                    call_params_hash = ?
                    AND
                    prompt_hash IN ({texts_hash_string})
            )
            SELECT 
                prompt_hash,
                llm_output
            FROM ranked_outputs 
            WHERE rn = 1
            """
        # TODO: we probably want to grab the most recent line in case there are multiple entries with the same has, though
        # that also seems somewhat unlikely...?
        result_df = db.execute(query, [call_params_hash]).df()

        if result_df.empty:
            text_df["llm_output"] = None
            text_df["found_in_cache"] = False
        else:
            print("CACHE HIT")

            merged_df = text_df.merge(result_df, on="prompt_hash", how="left")

            found_hashes = set(result_df["prompt_hash"])
            merged_df["found_in_cache"] = merged_df["prompt_hash"].isin(found_hashes)
            text_df = merged_df[["prompt", "llm_output", "found_in_cache"]]

        return text_df[["prompt", "llm_output", "found_in_cache"]]

    def compute_hash(self, text: str) -> str:
        text = text.strip()
        text = text.encode("utf-8")
        return hashlib.sha256(text).hexdigest()

    def _create_cache(self):
        db = self.db_handler.get_connection()
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                prompt_hash VARCHAR,
                call_params_hash VARCHAR,
                llm_output TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
