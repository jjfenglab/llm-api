"""
Caches LLM responses
"""

import hashlib
import json
import logging
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from lab_llm.constants import LLMModel
from lab_llm.duckdb_handler import DuckDBHandler


class LLMCache:
    def __init__(self, db_handler: DuckDBHandler, simplified_call_params: bool = False):
        self.db_handler = db_handler
        self.logger = logging.getLogger(__name__)
        self._create_cache()
        self.simplified_call_params = simplified_call_params

    def get_response(
        self,
        text: str,
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[bool, Optional[str]]:
        """Get a single response from cache. Implemented using get_responses for consistency."""
        try:
            # Use get_responses with single text to leverage shared logic
            df = self.get_responses(
                [text], model_type, seed, max_new_tokens, temperature
            )

            if len(df) > 0:
                found_in_cache = df.iloc[0]["found_in_cache"]
                llm_output = df.iloc[0]["llm_output"]
                return found_in_cache, llm_output
            else:
                return False, None
        except Exception as e:
            self.logger.error(f"Error retrieving cache response: {e}")
            return False, None

    def get_responses(
        self,
        texts: List[str],
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        output_type: BaseModel = None,
    ) -> pd.DataFrame:
        try:
            texts_hash = [self.compute_hash(text) for text in texts]
            text_df = pd.DataFrame({"prompt": texts, "prompt_hash": texts_hash})
            if self.simplified_call_params:
                call_params_hash = model_type.name.value
            else:
                call_params = json.dumps(
                    {
                        "model_type": model_type.name.value,
                        "seed": seed,
                        "max_new_tokens": max_new_tokens,
                        "temperature": float(temperature),  # Normalize to float
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

            result_df = self.db_handler.execute_with_retry(
                query, [call_params_hash]
            ).df()

            if result_df.empty:
                text_df["llm_output"] = None
                text_df["found_in_cache"] = False
            else:

                merged_df = text_df.merge(result_df, on="prompt_hash", how="left")

                found_hashes = set(result_df["prompt_hash"])
                merged_df["found_in_cache"] = merged_df["prompt_hash"].isin(
                    found_hashes
                )
                text_df = merged_df[["prompt", "llm_output", "found_in_cache"]]

            return text_df[["prompt", "llm_output", "found_in_cache"]]
        except Exception as e:
            self.logger.error(f"Error retrieving cache responses: {e}")
            # Return empty result with proper structure on error
            text_df = pd.DataFrame(
                {
                    "prompt": texts,
                    "llm_output": [None] * len(texts),
                    "found_in_cache": [False] * len(texts),
                }
            )
            return text_df

    def save_response(
        self,
        input_text: str,
        llm_output: str,
        model_type: LLMModel,
        seed: Optional[int],
        max_new_tokens: int,
        temperature: float,
    ):
        try:
            input_text_hash = self.compute_hash(input_text)
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": float(temperature),  # Normalize to float
                }
            )
            if self.simplified_call_params:
                call_params_hash = model_type.name.value
            else:
                call_params_hash = self.compute_hash(call_params)

            query = """
                INSERT INTO cache (prompt_hash, call_params_hash, llm_output)
                VALUES (?, ?, ?)
            """

            self.db_handler.execute_with_retry(
                query, [input_text_hash, call_params_hash, llm_output]
            )
        except Exception as e:
            self.logger.error(f"Error saving cache response: {e}")

    def save_responses(
        self,
        input_texts: str,
        llm_outputs: str,
        model_type: LLMModel,
        seed: Optional[int],
        max_new_tokens: int,
        temperature: float,
    ):
        try:
            input_texts_hash = [
                self.compute_hash(input_text) for input_text in input_texts
            ]
            call_params = json.dumps(
                {
                    "model_type": model_type.name.value,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens,
                    "temperature": float(temperature),  # Normalize to float
                }
            )
            if self.simplified_call_params:
                call_params_hash = model_type.name.value
            else:
                call_params_hash = self.compute_hash(call_params)
            insert_data = []
            for input_hash, llm_output in zip(input_texts_hash, llm_outputs):
                insert_data.append(
                    (
                        input_hash,
                        call_params_hash,
                        llm_output,
                    )
                )
            query = """
                INSERT INTO cache (prompt_hash, call_params_hash, llm_output)
                VALUES (?, ?, ?)
            """

            # Use individual saves for better reliability with retry logic
            for data in insert_data:
                self.db_handler.execute_with_retry(query, data)
        except Exception as e:
            self.logger.error(f"Error saving cache responses: {e}")

    def compute_hash(self, text: str) -> str:
        text = text.strip()
        text = text.encode("utf-8")
        return hashlib.sha256(text).hexdigest()

    def _create_cache(self):
        try:
            query = """
                CREATE TABLE IF NOT EXISTS cache (
                    prompt_hash VARCHAR,
                    call_params_hash VARCHAR,
                    llm_output TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """

            self.db_handler.execute_with_retry(query)
        except Exception as e:
            self.logger.error(f"Error creating cache table: {e}")
            raise
