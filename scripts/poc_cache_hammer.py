"""Hammer CachingCompletion from run_batch at high thread concurrency.

Three legs against one temp DuckDB file, all at max_parallel_jobs=100:
1. cold: 400 unique prompts  -> concurrent miss-lookup + insert from 100 threads
2. warm: same 400 prompts    -> concurrent cache reads; the fake LLM must see 0 calls
3. same-key: 200 identical prompts -> all threads race to insert one cache_key

Reports fake-LLM call counts, cache warning counts, and final row counts.

Usage: python scripts/poc_cache_hammer.py [--jobs 100]
"""

import argparse
import asyncio
import logging
import tempfile
import threading
import time
from pathlib import Path

import duckdb
from litellm import ModelResponse
from litellm.types.utils import Usage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from lab_llm.api import LLMApi
from lab_llm.caching_completion import CachingCompletion


def make_response(model: str, content: str) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-poc",
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content=content), finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class FakeLLM:
    """Fake completion function counting calls; brief sleep widens race windows."""

    def __init__(self, latency: float = 0.02):
        self.latency = latency
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, model, messages=None, **kwargs) -> ModelResponse:
        with self._lock:
            self.calls += 1
        time.sleep(self.latency)
        return make_response(model, f"echo: {messages[-1]['content']}")


class WarningCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def run_leg(name, api, prompts, jobs, fake, counter):
    before_calls, before_warns = fake.calls, len(counter.records)
    t0 = time.perf_counter()
    results = asyncio.run(api.run_batch(prompts, max_parallel_jobs=jobs, model="fake-model"))
    wall = time.perf_counter() - t0
    assert len(results) == len(prompts) and all(isinstance(r, str) for r in results)
    print(f"{name:>9}: n={len(prompts)}  llm_calls={fake.calls - before_calls}"
          f"  cache_warnings={len(counter.records) - before_warns}  wall={wall:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=100)
    args = parser.parse_args()

    counter = WarningCounter()
    logging.getLogger("lab_llm.caching_completion").addHandler(counter)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "hammer_cache.db")
        fake = FakeLLM()
        cache = CachingCompletion(db_path)
        api = LLMApi(cache(fake), track_usage=True)
        print(f"duckdb {duckdb.__version__}  max_parallel_jobs={args.jobs}  db={db_path}")

        unique = [f"unique prompt {i}" for i in range(400)]
        run_leg("cold", api, unique, args.jobs, fake, counter)
        run_leg("warm", api, unique, args.jobs, fake, counter)
        run_leg("same-key", api, ["identical prompt"] * 200, args.jobs, fake, counter)

        rows, keys = cache.connection.execute(
            "SELECT count(*), count(DISTINCT cache_key) FROM completion_cache").fetchone()
        print(f"final db: rows={rows} distinct_keys={keys} (expect 401/401: 400 unique + 1 identical)")
        for msg in counter.records[:5]:
            print(f"  warning sample: {msg}")


if __name__ == "__main__":
    main()
