"""Measure run_batch's true concurrency using an injected fake completion function.

Runs a batch of N prompts through LLMApi.run_batch with a completion function
that sleeps `--latency` seconds (simulating a GIL-releasing network call) and
records the peak number of simultaneously in-flight calls. On an unfixed tree
the peak saturates at the asyncio default-executor cap min(32, cpu_count + 4)
no matter what max_parallel_jobs is set to.

Usage: python scripts/poc_concurrency.py [--n 200] [--latency 0.5] [--jobs 8 20 64 100]
"""

import argparse
import asyncio
import os
import threading
import time

from litellm import ModelResponse
from litellm.types.utils import Usage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from lab_llm.api import LLMApi


def make_response(model: str, content: str) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-poc",
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content=content), finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class ConcurrencyProbe:
    """Fake completion function that sleeps and tracks peak in-flight calls."""

    def __init__(self, latency: float):
        self.latency = latency
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def __call__(self, model, messages=None, **kwargs) -> ModelResponse:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.latency)
        with self._lock:
            self.active -= 1
        return make_response(model or "fake-model", "ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--latency", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, nargs="+", default=[8, 20, 64, 100])
    args = parser.parse_args()

    cpu = os.cpu_count()
    default_cap = min(32, cpu + 4)
    print(f"cpu_count={cpu}  asyncio-default-executor cap=min(32, cpu+4)={default_cap}")
    print(f"batch n={args.n}  simulated latency={args.latency}s")
    print(f"{'max_parallel_jobs':>18} {'peak_concurrency':>17} {'wall_s':>8} {'ideal_wall_s':>13}")

    for mpj in args.jobs:
        probe = ConcurrencyProbe(args.latency)
        api = LLMApi(probe, track_usage=True)
        prompts = [f"prompt {i}" for i in range(args.n)]
        t0 = time.perf_counter()
        # Fresh asyncio.run per setting -> fresh event loop and default executor,
        # so one setting's thread pool cannot leak into the next measurement.
        results = asyncio.run(api.run_batch(prompts, max_parallel_jobs=mpj))
        wall = time.perf_counter() - t0
        assert probe.calls == args.n and all(r == "ok" for r in results)
        ideal = args.latency * (args.n / mpj + (1 if args.n % mpj else 0))
        print(f"{mpj:>18} {probe.peak:>17} {wall:>8.2f} {ideal:>13.2f}")


if __name__ == "__main__":
    main()
