import asyncio
import base64
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
from langchain.globals import set_debug
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import lab_llm.constants as constants
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.llm import LLM
from lab_llm.llm_cache import LLMCache
from lab_llm.usage_callback_handler import UsageCallbackHandler

"""
Please set your HF, OpenAI, and Versa tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""


def is_valid(model: BaseModel, data_str: str, context: Optional[Dict] = None) -> bool:
    """
    Validates data against a Pydantic model and returns True if valid, False otherwise.
    """
    try:
        model.model_validate_json(data_str, context=context)
        return True
    except ValidationError:
        return False


class LLMApi(LLM):
    def __init__(
        self,
        cache: LLMCache,
        seed: int,
        model_type: constants.LLMModel,
        error_handler: ErrorCallbackHandler,
        logging,
        reasoning_effort: str = "medium",
        verbosity: str = "medium",
        timeout: int = 60,
        return_exceptions: bool = False,
        track_usage: bool = False,
    ):
        super().__init__(seed, model_type, logging)
        self.cache = cache
        self.timeout = timeout
        self.is_api = True
        self.error_handler = error_handler
        self.return_exceptions = return_exceptions
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.track_usage = track_usage
        self.usage_handler = UsageCallbackHandler() if track_usage else None

    def _get_cache_reasoning_params(self):
        """Only include reasoning params in cache key for models that use them."""
        if constants.is_reasoning_model(self.model_type.name):
            return {"reasoning_effort": self.reasoning_effort, "verbosity": self.verbosity}
        return {"reasoning_effort": None, "verbosity": None}

    @staticmethod
    def _format_usage_line(prefix, input_tokens, output_tokens,
                           cached_tokens=0, reasoning_tokens=0, suffix=""):
        """Format a token usage line. Shows detail breakdowns when present."""
        parts = [f"{input_tokens:,} input"]
        if cached_tokens:
            parts[0] += f" ({cached_tokens:,} cached)"
        parts.append(f"{output_tokens:,} output")
        if reasoning_tokens:
            parts[1] += f" ({reasoning_tokens:,} reasoning)"
        msg = f"{prefix}: {' / '.join(parts)}"
        if suffix:
            msg += f" {suffix}"
        return msg

    def _report_usage(self, usage: dict, cached: bool = False):
        """Print and log token usage if track_usage is enabled."""
        if not self.track_usage:
            return

        if cached:
            msg = "Token usage: 0 input / 0 output (cached)"
        elif usage is None:
            return
        else:
            msg = self._format_usage_line(
                "Token usage",
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cached_tokens", 0),
                usage.get("reasoning_tokens", 0),
            )

        print(msg)
        self.logging.info(msg)

    def _report_batch_usage(
        self, input_tokens, output_tokens, cached_tokens, reasoning_tokens, num_queries
    ):
        """Print and log batch token usage if track_usage is enabled."""
        if not self.track_usage:
            return
        if input_tokens == 0 and output_tokens == 0:
            return

        msg = self._format_usage_line(
            "Batch token usage",
            input_tokens, output_tokens, cached_tokens, reasoning_tokens,
            suffix=f"({num_queries} queries)",
        )

        print(msg)
        self.logging.info(msg)

    def _serialize_llm_response(self, llm_response, response_model: BaseModel = None):
        try:
            if response_model is not None:
                return llm_response.model_dump_json()
            else:
                return llm_response.content
        except Exception as e:
            if self.return_exceptions:
                self.logging.error(e)
                return None
            else:
                self.logging.error(e)
                raise Exception(f"Was unable to serialize llm response {e}")

    def _get_callbacks(self):
        """Build the callbacks list, including usage handler if tracking is enabled."""
        callbacks = [self.error_handler]
        if self.usage_handler is not None:
            callbacks.append(self.usage_handler)
        return callbacks

    def get_client(self, max_new_tokens=4000, temperature=0, requests_per_second=None):
        if requests_per_second:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
        else:
            rate_limiter = None

        if constants.is_openai(self.model_type.name):
            access_token = os.getenv("OPENAI_ACCESS_TOKEN")
            kwargs = dict(
                api_key=access_token,
                model_name=self.model_type.name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                seed=self.seed,
                timeout=self.timeout,
                rate_limiter=rate_limiter,
                callbacks=self._get_callbacks(),
            )
            if constants.is_reasoning_model(self.model_type.name):
                kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["model_kwargs"] = {"extra_body": {"verbosity": self.verbosity}}
            return ChatOpenAI(**kwargs)
        elif constants.is_versa(self.model_type.name):
            api_key = os.environ.get("VERSA_API_KEY")
            versa_endpoint = os.environ.get("VERSA_ENDPOINT")
            if not versa_endpoint:
                raise ValueError(
                    "VERSA_ENDPOINT environment variable is required for Versa models. "
                    "Set it to your Azure OpenAI endpoint URL with '<model_name>' as placeholder."
                )
            resource_endpoint = versa_endpoint.replace(
                "<model_name>", self.model_type.name
            )
            kwargs = dict(
                api_key=api_key,
                api_version=constants.VERSA_API_VERSION,
                azure_endpoint=resource_endpoint,
                temperature=temperature,
                timeout=self.timeout,
                seed=self.seed,
                rate_limiter=rate_limiter,
                callbacks=self._get_callbacks(),
            )
            if constants.is_reasoning_model(self.model_type.name):
                kwargs["max_completion_tokens"] = max_new_tokens
                kwargs["reasoning_effort"] = self.reasoning_effort
                kwargs["model_kwargs"] = {"extra_body": {"verbosity": self.verbosity}}
                del kwargs["temperature"]
                del kwargs["seed"]
            else:
                kwargs["max_tokens"] = max_new_tokens
            return AzureChatOpenAI(**kwargs)
        elif constants.is_bedrock(self.model_type.name):
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_KEY_SECRET")
            base_url = os.getenv("BEDROCK_ENDPOINT_URL")
            kwargs = dict(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_access_key,
                region_name=constants.AWS_REGION,
                max_tokens=max_new_tokens,
                temperature=temperature,
                model=constants.BEDROCK_MAPPINGS[self.model_type.name],
                rate_limiter=rate_limiter,
                callbacks=self._get_callbacks(),
            )
            if base_url:
                kwargs["base_url"] = base_url
            return ChatBedrockConverse(**kwargs)

    # Note: if passing in an image here the prompt should contain the base64 encoded image.
    # see _encode_images for an example
    def get_output(
        self,
        prompt,
        max_new_tokens=4000,
        temperature=0,
        response_model: BaseModel = None,
        validation_context: Optional[Dict] = None,
    ) -> Optional[str | BaseModel]:
        set_debug(True)
        self.logging.info("LLM (%s) prompt %s", self.model_type, prompt)
        found_in_cache, cached_response = self.cache.get_response(
            str(prompt),
            self.model_type,
            self.seed,
            max_new_tokens,
            temperature,
            **self._get_cache_reasoning_params(),
        )

        if found_in_cache:
            self.logging.info("Cache hit")
            self._report_usage(None, cached=True)
            if response_model is not None:
                if cached_response is not None:
                    validated_model = response_model.model_validate_json(
                        cached_response, context=validation_context
                    )
                    return validated_model
                else:
                    self.logging.info("LLM response (cached) None")
                    return None
            else:
                # No response model, return the cached string directly (could be None)
                self.logging.info("LLM response (cached) %s", cached_response)
                return cached_response
        else:  # Not found in cache
            self.logging.info("Cache miss")
            if self.usage_handler is not None:
                self.usage_handler.reset()
            llm = self.get_client(max_new_tokens, temperature)
            if (response_model is not None) and (
                not constants.is_meta(self.model_type.name)
            ):
                llm = llm.with_structured_output(response_model)
            messages = [
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content=prompt),
            ]
            llm_response = llm.invoke(messages)
            if self.usage_handler is not None:
                self._report_usage(self.usage_handler.last_usage)
            if (response_model is not None) and constants.is_meta(self.model_type.name):
                llm_response = response_model.model_validate(
                    from_json(llm_response.content)
                )
            llm_response_content = self._serialize_llm_response(
                llm_response, response_model=response_model
            )

            # Only cache successful LLM responses.
            # Failed requests (None) are not cached, allowing automatic retry on next run.
            if llm_response_content is not None:
                self.cache.save_response(
                    str(prompt),
                    llm_response_content,
                    self.model_type,
                    self.seed,
                    max_new_tokens,
                    temperature,
                    **self._get_cache_reasoning_params(),
                )

        self.logging.info("LLM response %s", llm_response)
        return llm_response

    async def get_outputs(
        self,
        dataset: Dataset,
        batch_size: int = 10,
        max_new_tokens=4000,
        is_image=False,
        max_retries=2,
        temperature: float = 0,
        callback=None,
        validation_func=None,
        requests_per_second=None,
        response_model: BaseModel = None,
        validation_context: Optional[Dict] = None,
        prompt_cache_key: Optional[str] = None,
    ) -> Optional[List[str] | List[BaseModel]]:
        if is_image:
            # the collate_fn is used here because passing in the full payload with the base encoded image causing
            # dataloader to break. The image is encoded into base64 after the batch is created
            dataloader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=self._encode_images
            )
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size)
        self.logging.debug("Dataset length: %d", len(dataset))

        start_time = time.time()
        results = []
        for i, (batch_prompts, backup_batch_prompts) in enumerate(tqdm(dataloader)):
            got_result = False
            num_retries = 0
            while not got_result and (num_retries < max_retries):
                df = self.cache.get_responses(
                    (
                        self._make_prompts_strs(batch_prompts)
                        if num_retries == 0
                        else self._make_prompts_strs(backup_batch_prompts)
                    ),
                    self.model_type,
                    self.seed,
                    max_new_tokens,
                    temperature,
                    **self._get_cache_reasoning_params(),
                )
                if (
                    response_model
                    and (not self.return_exceptions)
                    and (num_retries < (max_retries - 1))
                ):
                    null_mask = [
                        not is_valid(response_model, res, context=validation_context)
                        for res in df.llm_output
                    ]
                    prompts_to_run = df.iloc[null_mask].prompt.values.tolist()
                else:
                    prompts_to_run = df[df.llm_output.isna()].prompt.values.tolist()

                    # Identify prompts needing an API call (not found in cache)
                    prompts_to_run_df = df[~df["found_in_cache"]].copy()
                    prompts_to_run = prompts_to_run_df["prompt"].tolist()
                try:
                    if prompts_to_run:
                        llm = self.get_client(
                            max_new_tokens, temperature, requests_per_second
                        )
                        if (response_model is not None) and (
                            not constants.is_meta(self.model_type.name)
                        ):
                            llm = llm.with_structured_output(response_model)

                        batch_results = await self._run_batch(
                            prompts_to_run,
                            llm,
                            max_new_tokens,
                            temperature,
                            response_model=(
                                response_model
                                if not constants.is_meta(self.model_type.name)
                                else None
                            ),
                            prompt_cache_key=prompt_cache_key,
                        )
                        for prompt, response in zip(prompts_to_run, batch_results):
                            df.loc[df["prompt"] == prompt, "llm_output"] = response

                    raw_results = df.llm_output.values.tolist()
                    # self.logging.info(raw_results)
                    if response_model is not None:
                        validated_results = []
                        for res in raw_results:
                            if res is not None:
                                try:
                                    validated_model = (
                                        response_model.model_validate_json(
                                            res, context=validation_context
                                        )
                                    )
                                    validated_results.append(validated_model)
                                except ValidationError as e:
                                    self.logging.error(
                                        f"Validation failed for response: {res}. Error: {e}"
                                    )
                                    validated_results.append(None)
                            else:
                                # Append None if res was None initially
                                validated_results.append(None)
                        batch_results = validated_results
                    else:
                        # No response model, just use raw results
                        batch_results = raw_results

                    assert len(batch_results) == len(batch_prompts)
                    if validation_func is not None:
                        validation_func(batch_results)

                    results += batch_results

                    if callback is not None:
                        callback(results)
                    got_result = True
                except Exception as e:
                    num_retries += 1
                    message = f"Failed batch idx {i}. Error {e}, {num_retries}"
                    self.logging.warning(message)
                    for res in raw_results:
                        self.logging.debug("parse result: %s", res)
                    if num_retries == max_retries:
                        raise ValueError("Error with LLM batch query")

        end_time = time.time()
        execution_time = end_time - start_time
        self.logging.info(f"Results took {execution_time} seconds")
        self.logging.info("NUM RESULTS %d (%d)", len(results), len(dataset))
        assert len(results) == len(dataset)
        return results

    async def _run_batch(
        self,
        prompts_to_run,
        llm,
        max_new_tokens,
        temperature,
        response_model: BaseModel = None,
        prompt_cache_key: Optional[str] = None,
    ):
        system_prompts = [
            [
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content=prompt),
            ]
            for prompt in prompts_to_run
        ]

        batch_kwargs = {"return_exceptions": self.return_exceptions}
        if prompt_cache_key and constants.is_versa(self.model_type.name):
            batch_kwargs["extra_body"] = {"prompt_cache_key": prompt_cache_key}
        batch_results = await llm.abatch(system_prompts, **batch_kwargs)
        batch_results_strs = []
        batch_input_tokens = 0
        batch_output_tokens = 0
        batch_cached_tokens = 0
        batch_reasoning_tokens = 0
        for idx, response in enumerate(batch_results):
            if isinstance(response, Exception):
                # Log the exception with prompt context for error tracking
                prompt = str(prompts_to_run[idx])
                self.logging.error(
                    f"LLM call failed for prompt {idx}: {type(response).__name__}: {response}"
                )

                # If error tracker is available, log with full context
                if (
                    hasattr(self.error_handler, "error_tracker")
                    and self.error_handler.error_tracker
                ):
                    prompt_hash = hashlib.sha256(
                        prompt.strip().encode("utf-8")
                    ).hexdigest()

                    self.error_handler.error_tracker.log_error(
                        error=response,
                        prompt=prompt,
                        prompt_hash=prompt_hash,
                        context={
                            "model_type": str(self.model_type.name),
                            "timeout": self.timeout,
                            #"max_tokens": max_new_tokens,
                            "max_completion_tokens": max_new_tokens,
                            "temperature": temperature,
                            "batch_index": idx,
                        },
                        include_traceback=False,
                    )

                batch_results_strs.append(None)  # Append None for failed calls
            elif response is not None:
                # Extract token usage before serializing.
                # Uses the shared parse_usage_metadata() so extraction logic
                # stays in one place (the callback handler captures usage for
                # single calls; here we read directly from batch responses
                # because abatch() returns a list with no per-item callbacks).
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    parsed = UsageCallbackHandler.parse_usage_metadata(
                        response.usage_metadata
                    )
                    if parsed:
                        batch_input_tokens += parsed.get('input_tokens', 0)
                        batch_output_tokens += parsed.get('output_tokens', 0)
                        batch_cached_tokens += parsed.get('cached_tokens', 0)
                        batch_reasoning_tokens += parsed.get('reasoning_tokens', 0)
                serialized = self._serialize_llm_response(
                    response, response_model=response_model
                )
                batch_results_strs.append(serialized)
            else:
                batch_results_strs.append(None)  # Append None if response was None

        # Log batch token usage for rate limit monitoring
        if batch_input_tokens > 0 or batch_output_tokens > 0:
            batch_total = batch_input_tokens + batch_output_tokens
            self.logging.info(
                f"Batch token usage: input={batch_input_tokens} (cached={batch_cached_tokens}), "
                f"output={batch_output_tokens} (reasoning={batch_reasoning_tokens}), total={batch_total}"
            )
        self._report_batch_usage(
            batch_input_tokens,
            batch_output_tokens,
            batch_cached_tokens,
            batch_reasoning_tokens,
            len(prompts_to_run),
        )

        # Only cache successful responses - filter out None values.
        successful_prompts = []
        successful_results = []
        for prompt, result in zip(prompts_to_run, batch_results_strs):
            if result is not None:
                successful_prompts.append(prompt)
                successful_results.append(str(result))

        if successful_prompts:
            self.cache.save_responses(
                self._make_prompts_strs(successful_prompts),
                successful_results,
                self.model_type,
                self.seed,
                max_new_tokens,
                temperature,
                **self._get_cache_reasoning_params(),
            )

        return batch_results_strs

    _make_prompts_strs = lambda self, prompts: [str(prompt) for prompt in prompts]

    def _encode_images(self, batch_data: list[dict, str]) -> list[dict]:
        updated_payloads = []
        for payload, image_path in batch_data:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )
            updated_payloads.append(payload)
        return updated_payloads, []
