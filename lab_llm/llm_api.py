import base64
import logging
import os
import time
from typing import Dict, List, Optional

from langchain.globals import set_debug
from langchain_anthropic import ChatAnthropic
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import lab_llm.constants as constants
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.llm import LLM
from lab_llm.llm_cache import LLMCache
from lab_llm.utils import is_valid

"""
Please set your HF, OpenAI, and Versa tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""


class LLMApi(LLM):
    def __init__(
        self,
        cache: LLMCache,
        seed: int,
        model_type: constants.LLMModel,
        error_handler: ErrorCallbackHandler,
        logging,
        timeout: int = 60,
        return_exceptions: bool = False,
    ):
        super().__init__(seed, model_type, logging)
        self.cache = cache
        self.timeout = timeout
        self.is_api = True
        self.error_handler = error_handler
        self.return_exceptions = return_exceptions

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
            return ChatOpenAI(
                api_key=access_token,
                model_name=self.model_type.name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                seed=self.seed,
                timeout=self.timeout,
                rate_limiter=rate_limiter,
                callbacks=[self.error_handler],
            )
        elif constants.is_versa(self.model_type.name):
            api_key = os.environ.get("VERSA_API_KEY")
            resource_endpoint = constants.VERSA_ENDPOINT.replace(
                "<model_name>", self.model_type.name
            )
            return AzureChatOpenAI(
                api_key=api_key,
                api_version=constants.VERSA_API_VERSION,
                azure_endpoint=resource_endpoint,
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout=self.timeout,
                seed=self.seed,
                rate_limiter=rate_limiter,
                callbacks=[self.error_handler],
            )
        elif constants.is_anthropic(self.model_type.name):
            access_token = os.getenv("ANTHROPIC_ACCESS_KEY")
            return ChatAnthropic(
                api_key=access_token,
                model_name=self.model_type.name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                max_retries=0,
                timeout=self.timeout,
                rate_limiter=rate_limiter,
                callbacks=[self.error_handler],
            )
        elif constants.is_bedrock(self.model_type.name):
            access_key = os.getenv("BEDROCK_ACCESS_KEY")
            secret_access_key = os.getenv("BEDROCK_ACCESS_KEY_SECRET")
            return ChatBedrockConverse(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_access_key,
                region_name=constants.AWS_REGION,
                max_tokens=max_new_tokens,
                temperature=temperature,
                model_id=constants.BEDROCK_MAPPINGS[self.model_type.name],
                rate_limiter=rate_limiter,
                callbacks=[self.error_handler],
            )

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
            prompt,
            self.model_type,
            self.seed,
            max_new_tokens,
            temperature,
        )

        if found_in_cache:
            self.logging.info("Cache hit")
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
            if (response_model is not None) and constants.is_meta(self.model_type.name):
                llm_response = response_model.model_validate(
                    from_json(llm_response.content)
                )
            llm_response_content = self._serialize_llm_response(
                llm_response, response_model=response_model
            )
            self.cache.save_response(
                prompt,
                llm_response_content,  # This will be None if serialization failed or validation failed
                self.model_type,
                self.seed,
                max_new_tokens,
                temperature,
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
    ) -> Optional[List[str] | List[BaseModel]]:
        if is_image:
            # the collate_fn is used here because passing in the full payload with the base encoded image causing
            # dataloader to break. The image is encoded into base64 after the batch is created
            dataloader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=self._encode_images
            )
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size)
        self.logging.info(f"DATASET LEN: {len(dataset)}")

        start_time = time.time()
        results = []

        # Use logging_redirect_tqdm to ensure tqdm progress bars are logged properly
        if self.logging:
            # Get all active loggers to redirect
            loggers_to_redirect = [
                logging.getLogger(name)
                for name in logging.root.manager.loggerDict
                if logging.getLogger(
                    name
                ).handlers  # Only include loggers with handlers
            ]
            # Also include root logger if it has handlers
            root_logger = logging.getLogger()
            if root_logger.handlers:
                loggers_to_redirect.append(root_logger)

            with logging_redirect_tqdm(loggers=loggers_to_redirect):
                for i, (batch_prompts, backup_batch_prompts) in enumerate(
                    tqdm(dataloader, desc="Processing batches", unit="batch")
                ):
                    batch_results = await self._process_single_batch(
                        i,
                        batch_prompts,
                        backup_batch_prompts,
                        max_new_tokens,
                        temperature,
                        response_model,
                        validation_context,
                        max_retries,
                        validation_func,
                        callback,
                        requests_per_second,
                    )
                    results.extend(batch_results)
        else:
            # Fallback to regular tqdm if no logger
            for i, (batch_prompts, backup_batch_prompts) in enumerate(
                tqdm(dataloader, desc="Processing batches", unit="batch")
            ):
                batch_results = await self._process_single_batch(
                    i,
                    batch_prompts,
                    backup_batch_prompts,
                    max_new_tokens,
                    temperature,
                    response_model,
                    validation_context,
                    max_retries,
                    validation_func,
                    callback,
                    requests_per_second,
                )
                results.extend(batch_results)

        end_time = time.time()
        execution_time = end_time - start_time
        self.logging.info(f"Results took {execution_time} seconds")
        self.logging.info("NUM RESULTS %d (%d)", len(results), len(dataset))
        assert len(results) == len(dataset)
        return results

    async def _process_single_batch(
        self,
        batch_idx,
        batch_prompts,
        backup_batch_prompts,
        max_new_tokens,
        temperature,
        response_model,
        validation_context,
        max_retries,
        validation_func,
        callback,
        requests_per_second,
    ):
        got_result = False
        num_retries = 0
        while not got_result and (num_retries < max_retries):
            batch_to_use = batch_prompts if num_retries == 0 else backup_batch_prompts
            df = self.cache.get_responses(
                list(batch_to_use),
                self.model_type,
                self.seed,
                max_new_tokens,
                temperature,
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
                    )
                    for prompt, response in zip(prompts_to_run, batch_results):
                        df.loc[df["prompt"] == prompt, "llm_output"] = response

                raw_results = df.llm_output.values.tolist()
                # self.logging.debug(raw_results)
                if response_model is not None:
                    validated_results = []
                    for res in raw_results:
                        if res is not None:
                            try:
                                validated_model = response_model.model_validate_json(
                                    res, context=validation_context
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
                    batch_results = raw_results

                assert len(batch_results) == len(batch_prompts)
                if validation_func is not None:
                    validation_func(batch_results)

                if callback is not None:
                    callback(batch_results)
                got_result = True

                return batch_results
            except Exception as e:
                num_retries += 1
                message = f"Failed batch idx {batch_idx}. Error {e}, {num_retries}"
                print(message)
                if "raw_results" in locals():
                    for res in raw_results:
                        print("parse", res)
                self.logging.error(message)
                if num_retries == max_retries:
                    raise ValueError("Error with LLM batch query")

    async def _run_batch(
        self,
        prompts_to_run,
        llm,
        max_new_tokens,
        temperature,
        response_model: BaseModel = None,
    ):
        system_prompts = [
            [
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content=prompt),
            ]
            for prompt in prompts_to_run
        ]

        batch_results = await llm.abatch(
            system_prompts, return_exceptions=self.return_exceptions
        )
        batch_results_strs = []
        for response in batch_results:
            if isinstance(response, Exception):
                # Log the exception if needed
                self.logging.error(f"LLM call failed: {response}")
                batch_results_strs.append(None)  # Append None for failed calls
            elif response is not None:
                serialized = self._serialize_llm_response(
                    response, response_model=response_model
                )
                batch_results_strs.append(serialized)
            else:
                batch_results_strs.append(None)  # Append None if response was None

        self.cache.save_responses(
            prompts_to_run,
            # Filter out Nones before saving? Or save Nones?
            # Saving Nones might be better for consistency with the returned list.
            [
                str(res) if res is not None else None for res in batch_results_strs
            ],  # Ensure cache gets strings or None
            self.model_type,
            self.seed,
            max_new_tokens,
            temperature,
        )

        return batch_results_strs

    def _encode_images(self, batch_data: list[dict, str]) -> list[dict]:
        updated_payloads = []
        for payload, image_paths in batch_data:
            # Support multiple images separated by "+"
            if isinstance(image_paths, str):
                image_paths = image_paths.split("+")
            elif not isinstance(image_paths, list):
                image_paths = [image_paths]

            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                payload.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    }
                )
            updated_payloads.append(payload)
        return updated_payloads
