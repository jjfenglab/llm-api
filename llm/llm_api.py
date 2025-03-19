from typing import Optional, List

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from torch.utils.data import Dataset, DataLoader
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.globals import set_debug

from pydantic_core import from_json
from tqdm import tqdm
import os
import asyncio
import base64
import time
import pandas as pd

from pydantic import BaseModel

from llm.llm import LLM
from llm.llm_cache import LLMCache
import llm.constants as constants
from llm.error_callback_handler import ErrorCallbackHandler

"""
Please set your HF, OpenAI, and Versa tokens in a .env file. Note: The API currently only supports image inference for
OpenAI models
"""

class LLMApi(LLM):
    def __init__(self,
                 cache: LLMCache,
                 seed: int,
                 model_type: constants.LLMModel,
                 error_handler: ErrorCallbackHandler,
                 logging,
                 timeout=60
                 ):
        super().__init__(seed, model_type, logging)
        self.cache = cache
        self.timeout = timeout
        self.is_api = True
        self.error_handler = error_handler
    
    def _serialize_llm_response(self, llm_response, response_model: BaseModel=None):
        if response_model is not None:
            return llm_response.model_dump_json()
        else:
            return llm_response.content

    def get_client(self, max_new_tokens=4000, temperature=0, requests_per_second=None):
        if requests_per_second:
            rate_limiter = InMemoryRateLimiter(
                    requests_per_second=requests_per_second,
                    check_every_n_seconds=0.1,
                    max_bucket_size=10
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
                    callbacks=[self.error_handler]
                    )
        elif constants.is_versa(self.model_type.name):
            api_key = os.environ.get('VERSA_API_KEY')
            resource_endpoint = constants.VERSA_ENDPOINT.replace("<model_name>", self.model_type.name)
            return AzureChatOpenAI(
                    api_key=api_key,
                    api_version=constants.VERSA_API_VERSION,
                    azure_endpoint=resource_endpoint,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=self.timeout,
                    seed=self.seed,
                    rate_limiter=rate_limiter,
                    callbacks=[self.error_handler]
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
                    callbacks=[self.error_handler]
                    )

    # Note: if passing in an image here the prompt should contain the base64 encoded image.
    # see _encode_images for an example
    def get_output(
            self, 
            prompt, 
            max_new_tokens=4000, 
            temperature=0,
            response_model:BaseModel=None,
            ) -> Optional[str | BaseModel]:
        set_debug(True)
        self.logging.info("LLM (%s) prompt %s", self.model_type, prompt)
        llm_response = self.cache.get_response(
            prompt, 
            self.model_type, 
            self.seed, 
            max_new_tokens, 
            temperature,
        )
        print("CACHED", llm_response)
        if (llm_response is not None) and (response_model is not None):
            llm_response = response_model.model_validate_json(llm_response)
        
        if llm_response is None:
            llm = self.get_client(max_new_tokens, temperature)
            if (response_model is not None) and (not constants.is_meta(self.model_type.name)):
                llm = llm.with_structured_output(response_model)
            messages = [
                     SystemMessage(content="You are a helpful assistant"),
                     HumanMessage(content=prompt)
                     ]
            llm_response = llm.invoke(messages)
            if (response_model is not None) and constants.is_meta(self.model_type.name):
                llm_response = response_model.model_validate(from_json(llm_response.content))
            llm_response_content = self._serialize_llm_response(llm_response, response_model=response_model)
            self.cache.save_response(
                prompt, 
                llm_response_content,
                self.model_type, 
                self.seed, 
                max_new_tokens, 
                temperature
            )
            
        self.logging.info("LLM response %s", llm_response)
        return llm_response

    async def get_outputs(
            self, 
            dataset: Dataset, 
            batch_size:int = 10, 
            max_new_tokens=4000,
            is_image = False,
            max_retries = 2,
            temperature:float = 0,
            callback = None,
            validation_func = None,
            requests_per_second = None,
            response_model:BaseModel=None,
            ) -> Optional[List[str] | List[BaseModel]]:
        llm = self.get_client(max_new_tokens, temperature, requests_per_second)
        if (response_model is not None) and (not constants.is_meta(self.model_type.name)):
            llm = llm.with_structured_output(response_model)
        
        if is_image:
            # the collate_fn is used here because passing in the full payload with the base encoded image causing
            # dataloader to break. The image is encoded into base64 after the batch is created
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=self._encode_images)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size)
        print("DATASET LEN", len(dataset))

        start_time = time.time()
        results = []
        for i, (batch_prompts, backup_batch_prompts) in enumerate(tqdm(dataloader)):
            got_result = False
            num_retries = 0
            while not got_result and (num_retries < max_retries):
                df = self.cache.get_responses(
                    batch_prompts if num_retries == 0 else backup_batch_prompts, 
                    self.model_type, 
                    self.seed, 
                    max_new_tokens, 
                    temperature
                )

                prompts_to_run = df[df.llm_output.isnull()].prompt.values.tolist()
                try: 
                    if prompts_to_run:
                        batch_results = await self._run_batch(
                                prompts_to_run, 
                                llm, 
                                max_new_tokens, 
                                temperature,
                                response_model=response_model if not constants.is_meta(self.model_type.name) else None
                                )
                        for prompt, response in zip(prompts_to_run, batch_results):
                            df.loc[df["prompt"] == prompt, "llm_output"] = response

                    raw_results = df.llm_output.values.tolist()
                    self.logging.info(raw_results)
                    if response_model is not None:
                        batch_results = [response_model.model_validate_json(res) for res in raw_results]
                    else:
                        batch_results = raw_results
                    if validation_func is not None:
                        validation_func(batch_results)
                        
                    results += batch_results

                    if callback is not None:
                        callback(results)
                    got_result = True
                except Exception as e:
                    num_retries += 1
                    message = f"Failed batch idx {i}. Error {e}, {num_retries}"
                    print(message)
                    for res in raw_results:
                        print("parse", res)
                    self.logging.error(message)
                    if num_retries == max_retries:
                        raise ValueError("Error with LLM batch query")

        end_time = time.time()
        execution_time = end_time - start_time
        self.logging.info(f"Results took {execution_time} seconds")
        self.logging.info("NUM RESULTS %d", len(results))
        assert len(results) == len(dataset)
        return results

    async def _run_batch(
            self, 
            prompts_to_run, 
            llm, 
            max_new_tokens, 
            temperature,
            response_model: BaseModel = None
            ):
        system_prompts = [[
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content=prompt)
            ] for prompt in prompts_to_run]
        
        batch_results = await llm.abatch(system_prompts)
        batch_results_strs = [self._serialize_llm_response(response, response_model=response_model) for response in batch_results]

        self.cache.save_responses(
            prompts_to_run, 
            batch_results_strs, 
            self.model_type, 
            self.seed, 
            max_new_tokens, 
            temperature
        )

        return batch_results_strs

    def _encode_images(self, batch_data: list[dict, str]) -> list[dict]:
        updated_payloads = []
        for (payload, image_path) in batch_data:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            updated_payloads.append(payload)
        return updated_payloads
