import logging
import asyncio
import os
from dotenv import load_dotenv

from lab_llm.llm_api import LLMApi
from lab_llm.constants import *
from lab_llm.dataset import *
from lab_llm.llm_cache import LLMCache
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.duckdb_handler import DuckDBHandler

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model = LLMModel(name=VersaOpenAi.GPT4_O_2024_08)
# model = LLMModel(name=Cohere.COMMAND_R)
model = LLMModel(name=OpenAi.GPT4_O_MINI)
cache_path = os.environ.get("LLM_CACHE_PATH", "./cache.db")
db_handle = DuckDBHandler(cache_path)
cache = LLMCache(db_handle)


handler = ErrorCallbackHandler(logger)
api = LLMApi(cache, 10, model, handler, logger)

test_dataset = TextDataset(
        ["clinical_note", "what is the color red"], 
        )

# prompt = "test"
# print(api.get_output(prompt))

output = asyncio.run(api.get_outputs(test_dataset))

print(output)
