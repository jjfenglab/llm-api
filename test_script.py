from llm.llm_api import LLMApi
from llm.constants import *
from llm.dataset import *
from llm.llm_cache import LLMCache
from llm.error_callback_handler import ErrorCallbackHandler
from llm.duckdb_handler import DuckDBHandler
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model = LLMModel(name=VersaOpenAi.GPT4_O_2024_08)
# model = LLMModel(name=Cohere.COMMAND_R)
model = LLMModel(name=OpenAi.GPT4_O_MINI)
db_handle = DuckDBHandler("/Users/avkothari/Desktop/llm-judge/cache.db")
cache = LLMCache(db_handle)


handler = ErrorCallbackHandler(logger)
api = LLMApi(cache, 10, model, handler, logger)

test_dataset = TextDataset(
        ["", "what is the color red"], 
        )

# prompt = "test"
# print(api.get_output(prompt))

output = asyncio.run(api.get_outputs(test_dataset))

print(output)
