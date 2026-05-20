from lab_llm.versa import make_versa_openai_completion, VersaOpenAIModelRamp, make_versa_claude_completion, VersaClaudeModelRamp
from lab_llm import LLMApi, wrap_completion_function, CachingCompletion, ErrorTracker, UsageTracker
import dotenv
import logging
import json
from pydantic import BaseModel, Field
from litellm import Message
import litellm

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()

def weather_lookup(city: str, state: str, country: str) -> dict:
    print("Running function call")
    if city == 'Seattle':
        return {
            "temperature": {
                "January": 47,
                "February": 50,
                "March": 54,
                "April": 60,
                "May": 66,
                "June": 72,
                "July": 77,
                "August": 77,
                "September": 71,
                "October": 61,
                "November": 52,
                "December": 46
            },
            "rainfall": {
                "January": 5.4,
                "February": 3.9,
                "March": 3.7,
                "April": 2.5,
                "May": 1.9,
                "June": 1.5,
                "July": 0.7,
                "August": 0.9,
                "September": 1.6,
                "October": 3.3,
                "November": 5.7,
                "December": 6.0
            }
        }
    else:
        return {}

usage_tracker = UsageTracker()

api = LLMApi(wrap_completion_function(
    # This is the base completion function - works out-of-the-box with litellm.completion
    # The versa version loads endpoint and API keys from the VERSA_ENDPOINT and VERSA_API_KEY env variables.
    make_versa_openai_completion(),
    # make_versa_openai_completion(), # Versa OpenAI version
    # litellm.completion, # General public API version
    cache=CachingCompletion("./llm_cache.db"),

    # Optional parameters
    model_ramp=VersaOpenAIModelRamp, # allows selecting models by size
    # model=VersaClaude.CLAUDE_HAIKU_4_5, # simple default if only one model size is needed
    error_tracker=ErrorTracker(logging.getLogger(__name__), log_file="errors.txt"),
    usage_tracker=usage_tracker,
    seed=42
))

messages: list[Message] = [
    Message(role="system", content="You are a helpful assistant. Always check the weather using the weather_lookup tool before answering."),
    Message(role="user", content="What is the weather like in Seattle in January?")
]

class ResponseModel(BaseModel):
    weather_report: str = Field("The weather in the requested city")
    suggested_clothing: str = Field("What the user should wear when visiting at this time")

# Run LLM with tool calls
message = api.run(
    messages,
    model="xs", # request a model size from the model ramp
    tools=[weather_lookup],
    response_format=ResponseModel,
    reasoning_effort='low',
    max_tokens=200,
    drop_params=True
)

print(message)

print("\n\nTotal usage:", usage_tracker.total_usage())