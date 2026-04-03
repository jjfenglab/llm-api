from lab_llm.new.versa_openai import versa_openai_completion, VersaOpenAIModels, DefaultVersaModelRamp
from lab_llm.new import make_function_tool, CachingCompletion, ErrorTracker, UsageTracker
import dotenv
import logging
import json
from pydantic import BaseModel, Field
from litellm import Message, ModelResponse

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

# We define all the functionality we want to add to our completion process by wrapping
# the completion function in decorators
error_tracker = ErrorTracker(logging.getLogger(__name__), log_file="errors.txt")
cache = CachingCompletion("./llm_cache.db")
usage_tracker = UsageTracker()

completion = usage_tracker(
    error_tracker(
        cache(
            DefaultVersaModelRamp(
                # This is the base completion function - works out-of-the-box with litellm.completion
                # The versa version loads endpoint and API keys from the VERSA_ENDPOINT and VERSA_API_KEY env variables.
                versa_openai_completion()
            )
        )
    )  
)

messages: list[Message] = [
    Message(role="system", content="You are a helpful assistant. Always check the weather using the weather_lookup tool before answering."),
    Message(role="user", content="What is the weather like in Seattle in January?")
]

class ResponseModel(BaseModel):
    weather_report: str = Field("The weather in the requested city")
    suggested_clothing: str = Field("What the user should wear when visiting at this time")

while True:
    response = completion(
        "xs",
        messages=messages,
        tools=[
            make_function_tool(weather_lookup)
        ],
        response_format=ResponseModel,
        reasoning_effort='low',
        max_tokens=200,
        drop_params=True
    )
    print(response, response.usage)
    message = response.choices[0].message
    messages.append(message)
    if tool_calls := message.tool_calls:
        for tool_call in tool_calls:
            assert tool_call.function.name == "weather_lookup"
            args = json.loads(tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(weather_lookup(args["city"],
                                              args["state"],
                                              args["country"]))
            })
    else:
        break

print("\n\nTotal usage:", usage_tracker.total_usage())