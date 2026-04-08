import os
from typing import Callable, Any, Optional
from ..constants import VersaClaude, VERSA_API_VERSION
from ..types import CompletionFunction
from ..parameter_wrappers import ModelRamp
import logging
from functools import wraps

VersaClaudeModelRamp = ModelRamp([
    VersaClaude.CLAUDE_HAIKU_4_5,
    VersaClaude.CLAUDE_SONNET_4_6,
    VersaClaude.CLAUDE_OPUS_4_6
])

VERSA_AWS_REGION = "us-west-2"

def make_versa_claude_completion(completion_func: Optional[CompletionFunction] = None, access_key: Optional[str] = None, secret_key: Optional[str] = None, endpoint: Optional[str] = None, region: Optional[str] = None) -> CompletionFunction:
    """
    Wrapper around litellm.completion that configures Claude on AWS Bedrock parameters from environment variables.

    Maps environment variables:
        VERSA_ENDPOINT: endpoint URL
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: secret key
        AWS_REGION: region code
        
    Args:
        completion_func: Function to wrap. If none is passed, uses litellm.completion.

    Returns:
        Wrapped function that passes in Versa Azure OpenAI credentials
    """
    # Get Bedrock configuration
    endpoint = endpoint or os.getenv("VERSA_ENDPOINT")
    access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    region = region or os.getenv("AWS_REGION", VERSA_AWS_REGION)

    if not endpoint:
        raise ValueError("Versa endpoint must be provided via VERSA_ENDPOINT environment variable")
    if not access_key:
        raise ValueError("Access key must be provided via AWS_ACCESS_KEY_ID environment variable")
    if not secret_key:
        raise ValueError("Access key must be provided via AWS_SECRET_ACCESS_KEY environment variable")
    
    # Added to standard Versa endpoint URL
    endpoint = os.path.join(endpoint, "awsai")

    logging.info("Versa AWS Bedrock Endpoint URL: %s", endpoint)

    # Set Bedrock parameters for litellm
    bedrock_kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "aws_region_name": region,
        "aws_bedrock_runtime_endpoint": endpoint
    }

    if not completion_func:
        import litellm
        completion_func = litellm.completion

    @wraps(completion_func)
    def wrapper(model: str, **kwargs) -> Any:
        kwargs = {**kwargs}
        if "reasoning_effort" in kwargs:
            effort = kwargs.pop("reasoning_effort")
            kwargs.setdefault("output_config", {})
            kwargs["output_config"]["effort"] = effort
        # if "max_tokens" in kwargs:
        #     kwargs.setdefault("thinking", {
        #         "type": "enabled"
        #     })
        #     kwargs["thinking"]["budget_tokens"] = max(kwargs["thinking"].get("budget_tokens", 0), int(kwargs["max_tokens"] * 0.5))
        # print("Kwargs:", kwargs)
        return completion_func(model, **{**kwargs, **bedrock_kwargs})
    return wrapper