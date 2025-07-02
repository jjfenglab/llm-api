from typing import Dict, Optional

import tiktoken
from pydantic import BaseModel, ValidationError


def count_tokens(text, model="gpt-4"):
    # supported models: https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken/model.py#L24
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def is_valid(model: BaseModel, data_str: str, context: Optional[Dict] = None) -> bool:
    """
    Validates data against a Pydantic model and returns True if valid, False otherwise.
    """
    try:
        model.model_validate_json(data_str, context=context)
        return True
    except ValidationError:
        return False
