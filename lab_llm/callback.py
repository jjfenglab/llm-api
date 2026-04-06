from typing import Any, Protocol
from .types import CompletionFunctionWrapper, CompletionKwargs, ModelResponse
from functools import wraps

class CompletionCallbackFunction(Protocol):
    def __call__(self, model: str, input_kwargs: CompletionKwargs, result: ModelResponse):
        ...
        
class CompletionCallback(CompletionFunctionWrapper):
    """
    A completion function wrapper that calls a given function whenever an
    LLM call completes. Callbacks will not run if the completion raises an 
    exception. Callbacks cannot modify the output of the completion.
    """
    
    def __init__(self, callback: CompletionCallbackFunction):
        self._callback = callback

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(model: str, **kwargs) -> Any:
            result = func(model, **kwargs)
            if self._callback:
                self._callback(model, kwargs, result)
            return result
        
        return wrapped_func
    