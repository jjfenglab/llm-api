import os
from typing import Callable, Any, Optional, Literal
from collections.abc import Sequence
from .types import CompletionFunction, CompletionFunctionWrapper
import logging
from functools import wraps

type ModelSize = Literal["xs", "sm", "md", "lg", "xl"]
MODEL_SIZES: list[ModelSize] = ["xs", "sm", "md", "lg", "xl"]

class ModelRamp(CompletionFunctionWrapper):
    """
    A completion function wrapper that routes model size requests to specific models
    using a "model ramp", or sequence of models of increasing size. The supported
    size requests are "xs", "sm", "md", "lg", and "xl". The provided model ramp
    will be automatically mapped to these sizes.
    """
    _model_ramp: dict[ModelSize, str]

    def __init__(self, models: Sequence[str], default_size: ModelSize = "md"):
        if len(models) == 1:
            self._model_ramp = {
                k: models[0] for k in MODEL_SIZES
            }
        elif len(models) == 2:
            self._model_ramp = {
                "xs": models[0], "sm": models[0],
                "md": models[1], "lg": models[1], "xl": models[1]
            }
        elif len(models) == 3:
            self._model_ramp = {
                "xs": models[0], "sm": models[0],
                "md": models[1], 
                "lg": models[2], "xl": models[2]
            }
        elif len(models) == 4:
            self._model_ramp = {
                "xs": models[0], 
                "sm": models[1],
                "md": models[2], 
                "lg": models[3], "xl": models[3]
            }
        else:
            self._model_ramp = dict(zip(MODEL_SIZES, models))
        self._default_size = default_size

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(size_request: Optional[str], **kwargs) -> Any:
            # Route the model if it is a size request; if it is not one of the requested sizes then just pass it through
            if size_request is None:
                size_request = self._default_size
            if size_request in self._model_ramp:
                return func(self._model_ramp[size_request], **kwargs)
            return func(size_request, **kwargs)
        
        return wrapped_func
    
class ModelDefault(CompletionFunctionWrapper):
    """
    Injects the given default model (or model request) if the model provided is None.
    """
    def __init__(self, default_model: str):
        self._default_model = default_model

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(model: Optional[str], **kwargs) -> Any:
            if model is None:
                model = self._default_model
            return func(model, **kwargs)
        
        return wrapped_func