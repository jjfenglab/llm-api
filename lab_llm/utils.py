"""
Tools for creating function tools from Python functions using inspection.
(Generated with Claude Code)
"""

import inspect
from typing import Any, Callable, get_type_hints, get_origin, get_args, Union, Annotated, Optional, List, Union
from collections.abc import Mapping
from .types import FunctionDefinition, FunctionToolDict, MessageDict, Tool
from litellm import Message

class FunctionToolWrapper(Tool):
    """
    A helper type that wraps Python functions and produces JSON schemas
    for passing them into LLM calls, while also remaining callable.
    """
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        # Get function name
        self._name = name or func.__name__

        # Get description from docstring
        description = description or inspect.getdoc(func) or f"Function {self._name}"

        # Get function signature and type hints
        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError):
            # Fallback if type hints can't be resolved
            type_hints = {}

        # Build JSON schema for parameters
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip self parameter for methods
            if param_name == 'self':
                continue

            # Determine if parameter is required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

            # Get parameter type information
            param_type = type_hints.get(param_name)
            param_schema = _type_to_json_schema(param_type, param.annotation)

            properties[param_name] = param_schema

            if get_origin(param_type) is Annotated:
                args = get_args(param_type)
                if len(args) >= 2:
                    # First arg is the actual type, second+ args are metadata
                    param_type = args[0]
                    # Look for string descriptions in metadata
                    for metadata in args[1:]:
                        if isinstance(metadata, str):
                            properties["description"] = metadata
                            break


        # Build the complete parameter schema
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        # Create the function definition
        function_def: FunctionDefinition = {
            "name": self._name,
            "description": description,
            "parameters": parameters,
            "strict": False
        }

        # Create the complete FunctionToolDict
        self._schema: FunctionToolDict = {
            "type": "function",
            "function": function_def
        }
        self._func = func

    def __call__(self, **kwargs):
        return self._func(**kwargs)
    
    @property
    def name(self):
        return self._name
    
    def to_json_schema(self):
        return self._schema


def _type_to_json_schema(type_hint: Any, annotation: Any) -> dict:
    """
    Convert a Python type hint to JSON schema format.

    Args:
        type_hint: The resolved type hint
        annotation: The raw annotation from inspect

    Returns:
        JSON schema dict for the parameter
    """
    # Default schema
    schema = {"type": "string"}

    if type_hint is None and annotation != inspect.Parameter.empty:
        # Try to infer from annotation string
        if hasattr(annotation, '__name__'):
            type_name = annotation.__name__
        else:
            type_name = str(annotation)
    elif type_hint is not None:
        type_name = getattr(type_hint, '__name__', str(type_hint))
    else:
        return schema

    # Handle basic types
    if type_hint == int or type_name == 'int':
        schema = {"type": "integer"}
    elif type_hint == float or type_name == 'float':
        schema = {"type": "number"}
    elif type_hint == bool or type_name == 'bool':
        schema = {"type": "boolean"}
    elif type_hint == str or type_name == 'str':
        schema = {"type": "string"}
    elif type_hint == list or type_name == 'list':
        schema = {"type": "array", "items": {"type": "string"}}
    elif type_hint == dict or type_name == 'dict':
        schema = {"type": "object"}

    # Handle Union types (including Optional)
    elif hasattr(type_hint, '__origin__') and get_origin(type_hint) is Union:
        args = get_args(type_hint)
        # Check if it's Optional (Union with None)
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            schema = _type_to_json_schema(non_none_type, non_none_type)
        else:
            # Multiple types - default to string
            schema = {"type": "string"}

    # Handle List with type parameter
    elif hasattr(type_hint, '__origin__') and get_origin(type_hint) == list:
        args = get_args(type_hint)
        if args:
            item_schema = _type_to_json_schema(args[0], args[0])
            schema = {"type": "array", "items": item_schema}
        else:
            schema = {"type": "array", "items": {"type": "string"}}

    return schema

def normalize_messages(messages: Union[str, List[Union[str, dict, Message]]]) -> List[MessageDict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    result = []
    for msg in messages:
        if isinstance(msg, str):
            result.append({"role": "user", "content": msg})
        elif hasattr(msg, 'model_dump'):  # Pydantic Message
            result.append(msg.model_dump())
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append(dict(msg))
    return result

def normalize_tools(tools: Optional[List[Union[Callable, Tool, FunctionToolDict]]]) -> Mapping[str, Tool]:
    result = {}
    if not tools: return result

    for tool in tools:
        if callable(tool) and hasattr(tool, "to_json_schema") and hasattr(tool, "name"):
            result[tool.name] = tool
        elif callable(tool):
            tool_wrapper = FunctionToolWrapper(tool)
            result[tool_wrapper.name] = tool_wrapper
        elif isinstance(tool, Mapping):
            assert "name" in tool, "Tool dictionary must have 'name' field"
            result[tool["name"]] = tool
    return result

