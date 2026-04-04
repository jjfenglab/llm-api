"""
Tools for creating function tools from Python functions using inspection.
(Generated with Claude Code)
"""

import inspect
from typing import Any, Callable, get_type_hints, get_origin, get_args, Union, Annotated, Optional, List, Union
from .types import FunctionDefinition, FunctionTool
from litellm import Message

def make_function_tool(func: Callable) -> FunctionTool:
    """
    Create a FunctionTool from a Python function using inspection.

    Args:
        func: The Python function to convert to a FunctionTool

    Returns:
        FunctionTool dictionary with name, description, and parameter schema
    """
    # Get function name
    name = func.__name__

    # Get description from docstring
    description = inspect.getdoc(func) or f"Function {name}"

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
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": False
    }

    # Create the complete FunctionTool
    tool: FunctionTool = {
        "type": "function",
        "function": function_def
    }

    return tool


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

def normalize_messages(messages: Union[str, List[Union[str, dict, Message]]]) -> List[dict]:
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

def normalize_tools(tools: Optional[List[Union[Callable, FunctionTool]]]) -> Optional[List[FunctionTool]]:
    if not tools:
        return None

    result = []
    for tool in tools:
        if callable(tool) and not isinstance(tool, dict):
            result.append(make_function_tool(tool))
        else:
            result.append(tool)
    return result

