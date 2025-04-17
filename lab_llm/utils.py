import tiktoken

def count_tokens(text, model="gpt-4"):
    # supported models: https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken/model.py#L24
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
