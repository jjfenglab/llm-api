from warnings import deprecated

VERSA_API_VERSION = "2024-12-01-preview"
AWS_REGION = "us-west-2"


class VersaOpenAI:
    @property
    @deprecated("Use GPT4_O_2024_11 instead")
    def GPT4_O_2024_08(cls):
        return "azure/gpt-4o-2024-08-06"
    @property
    @deprecated("Use GPT4_1_MINI_2025_04 instead")
    def GPT4_O_MINI_2024_07(cls):
        return "azure/gpt-4o-mini-2024-07-18"
    GPT4_O_2024_11 = "azure/gpt-4o-2024-11-20"
    @property
    @deprecated("Use GPT4_1_MINI_2025_04 instead")
    def GPT4_O_MINI_2024_11(): 
        return "azure/gpt-4o-mini-2024-11-20"
    GPT4_1_MINI_2025_04 = "azure/gpt-4.1-mini-2025-04-14"

    GPT5_2025_08 = "azure/gpt-5-2025-08-07"
    GPT5_MINI_2025_08 = "azure/gpt-5-mini-2025-08-07"
    GPT5_NANO_2025_08 = "azure/gpt-5-nano-2025-08-07"
    GPT5_2_2025_12 = "azure/gpt-5.2-2025-12-11"
    O_1_2024_12 = "azure/o1-2024-12-17"
    O_4_MINI_2025_04 = "azure/o4-mini-2025-04-16"

class OpenAI:
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_O = "gpt-4o"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"


class Claude:
    HAIKU_4_5 = "anthropic/claude-haiku-4-5"
    SONNET_4 = "anthropic/claude-sonnet-4"
    OPUS_4_5 = "anthropic/claude-opus-4-5"
    SONNET_4_5 = "anthropic/claude-sonnet-4-5"
    HAIKU_4_6 = "anthropic/claude-haiku-4-5"
    SONNET_4_6 = "anthropic/claude-sonnet-4-6"
    OPUS_4_6 = "anthropic/claude-opus-4-6"

class VersaClaude:
    CLAUDE_SONNET_4 = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_OPUS_4_1 = "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_HAIKU_4_5 = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_OPUS_4_5 = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_SONNET_4_6 = "bedrock/us.anthropic.claude-sonnet-4-6"
    CLAUDE_OPUS_4_6 = "bedrock/us.anthropic.claude-opus-4-6-v1"

# Reasoning models (the GPT-5 family and the o-series) accept `reasoning_effort`,
# `verbosity`, and the Responses reasoning summary, and reject sampling params
# such as `temperature`. `is_reasoning_model` is the single source of truth for
# this distinction across the package.
#
# Classification is by model *family*: the litellm provider prefix (`azure/`,
# `openai/`, ...) and version suffix are ignored and the leading family token is
# matched, so new snapshots (e.g. `gpt-5-mini-2025-11-xx`) are recognized without
# editing a list. REASONING_MODELS is an explicit escape hatch for any reasoning
# model whose id does not follow the family naming; it is consulted as a fallback.
_REASONING_MODEL_FAMILIES = ("gpt-5", "o1", "o3", "o4")

REASONING_MODELS = {
    OpenAI.GPT5,
    OpenAI.GPT5_MINI,
    OpenAI.GPT5_NANO,
    VersaOpenAI.GPT5_2025_08,
    VersaOpenAI.GPT5_MINI_2025_08,
    VersaOpenAI.GPT5_NANO_2025_08,
    VersaOpenAI.GPT5_2_2025_12,
    VersaOpenAI.O_1_2024_12,
    VersaOpenAI.O_4_MINI_2025_04,
}
_EXPLICIT_REASONING_NAMES = {model.split("/")[-1].lower() for model in REASONING_MODELS}


def is_reasoning_model(model: str | None) -> bool:
    """Whether `model` names a reasoning model, ignoring provider prefix and version.

    Matches the model family (GPT-5, o-series) so new snapshots are recognized
    automatically, falling back to the explicit REASONING_MODELS set for ids that
    do not follow the family naming.
    """
    if not model:
        return False
    name = model.split("/")[-1].lower()
    return name.startswith(_REASONING_MODEL_FAMILIES) or name in _EXPLICIT_REASONING_NAMES
