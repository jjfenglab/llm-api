from typing_extensions import deprecated

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

    GPT5_4_MINI_2026_03 = "azure/gpt-5.4-mini-2026-03-17"
    GPT5_4_NANO_2026_03 = "azure/gpt-5.4-nano-2026-03-17"
    GPT5_5_2026_04 = "azure/gpt-5.5-2026-04-24"

class OpenAI:
    GPT4_O_MINI = "gpt-4o-mini"
    GPT4_O = "gpt-4o"
    GPT5 = "gpt-5"
    GPT5_MINI = "gpt-5-mini"
    GPT5_NANO = "gpt-5-nano"
    GPT5_4_MINI = "gpt-5.4-mini"
    GPT5_4 = "gpt-5.4"
    GPT5_5 = "gpt-5.5"


class Claude:
    HAIKU_4_5 = "anthropic/claude-haiku-4-5"
    SONNET_4 = "anthropic/claude-sonnet-4"
    OPUS_4_5 = "anthropic/claude-opus-4-5"
    SONNET_4_5 = "anthropic/claude-sonnet-4-5"
    HAIKU_4_6 = "anthropic/claude-haiku-4-5"
    SONNET_4_6 = "anthropic/claude-sonnet-4-6"
    OPUS_4_6 = "anthropic/claude-opus-4-6"
    FABLE_5 = "anthropic/claude-fable-5"
    OPUS_4_8 = "anthropic/claude-opus-4-8"

class VersaClaude:
    @property
    @deprecated("Use CLAUDE_SONNET_4_6 instead")
    def CLAUDE_SONNET_4(cls):
        return "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_OPUS_4_1 = "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_HAIKU_4_5 = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_OPUS_4_5 = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_SONNET_4_6 = "bedrock/us.anthropic.claude-sonnet-4-6"
    CLAUDE_OPUS_4_6 = "bedrock/us.anthropic.claude-opus-4-6-v1"
    CLAUDE_OPUS_4_8 = "bedrock/us.anthropic.claude-opus-4-8"

# Reasoning models support reasoning_effort and verbosity parameters
REASONING_MODELS = {
    OpenAI.GPT5,
    OpenAI.GPT5_MINI,
    OpenAI.GPT5_NANO,
    OpenAI.GPT5_4_MINI,
    OpenAI.GPT5_4,
    OpenAI.GPT5_5,
    VersaOpenAI.GPT5_2025_08,
    VersaOpenAI.GPT5_MINI_2025_08,
    VersaOpenAI.GPT5_NANO_2025_08,
    VersaOpenAI.GPT5_5_2026_04,
    VersaOpenAI.GPT5_4_MINI_2026_03,
    VersaOpenAI.GPT5_4_NANO_2026_03,
    VersaOpenAI.O_1_2024_12,
    VersaOpenAI.O_4_MINI_2025_04
}
is_reasoning_model = lambda x: x in REASONING_MODELS
