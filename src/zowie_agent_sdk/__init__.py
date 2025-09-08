from importlib.metadata import PackageNotFoundError, version

from .core import (
    AgentResponse,
    AgentResponseContinue,
    AgentResponseFinish,
    Content,
    Context,
    GoogleConfig,
    LLMResponse,
    Metadata,
    OpenAIConfig,
    configure_llm,
    start_agent,
)

__all__ = [
    "start_agent",
    "configure_llm",
    "Context",
    "AgentResponseContinue",
    "AgentResponseFinish",
    "AgentResponse",
    "GoogleConfig",
    "OpenAIConfig",
    "Content",
    "LLMResponse",
    "Metadata",
    "__version__",
]

try:
    __version__ = version("zowie-agent-sdk")
except PackageNotFoundError:  # during local dev / editable installs
    __version__ = "0.0.0"