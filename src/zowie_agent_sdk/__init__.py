from importlib.metadata import PackageNotFoundError, version

from .core import (
    Agent,
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
    create_agent,
    start_agent,
)

__all__ = [
    "Agent",
    "create_agent",
    "Context",
    "AgentResponseContinue",
    "AgentResponseFinish",
    "AgentResponse",
    "GoogleConfig",
    "OpenAIConfig",
    "Content",
    "LLMResponse",
    "Metadata",
    # Deprecated but kept for backward compatibility
    "start_agent",
    "configure_llm",
    "__version__",
]

try:
    __version__ = version("zowie-agent-sdk")
except PackageNotFoundError:  # during local dev / editable installs
    __version__ = "0.0.0"