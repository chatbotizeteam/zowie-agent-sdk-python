from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .agent import Agent
from .context import Context
from .types import (
    AgentResponse,
    AgentResponseContinue,
    AgentResponseFinish,
    Content,
    GoogleConfig,
    LLMResponse,
    Metadata,
    OpenAIConfig,
)

__all__ = (
    "Agent",
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
)

_DIST_NAME = "zowie-agent-sdk"
try:
    __version__ = _version(_DIST_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
