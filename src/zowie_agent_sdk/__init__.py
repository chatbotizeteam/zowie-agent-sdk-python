from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .agent import Agent
from .context import Context
from .domain import (
    AgentResponse,
    APIKeyAuth,
    AuthConfig,
    BasicAuth,
    BearerTokenAuth,
    ContinueConversationResponse,
    GoogleProviderConfig,
    LLMConfig,
    OpenAIProviderConfig,
    TransferToBlockResponse,
)
from .protocol import (
    IncomingRequest,
    Message,
    Metadata,
    Persona,
)

__all__ = (
    "Agent",
    "Context",
    "ContinueConversationResponse",
    "TransferToBlockResponse",
    "AgentResponse",
    "APIKeyAuth",
    "BasicAuth",
    "BearerTokenAuth",
    "AuthConfig",
    "GoogleProviderConfig",
    "OpenAIProviderConfig",
    "LLMConfig",
    "IncomingRequest",
    "Message",
    "Metadata",
    "Persona",
    "__version__",
)

_DIST_NAME = "zowie-agent-sdk"
try:
    __version__ = _version(_DIST_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
