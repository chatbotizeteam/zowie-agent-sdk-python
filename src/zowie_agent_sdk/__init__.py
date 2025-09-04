from importlib.metadata import version, PackageNotFoundError

from .core import start_agent, Context, AgentResponseContinue, AgentResponseFinish, AgentResponse, configure_llm, GoogleConfig, OpenAIConfig, Content

__all__ = ["start_agent", "configure_llm",  "Context", "AgentResponseContinue", "AgentResponseFinish", "AgentResponse", "GoogleConfig", "OpenAIConfig", "Content", "__version__"]

try:
    __version__ = version("zowie-agent-sdk")
except PackageNotFoundError:  # during local dev / editable installs
    __version__ = "0.0.0"
