"""Internal SDK domain types.

These types are used internally by the SDK for configuration,
processing, and agent implementation. They are not part of the
external API contract.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# Agent response types (internal)
class ContinueConversationResponse(BaseModel):
    """Internal response to continue the conversation."""

    type: Literal["continue"] = "continue"
    message: str


class TransferToBlockResponse(BaseModel):
    """Internal response to transfer to another block."""

    type: Literal["finish"] = "finish"
    message: Optional[str] = None
    next_block: str


AgentResponse = Annotated[
    Union[ContinueConversationResponse, TransferToBlockResponse], Field(discriminator="type")
]


# LLM provider configuration


class OpenAIProviderConfig(BaseModel):
    """Configuration for OpenAI LLM provider."""

    provider: Literal["openai"] = "openai"
    api_key: str
    model: str


class GoogleProviderConfig(BaseModel):
    """Configuration for Google LLM provider."""

    provider: Literal["google"] = "google"
    api_key: str
    model: str


LLMConfig = Annotated[
    Union[OpenAIProviderConfig, GoogleProviderConfig], Field(discriminator="provider")
]


# Authentication configuration


class APIKeyAuth(BaseModel):
    """API key authentication configuration."""

    type: Literal["api_key"] = "api_key"
    header_name: str
    api_key: str


class BasicAuth(BaseModel):
    """Basic authentication configuration."""

    type: Literal["basic"] = "basic"
    username: str
    password: str


class BearerTokenAuth(BaseModel):
    """Bearer token authentication configuration."""

    type: Literal["bearer"] = "bearer"
    token: str


AuthConfig = Annotated[Union[APIKeyAuth, BasicAuth, BearerTokenAuth], Field(discriminator="type")]


# LLM interaction types
class LLMResponse(BaseModel):
    """Response from LLM providers."""

    text: str
    raw_response: Any
    provider: str
    model: str
