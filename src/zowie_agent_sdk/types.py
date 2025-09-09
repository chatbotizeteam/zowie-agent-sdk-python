from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class AgentResponseContinue(BaseModel):
    type: Literal["continue"] = "continue"
    message: str


class AgentResponseFinish(BaseModel):
    type: Literal["finish"] = "finish"
    message: Optional[str] = None
    next_block: str


AgentResponse = Annotated[
    Union[AgentResponseContinue, AgentResponseFinish], Field(discriminator="type")
]


class OpenAIConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    api_key: str
    model: str


class GoogleConfig(BaseModel):
    provider: Literal["google"] = "google"
    api_key: str
    model: str


LLMConfig = Annotated[Union[OpenAIConfig, GoogleConfig], Field(discriminator="provider")]


class APIKeyAuth(BaseModel):
    type: Literal["api_key"] = "api_key"
    header_name: str
    api_key: str


class BasicAuth(BaseModel):
    type: Literal["basic"] = "basic"
    username: str
    password: str


class BearerTokenAuth(BaseModel):
    type: Literal["bearer"] = "bearer"
    token: str


AuthConfig = Annotated[Union[APIKeyAuth, BasicAuth, BearerTokenAuth], Field(discriminator="type")]


class LLMCallEventPayload(BaseModel):
    prompt: str
    response: str
    model: str
    durationInMillis: int


class APICallEventPayload(BaseModel):
    url: str
    requestHeaders: Dict[str, str]
    requestMethod: str
    requestBody: Optional[str]
    responseHeaders: Dict[str, str]
    responseStatusCode: int
    responseBody: Optional[str]
    durationInMillis: int


class LLMCallEvent(BaseModel):
    type: Literal["llm_call"] = "llm_call"
    payload: LLMCallEventPayload


class APICallEvent(BaseModel):
    type: Literal["api_call"] = "api_call"
    payload: APICallEventPayload


Event = Annotated[Union[LLMCallEvent, APICallEvent], Field(discriminator="type")]


class Persona(BaseModel):
    name: Optional[str]
    business_context: Optional[str]
    tone_of_voice: Optional[str]


class Content(BaseModel):
    text: str
    role: Literal["model", "user"]


class LLMResponse(BaseModel):
    text: str
    raw_response: Any
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class Metadata(BaseModel):
    requestId: str
    chatbotId: str
    conversationId: str
    interactionId: Optional[str] = None


class Message(BaseModel):
    author: str
    content: str
    timestamp: str


class SendMessagePayload(BaseModel):
    message: str


class GoToNextBlockPayload(BaseModel):
    message: Optional[str] = None
    nextBlockReferenceKey: str


class SendMessageCommand(BaseModel):
    type: Literal["send_message"] = "send_message"
    payload: SendMessagePayload


class GoToNextBlockCommand(BaseModel):
    type: Literal["go_to_next_block"] = "go_to_next_block"
    payload: GoToNextBlockPayload


Command = Annotated[
    Union[
        SendMessageCommand,
        GoToNextBlockCommand,
    ],
    Field(discriminator="type"),
]


class ExternalAgentResponse(BaseModel):
    command: Command
    valuesToSave: Optional[Dict[str, Any]] = None
    events: Optional[List[Event]] = None
