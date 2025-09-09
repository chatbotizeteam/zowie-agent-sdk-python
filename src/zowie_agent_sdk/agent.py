from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Request

from .auth import AuthValidator
from .context import Context
from .http import HTTPClient
from .llm import LLM
from .types import (
    AgentResponse,
    AgentResponseContinue,
    AgentResponseFinish,
    AuthConfig,
    Event,
    ExternalAgentResponse,
    GoToNextBlockCommand,
    GoToNextBlockPayload,
    LLMConfig,
    Metadata,
    Persona,
    SendMessageCommand,
    SendMessagePayload,
)


class Agent(ABC):
    def __init__(
        self,
        llm_config: LLMConfig,
        http_timeout_seconds: Optional[float] = None,
        auth_config: Optional[AuthConfig] = None,
    ):
        self.llm_config = llm_config
        self.http_timeout_seconds = http_timeout_seconds
        self.auth_validator = AuthValidator(auth_config)
        self.app = FastAPI()
        self._setup_routes()

    @abstractmethod
    def handle(self, context: Context) -> AgentResponse:
        """Override this method to implement your agent's logic"""
        pass

    def _setup_routes(self) -> None:
        def auth_dependency(request: Request) -> None:
            return self.auth_validator(request)
        
        @self.app.post("/")
        def handle_request(
            input_json: Dict[str, Any], _: None = Depends(auth_dependency)
        ) -> ExternalAgentResponse:
            valueStorage: Dict[str, Any] = {}
            events: List[Event] = []

            def storeValue(key: str, value: Any) -> None:
                valueStorage[key] = value

            metadata = Metadata(
                requestId=input_json["metadata"]["requestId"],
                chatbotId=input_json["metadata"]["chatbotId"],
                conversationId=input_json["metadata"]["conversationId"],
                interactionId=input_json["metadata"].get("interactionId"),
            )

            persona = None
            if "persona" in input_json and input_json["persona"] is not None:
                persona = Persona(name=None, business_context=None, tone_of_voice=None)
                if input_json["persona"].get("name") is not None:
                    persona.name = input_json["persona"]["name"]
                if input_json["persona"].get("businessContext") is not None:
                    persona.business_context = input_json["persona"]["businessContext"]
                if input_json["persona"].get("toneOfVoice") is not None:
                    persona.tone_of_voice = input_json["persona"]["toneOfVoice"]

            llm = LLM(config=self.llm_config, events=events, persona=persona)
            if self.http_timeout_seconds is None:
                http_client = HTTPClient(events=events)
            else:
                http_client = HTTPClient(
                    events=events, default_timeout_seconds=self.http_timeout_seconds
                )

            from .types import Message

            messages = [Message(**msg) for msg in input_json["messages"]]

            context = Context(
                metadata=metadata,
                messages=messages,
                context=input_json.get("context"),
                store_value=storeValue,
                llm=llm,
                http=http_client,
            )

            result = self.handle(context)

            if isinstance(result, AgentResponseContinue):
                response = ExternalAgentResponse(
                    command=SendMessageCommand(payload=SendMessagePayload(message=result.message)),
                    valuesToSave=valueStorage if valueStorage else None,
                    events=events if events else None,
                )
            elif isinstance(result, AgentResponseFinish):
                payload = GoToNextBlockPayload(
                    nextBlockReferenceKey=result.next_block,
                    message=result.message,
                )
                response = ExternalAgentResponse(
                    command=GoToNextBlockCommand(payload=payload),
                    valuesToSave=valueStorage if valueStorage else None,
                    events=events if events else None,
                )

            return response
