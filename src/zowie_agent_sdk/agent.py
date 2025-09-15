from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Request

from .auth import AuthValidator
from .context import Context
from .domain import (
    AgentResponse,
    AuthConfig,
    ContinueConversationResponse,
    LLMConfig,
    TransferToBlockResponse,
)
from .http import HTTPClient
from .llm import LLM
from .protocol import (
    Event,
    ExternalAgentResponse,
    GoToNextBlockCommand,
    GoToNextBlockPayload,
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
        include_persona_by_default: bool = True,
        include_http_headers_by_default: bool = True,
        log_level: str = "INFO",
    ):
        self.llm_config = llm_config
        self.http_timeout_seconds = http_timeout_seconds
        self.include_persona_by_default = include_persona_by_default
        self.include_http_headers_by_default = include_http_headers_by_default
        self.auth_validator = AuthValidator(auth_config)
        
        # Setup logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(f"zowie_agent.{self.__class__.__name__}")
        
        self.app = FastAPI()
        self._setup_routes()
        
        self.logger.info(f"Agent {self.__class__.__name__} initialized")

    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration for the agent."""
        # Only configure logging if it hasn't been configured yet
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

    @abstractmethod
    def handle(self, context: Context) -> AgentResponse:
        """Override this method to implement your agent's logic"""
        pass

    def _setup_routes(self) -> None:
        def auth_dependency(request: Request) -> None:
            return self.auth_validator(request)

        @self.app.get("/health")
        def health_check() -> Dict[str, Any]:
            """Health check endpoint to verify agent is running."""
            from .utils import get_time_ms
            return {
                "status": "healthy",
                "agent": self.__class__.__name__,
                "timestamp": get_time_ms()
            }

        @self.app.post("/")
        def handle_request(
            input_json: Dict[str, Any], _: None = Depends(auth_dependency)
        ) -> ExternalAgentResponse:
            request_id = input_json.get("metadata", {}).get("requestId", "unknown")
            self.logger.info(f"Processing request {request_id}")
            
            valueStorage: Dict[str, Any] = {}
            events: List[Event] = []

            def storeValue(key: str, value: Any) -> None:
                valueStorage[key] = value

            try:
                metadata = Metadata(**input_json["metadata"])

                persona = None
                if "persona" in input_json and input_json["persona"] is not None:
                    persona = Persona(**input_json["persona"])

                llm = LLM(
                    config=self.llm_config, 
                    events=events, 
                    persona=persona,
                    agent_include_persona_default=self.include_persona_by_default
                )
                if self.http_timeout_seconds is None:
                    http_client = HTTPClient(
                        events=events, 
                        include_headers_by_default=self.include_http_headers_by_default
                    )
                else:
                    http_client = HTTPClient(
                        events=events, 
                        default_timeout_seconds=self.http_timeout_seconds,
                        include_headers_by_default=self.include_http_headers_by_default
                    )

                from .protocol import Message

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

                if isinstance(result, ContinueConversationResponse):
                    response = ExternalAgentResponse(
                        command=SendMessageCommand(payload=SendMessagePayload(message=result.message)),
                        valuesToSave=valueStorage if valueStorage else None,
                        events=events if events else None,
                    )
                elif isinstance(result, TransferToBlockResponse):
                    payload = GoToNextBlockPayload(
                        nextBlockReferenceKey=result.next_block,
                        message=result.message,
                    )
                    response = ExternalAgentResponse(
                        command=GoToNextBlockCommand(payload=payload),
                        valuesToSave=valueStorage if valueStorage else None,
                        events=events if events else None,
                    )

                self.logger.info(f"Request {request_id} processed successfully")
                return response
                
            except Exception as e:
                self.logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
                raise
