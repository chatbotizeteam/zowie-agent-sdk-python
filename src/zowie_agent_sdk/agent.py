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
    IncomingRequest,
    SendMessageCommand,
    SendMessagePayload,
)


class Agent(ABC):
    """Base class for Zowie agents.

    Agents handle incoming requests from Zowie's Decision Engine, process them using
    LLMs and external APIs, and return responses to continue conversations or
    transfer to other workflow blocks.

    Example:
        class MyAgent(Agent):
            def handle(self, context: Context) -> AgentResponse:
                response = context.llm.generate_content(
                    messages=context.messages,
                    system_instruction="You are a helpful assistant"
                )
                return ContinueConversationResponse(message=response.text)
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        http_timeout_seconds: Optional[float] = None,
        auth_config: Optional[AuthConfig] = None,
        include_persona_by_default: bool = True,
        include_context_by_default: bool = True,
        include_http_headers_by_default: bool = True,
        include_request_bodies_in_events_by_default: bool = True,
        log_level: str = "INFO",
    ):
        self.llm_config = llm_config
        self.http_timeout_seconds = http_timeout_seconds
        self.include_persona_by_default = include_persona_by_default
        self.include_context_by_default = include_context_by_default
        self.include_http_headers_by_default = include_http_headers_by_default
        self.include_request_bodies_in_events_by_default = (
            include_request_bodies_in_events_by_default
        )
        self.auth_validator = AuthValidator(auth_config)

        self._base_llm = LLM(
            config=self.llm_config,
            events=[],
            persona=None,
            context=None,
            include_persona_default=self.include_persona_by_default,
            include_context_default=self.include_context_by_default,
        )

        if self.http_timeout_seconds is None:
            self._base_http_client = HTTPClient(
                include_headers_by_default=self.include_http_headers_by_default,
                include_request_bodies_by_default=self.include_request_bodies_in_events_by_default,
            )
        else:
            self._base_http_client = HTTPClient(
                default_timeout_seconds=self.http_timeout_seconds,
                include_headers_by_default=self.include_http_headers_by_default,
                include_request_bodies_by_default=self.include_request_bodies_in_events_by_default,
            )

        self._setup_logging(log_level)
        self.logger = logging.getLogger(f"zowie_agent.{self.__class__.__name__}")

        self.app = FastAPI()
        self._setup_routes()

        self.logger.info(f"Agent {self.__class__.__name__} initialized")

    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration for the agent."""
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
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
                "timestamp": get_time_ms(),
            }

        @self.app.post("/")
        def handle_request(
            input_json: Dict[str, Any], _: None = Depends(auth_dependency)
        ) -> ExternalAgentResponse:
            request_id = "unknown"
            try:
                # Validate the complete incoming request
                request = IncomingRequest(**input_json)

                request_id = request.metadata.requestId
                self.logger.info(f"Processing request {request_id}")

                valueStorage: Dict[str, Any] = {}
                events: List[Event] = []

                def storeValue(key: str, value: Any) -> None:
                    valueStorage[key] = value

                context = Context(
                    metadata=request.metadata,
                    messages=request.messages,
                    store_value=storeValue,
                    llm=self._base_llm,
                    http=self._base_http_client,
                    persona=request.persona,
                    context=request.context,
                    events=events,
                )

                result = self.handle(context)

                if isinstance(result, ContinueConversationResponse):
                    response = ExternalAgentResponse(
                        command=SendMessageCommand(
                            payload=SendMessagePayload(message=result.message)
                        ),
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
