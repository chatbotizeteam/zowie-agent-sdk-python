from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import requests
from pydantic import BaseModel

from .domain import LLMResponse
from .http import HTTPClient
from .llm import LLM
from .protocol import Event, Message, Metadata, Persona

T = TypeVar("T", bound=BaseModel)


class Context:
    """Request context provided to agent implementations.

    Contains all data needed to process a request: conversation messages,
    metadata, persona, and pre-configured LLM and HTTP clients with
    automatic event tracking.
    """

    metadata: Metadata
    messages: List[Message]
    store_value: Callable[[str, Any], None]
    http: "ContextualHTTPClient"
    persona: Optional[Persona]
    context: Optional[str]
    events: List[Event]
    _base_llm: LLM
    _base_http: HTTPClient

    def __init__(
        self,
        metadata: Metadata,
        messages: List[Message],
        store_value: Callable[[str, Any], None],
        llm: LLM,
        http: HTTPClient,
        persona: Optional[Persona] = None,
        context: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> None:
        self.metadata = metadata
        self.messages = messages
        self.context = context
        self.store_value = store_value
        self._base_llm = llm
        self._base_http = http
        self.persona = persona
        self.context = context
        self.events = events if events is not None else []
        self.http = ContextualHTTPClient(self._base_http, self.events)

    @property
    def llm(self) -> "ContextualLLM":
        """LLM instance that automatically includes persona and context."""
        return ContextualLLM(self._base_llm, self.persona, self.context, self.events)


class ContextualLLM:
    """LLM wrapper that automatically passes persona/context to LLM calls."""

    def __init__(
        self,
        base_llm: LLM,
        persona: Optional[Persona],
        context: Optional[str],
        events: List[Event],
    ):
        self._base_llm = base_llm
        self._persona = persona
        self._context = context
        self._events = events

    def generate_content(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
    ) -> LLMResponse:
        return self._base_llm.generate_content(
            messages=messages,
            system_instruction=system_instruction,
            include_persona=include_persona,
            include_context=include_context,
            persona=self._persona,
            context=self._context,
            events=self._events,
        )

    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
    ) -> T:
        return self._base_llm.generate_structured_content(
            messages=messages,
            schema=schema,
            system_instruction=system_instruction,
            include_persona=include_persona,
            include_context=include_context,
            persona=self._persona,
            context=self._context,
            events=self._events,
        )


class ContextualHTTPClient:
    """HTTP wrapper that automatically passes events to HTTP calls for clean user API."""

    def __init__(self, base_http: HTTPClient, events: List[Event]):
        self._base_http = base_http
        self._events = events

    def get(
        self,
        url: str,
        headers: Dict[str, str],
        timeout_seconds: Optional[float] = None,
        include_headers: Optional[bool] = None,
    ) -> requests.Response:
        """Make a GET request with automatic event tracking."""
        return self._base_http.get(url, headers, self._events, timeout_seconds, include_headers)

    def post(
        self,
        url: str,
        json: Any,
        headers: Dict[str, str],
        timeout_seconds: Optional[float] = None,
        include_headers: Optional[bool] = None,
    ) -> requests.Response:
        """Make a POST request with automatic event tracking."""
        return self._base_http.post(
            url, json, headers, self._events, timeout_seconds, include_headers
        )

    def put(
        self,
        url: str,
        json: Any,
        headers: Dict[str, str],
        timeout_seconds: Optional[float] = None,
        include_headers: Optional[bool] = None,
    ) -> requests.Response:
        """Make a PUT request with automatic event tracking."""
        return self._base_http.put(
            url, json, headers, self._events, timeout_seconds, include_headers
        )

    def patch(
        self,
        url: str,
        json: Any,
        headers: Dict[str, str],
        timeout_seconds: Optional[float] = None,
        include_headers: Optional[bool] = None,
    ) -> requests.Response:
        """Make a PATCH request with automatic event tracking."""
        return self._base_http.patch(
            url, json, headers, self._events, timeout_seconds, include_headers
        )

    def delete(
        self,
        url: str,
        headers: Dict[str, str],
        timeout_seconds: Optional[float] = None,
        include_headers: Optional[bool] = None,
    ) -> requests.Response:
        """Make a DELETE request with automatic event tracking."""
        return self._base_http.delete(url, headers, self._events, timeout_seconds, include_headers)
