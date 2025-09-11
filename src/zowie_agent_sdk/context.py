from __future__ import annotations

from typing import Any, Callable, List, Optional

from .http import HTTPClient
from .llm import LLM
from .protocol import Message, Metadata


class Context:
    metadata: Metadata
    messages: List[Message]
    context: Optional[str]
    store_value: Callable[[str, Any], None]
    llm: LLM
    http: HTTPClient

    def __init__(
        self,
        metadata: Metadata,
        messages: List[Message],
        context: Optional[str],
        store_value: Callable[[str, Any], None],
        llm: LLM,
        http: HTTPClient,
    ) -> None:
        self.metadata = metadata
        self.messages = messages
        self.context = context
        self.store_value = store_value
        self.llm = llm
        self.http = http
