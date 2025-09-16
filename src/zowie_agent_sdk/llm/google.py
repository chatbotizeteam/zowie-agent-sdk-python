from __future__ import annotations

import json as libJson
import logging
from typing import List, Optional, Type, TypeVar

from google import genai
from pydantic import BaseModel

from ..domain import GoogleProviderConfig
from ..protocol import (
    Event,
    LLMCallEvent,
    LLMCallEventPayload,
    Message,
    Persona,
)
from ..utils import get_time_ms
from .base import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class GoogleProvider(BaseLLMProvider):
    def __init__(
        self,
        config: GoogleProviderConfig,
        events: List[Event],
        include_persona_default: bool = True,
        include_context_default: bool = True,
    ):
        super().__init__(
            config,
            events,
            include_persona_default,
            include_context_default,
        )
        self.client: genai.Client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger("zowie_agent.GoogleProvider")

    def _prepare_messages(self, messages: List[Message]) -> List[genai.types.ContentDict]:
        """Convert Message objects to Google's ContentDict format."""
        prepared_contents: List[genai.types.ContentDict] = []
        for message in messages:
            # Map Message.author to Google's role format
            if message.author == "User":
                role = "user"
            elif message.author == "Chatbot":
                role = "model"  # Google uses "model" for assistant responses
            else:
                role = "user"  # Default fallback

            prepared_contents.append({"role": role, "parts": [{"text": message.content}]})
        return prepared_contents

    def generate_content(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> str:
        prepared_contents = self._prepare_messages(messages)
        instructions_str = self._build_system_instruction(
            system_instruction, include_persona, include_context, persona, context
        )

        events = events if events is not None else self.events

        self.logger.debug(f"Making Google LLM request with model {self.model}")

        prepared_config = genai.types.GenerateContentConfig(
            system_instruction=instructions_str,
        )

        start = get_time_ms()
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prepared_contents, config=prepared_config
            )
            stop = get_time_ms()
            duration = stop - start

            self.logger.debug(
                f"Google LLM request completed in {duration}ms with model {self.model}"
            )
        except Exception as e:
            stop = get_time_ms()
            duration = stop - start
            self.logger.error(f"Google LLM request failed after {duration}ms: {str(e)}")
            raise

        prompt_data = {
            "messages": [msg.model_dump() for msg in messages],
            "system_instruction": instructions_str,
        }

        text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(prompt_data, indent=2, sort_keys=True, ensure_ascii=False),
                    response=text,
                    durationInMillis=stop - start,
                )
            )
        )

        return text

    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> T:
        prepared_contents = self._prepare_messages(messages)
        instructions_str = self._build_system_instruction(
            system_instruction, include_persona, include_context, persona, context
        )

        events = events if events is not None else self.events
        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model,
            contents=prepared_contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "system_instruction": instructions_str if instructions_str else None,
            },
        )
        stop = get_time_ms()

        prompt_data = {
            "messages": [msg.model_dump() for msg in messages],
            "system_instruction": instructions_str,
            "response_schema": schema.model_json_schema(),
        }

        events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(prompt_data, indent=2, sort_keys=True, ensure_ascii=False),
                    response=libJson.dumps(
                        response.parsed.model_dump()
                        if isinstance(response.parsed, BaseModel)
                        else response.parsed,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    durationInMillis=stop - start,
                )
            )
        )

        # Return the instantiated model using Google's native parsing
        parsed_result = response.parsed
        if not isinstance(parsed_result, schema):
            raise ValueError(f"Expected {schema.__name__} instance, got {type(parsed_result)}")
        return parsed_result
