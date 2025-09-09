from __future__ import annotations

import json as libJson
from typing import Any, List, Optional

from google import genai

from ..types import (
    Content,
    Event,
    GoogleConfig,
    LLMCallEvent,
    LLMCallEventPayload,
    LLMResponse,
    Persona,
)
from ..utils import get_time_ms
from .base import BaseLLMProvider


class GoogleProvider(BaseLLMProvider):
    def __init__(
        self,
        config: GoogleConfig,
        events: List[Event],
        persona: Optional[Persona],
    ):
        super().__init__(config, events, persona)
        self.client: genai.Client = genai.Client(api_key=self.api_key)

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        prepared_contents = []
        for content in contents:
            prepared_contents.append({"role": content.role, "parts": [{"text": content.text}]})

        prepared_config = genai.types.GenerateContentConfig()

        system_instruction_temp = self._build_persona_instruction()
        if system_instruction:
            system_instruction_temp += system_instruction

        if system_instruction_temp:
            prepared_config.system_instruction = system_instruction_temp

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model, contents=prepared_contents, config=prepared_config
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": system_instruction_temp,
            "contents": prepared_contents,
        }

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(prompt_data, indent=2, sort_keys=True, ensure_ascii=False),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="google",
            model=self.model,
        )

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        prepared_contents = []
        for content in contents:
            prepared_contents.append({"role": content.role, "parts": [{"text": content.text}]})

        prepared_config = genai.types.GenerateContentConfig(
            response_json_schema=schema,
            response_mime_type="application/json",
        )

        system_instruction_temp = self._build_persona_instruction()
        if system_instruction:
            system_instruction_temp += system_instruction

        if system_instruction_temp:
            prepared_config.system_instruction = system_instruction_temp

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model, contents=prepared_contents, config=prepared_config
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": system_instruction_temp,
            "response_json_schema": schema,
            "contents": prepared_contents,
        }

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(prompt_data, indent=2, sort_keys=True, ensure_ascii=False),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="google",
            model=self.model,
        )
