from __future__ import annotations

import json as libJson
from typing import Any, Dict, List, Optional, Type, Union

from google import genai
from pydantic import BaseModel

from ..domain import (
    Content,
    GoogleProviderConfig,
    LLMResponse,
)
from ..protocol import (
    Event,
    LLMCallEvent,
    LLMCallEventPayload,
    Persona,
)
from ..utils import get_time_ms
from .base import BaseLLMProvider


class GoogleProvider(BaseLLMProvider):
    def __init__(
        self,
        config: GoogleProviderConfig,
        events: List[Event],
        persona: Optional[Persona],
    ):
        super().__init__(config, events, persona)
        self.client: genai.Client = genai.Client(api_key=self.api_key)

    def _prepare_contents(self, contents: List[Content]) -> List[genai.types.ContentDict]:
        prepared_contents: List[genai.types.ContentDict] = []
        for content in contents:
            prepared_contents.append({"role": content.role, "parts": [{"text": content.text}]})
        return prepared_contents

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None
    ) -> LLMResponse:
        prepared_contents = self._prepare_contents(contents)
        instructions_str = self._build_system_instruction(system_instruction)

        prepared_config = genai.types.GenerateContentConfig()
        if instructions_str:
            prepared_config.system_instruction = instructions_str

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model, contents=prepared_contents, config=prepared_config
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": instructions_str,
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
        schema: Union[Dict[str, Any], Type[BaseModel]],
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        prepared_contents = self._prepare_contents(contents)
        json_schema = self._parse_schema(schema)
        instructions_str = self._build_system_instruction(system_instruction)

        prepared_config = genai.types.GenerateContentConfig(
            response_json_schema=json_schema,
            response_mime_type="application/json",
        )
        if instructions_str:
            prepared_config.system_instruction = instructions_str

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model, contents=prepared_contents, config=prepared_config
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": instructions_str,
            "response_json_schema": json_schema,
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
