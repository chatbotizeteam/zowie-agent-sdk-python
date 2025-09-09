from __future__ import annotations

import json as libJson
from typing import Any, List, Optional

import openai
from openai._types import NOT_GIVEN
from openai.types.responses import ResponseInputItemParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses.response_text_config_param import ResponseTextConfigParam

from ..types import (
    Content,
    Event,
    LLMCallEvent,
    LLMCallEventPayload,
    LLMResponse,
    OpenAIConfig,
    Persona,
)
from ..utils import get_time_ms
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        config: OpenAIConfig,
        events: List[Event],
        persona: Optional[Persona],
    ):
        super().__init__(config, events, persona)
        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key)

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        # Build instructions from persona + optional system instruction
        instructions_str = self._build_persona_instruction()
        if system_instruction:
            instructions_str += system_instruction

        # Build Responses API input messages (typed)
        input_messages: List[ResponseInputItemParam] = []
        for content in contents:
            role = "assistant" if content.role == "model" else "user"
            message: ResponseInputItemParam = {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": content.text,
                    }
                ],
            }
            input_messages.append(message)

        start = get_time_ms()
        response = self.client.responses.create(
            model=self.model,
            input=input_messages if input_messages else NOT_GIVEN,
            instructions=instructions_str or NOT_GIVEN,
        )
        stop = get_time_ms()

        prompt_data = {"instructions": instructions_str, "input": input_messages}

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

        text = getattr(response, "output_text", "") or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="openai",
            model=self.model,
        )

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Build instructions from persona + optional system instruction
        instructions_str = self._build_persona_instruction()
        if system_instruction:
            instructions_str += system_instruction

        # Build Responses API input messages (typed)
        input_messages: List[ResponseInputItemParam] = []
        for content in contents:
            role = "assistant" if content.role == "model" else "user"
            message: ResponseInputItemParam = {
                "type": "message",
                "role": role,  # type: ignore[typeddict-item]
                "content": [
                    {
                        "type": "input_text",
                        "text": content.text,
                    }
                ],
            }
            input_messages.append(message)

        response_format_cfg: ResponseTextConfigParam = {
            "format": ResponseFormatTextJSONSchemaConfigParam(
                name="structured_output",
                type="json_schema",
                schema=schema,
                strict=True,
            )
        }

        start = get_time_ms()
        response = self.client.responses.create(
            model=self.model,
            input=input_messages if input_messages else NOT_GIVEN,
            instructions=instructions_str or NOT_GIVEN,
            text=response_format_cfg,
        )
        stop = get_time_ms()

        prompt_data = {
            "instructions": instructions_str,
            "response_json_schema": schema,
            "input": input_messages,
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

        text = getattr(response, "output_text", "") or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="openai",
            model=self.model,
        )
