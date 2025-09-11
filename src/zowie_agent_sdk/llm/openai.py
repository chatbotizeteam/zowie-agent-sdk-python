from __future__ import annotations

import json as libJson
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast

import openai
from openai._types import NOT_GIVEN
from openai.types.responses import ResponseInputItemParam
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from pydantic import BaseModel

from ..domain import (
    Content,
    LLMResponse,
    OpenAIProviderConfig,
)
from ..protocol import (
    Event,
    LLMCallEvent,
    LLMCallEventPayload,
    Persona,
)
from ..utils import get_time_ms
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        config: OpenAIProviderConfig,
        events: List[Event],
        persona: Optional[Persona],
    ):
        super().__init__(config, events, persona)
        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key)

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None
    ) -> LLMResponse:
        instructions_str = self._build_persona_instruction()
        if system_instruction:
            instructions_str += system_instruction

        input_messages: List[ResponseInputItemParam] = []
        for content in contents:
            role = "assistant" if content.role == "model" else "user"
            message: EasyInputMessageParam = {
                "type": "message",
                "role": cast("Literal['user', 'assistant', 'system', 'developer']", role),
                "content": content.text,
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
        schema: Union[Dict[str, Any], str, Type[BaseModel]],
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        instructions_str = self._build_persona_instruction()
        if system_instruction:
            instructions_str += system_instruction

        input_messages: List[ResponseInputItemParam] = []
        for content in contents:
            role = "assistant" if content.role == "model" else "user"
            message: EasyInputMessageParam = {
                "type": "message",
                "role": cast("Literal['user', 'assistant', 'system', 'developer']", role),
                "content": content.text,
            }
            input_messages.append(message)

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
            schema_name = schema.__name__
        elif isinstance(schema, dict):
            json_schema = schema
            schema_name = json_schema.get("title", "structured_output")
        elif isinstance(schema, str):
            try:
                json_schema = libJson.loads(schema)
            except libJson.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema string: {e}") from e
            schema_name = json_schema.get("title", "structured_output")

        response_format_cfg: ResponseTextConfigParam = {
            "format": ResponseFormatTextJSONSchemaConfigParam(
                name=schema_name,
                type="json_schema",
                schema=json_schema,
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
            "response_json_schema": json_schema,
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
