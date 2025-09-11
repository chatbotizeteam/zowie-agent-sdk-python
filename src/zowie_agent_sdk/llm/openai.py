from __future__ import annotations

import json as libJson
from typing import Any, Dict, List, Optional, Type, Union

import openai
from openai.types.chat import ChatCompletionMessageParam
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

    def _prepare_contents(self, contents: List[Content]) -> List[ChatCompletionMessageParam]:
        """Prepare contents for OpenAI chat completion format."""
        messages: List[ChatCompletionMessageParam] = []
        for content in contents:
            if content.role == "model":
                messages.append({
                    "role": "assistant",
                    "content": content.text,
                })
            else:
                messages.append({
                    "role": "user", 
                    "content": content.text,
                })
        return messages

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None
    ) -> LLMResponse:
        messages = self._prepare_contents(contents)
        instructions_str = self._build_system_instruction(system_instruction)

        # Add system message if we have instructions
        if instructions_str:
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": instructions_str,
            }
            messages = [system_message] + messages

        start = get_time_ms()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        stop = get_time_ms()

        prompt_data = {"messages": messages}

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
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                text = choice.message.content

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
        messages = self._prepare_contents(contents)
        instructions_str = self._build_system_instruction(system_instruction)

        # Add system message if we have instructions
        if instructions_str:
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": instructions_str,
            }
            messages = [system_message] + messages

        # Use native Pydantic support when possible
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return self._generate_with_pydantic_native(messages, schema)
        else:
            return self._generate_with_json_schema(messages, schema)

    def _generate_with_pydantic_native(
        self, messages: List[ChatCompletionMessageParam], schema: Type[BaseModel]
    ) -> LLMResponse:
        """Generate structured content using native Pydantic support."""
        start = get_time_ms()
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
        )
        stop = get_time_ms()

        prompt_data = {
            "messages": messages,
            "response_format": schema.__name__,
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
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                text = choice.message.content

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="openai",
            model=self.model,
        )

    def _generate_with_json_schema(
        self, 
        messages: List[ChatCompletionMessageParam], 
        schema: Union[Dict[str, Any], str]
    ) -> LLMResponse:
        """Generate structured content using JSON schema for dict/string inputs."""
        json_schema = self._parse_schema(schema)
        schema_name = json_schema.get("title", "structured_output")

        start = get_time_ms()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                }
            }
        )
        stop = get_time_ms()

        prompt_data = {
            "messages": messages,
            "response_json_schema": json_schema,
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
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                text = choice.message.content

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="openai",
            model=self.model,
        )
