from __future__ import annotations

import json as libJson
import logging
from typing import List, Optional, Type, TypeVar

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..domain import OpenAIProviderConfig
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


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        config: OpenAIProviderConfig,
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
        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger("zowie_agent.OpenAIProvider")

    def _prepare_messages(self, messages: List[Message]) -> List[ChatCompletionMessageParam]:
        """Convert Message objects to OpenAI chat completion format."""
        openai_messages: List[ChatCompletionMessageParam] = []
        for message in messages:
            # Map Message.author to OpenAI's role format
            if message.author == "User":
                role = "user"
            elif message.author == "Chatbot":
                role = "assistant"  # OpenAI uses "assistant" for bot responses
            else:
                role = "user"  # Default fallback

            message_param: ChatCompletionMessageParam = {
                "role": role,  # pyright: ignore
                "content": message.content,
            }
            openai_messages.append(message_param)
        return openai_messages

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
        openai_messages = self._prepare_messages(messages)
        instructions_str = self._build_system_instruction(
            system_instruction, include_persona, include_context, persona, context
        )

        events = events if events is not None else self.events
        if instructions_str:
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": instructions_str,
            }
            openai_messages = [system_message] + openai_messages

        self.logger.debug(f"Making OpenAI LLM request with model {self.model}")

        start = get_time_ms()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
            )
            stop = get_time_ms()
            duration = stop - start

            self.logger.debug(
                f"OpenAI LLM request completed in {duration}ms with model {self.model}"
            )
        except Exception as e:
            stop = get_time_ms()
            duration = stop - start
            self.logger.error(f"OpenAI LLM request failed after {duration}ms: {str(e)}")
            raise

        prompt_data = {
            "messages": [msg.model_dump(mode="json") for msg in messages],
            "system_instruction": instructions_str,
        }

        events.append(
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
        openai_messages = self._prepare_messages(messages)
        instructions_str = self._build_system_instruction(
            system_instruction, include_persona, include_context, persona, context
        )

        events = events if events is not None else self.events
        if instructions_str:
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": instructions_str,
            }
            openai_messages = [system_message] + openai_messages

        # Use OpenAI's native Pydantic support
        start = get_time_ms()
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=openai_messages,
            response_format=schema,
        )
        stop = get_time_ms()

        prompt_data = {
            "messages": [msg.model_dump(mode="json") for msg in messages],
            "system_instruction": instructions_str,
            "response_schema": schema.model_json_schema(),
        }

        events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(prompt_data, indent=2, sort_keys=True, ensure_ascii=False),
                    response=libJson.dumps(
                        response.choices[0].message.parsed.model_dump()
                        if isinstance(response.choices[0].message.parsed, BaseModel)
                        else response.choices[0].message.parsed,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    durationInMillis=stop - start,
                )
            )
        )

        # Return the instantiated model using OpenAI's native parsing
        parsed_result = response.choices[0].message.parsed
        if not isinstance(parsed_result, schema):
            raise ValueError(f"Expected {schema.__name__} instance, got {type(parsed_result)}")
        return parsed_result
