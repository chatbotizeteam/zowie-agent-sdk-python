from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from ..domain import (
    GoogleProviderConfig,
    LLMConfig,
    LLMResponse,
    OpenAIProviderConfig,
)
from ..protocol import (
    Event,
    Message,
    Persona,
)

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    def __init__(
        self,
        config: Union[GoogleProviderConfig, OpenAIProviderConfig],
        events: List[Event],
        include_persona_default: bool = True,
        include_context_default: bool = True,
    ):
        self.model = config.model
        self.api_key = config.api_key
        self.events = events
        self.config = config
        self.include_persona_default = include_persona_default
        self.include_context_default = include_context_default

    @abstractmethod
    def generate_content(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context_data: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context_data: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> T:
        pass

    @abstractmethod
    def _prepare_messages(self, messages: List[Message]) -> Any:
        """Prepare messages for the specific provider's API format."""
        pass

    def _build_system_instruction(
        self,
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context_data: Optional[str] = None,
    ) -> str:
        """Build system instruction combining persona, instructions, and context.

        Args:
            system_instruction: Optional system instruction
            include_persona: Override for persona inclusion (None = use default)
            include_context: Override for context inclusion (None = use default)
            persona: Request-specific persona (overrides instance persona)
            context_data: Request-specific context (overrides instance context)
        """
        should_include_persona = (
            include_persona if include_persona is not None else self.include_persona_default
        )
        should_include_context = (
            include_context if include_context is not None else self.include_context_default
        )

        effective_persona = persona
        effective_context = context_data

        instructions_str = ""

        # Add persona if needed
        if should_include_persona and effective_persona:
            instructions_str = self._build_persona_instruction(effective_persona)

        # Add instructions tags only if content exists
        if system_instruction:
            if instructions_str:
                instructions_str += "\n\n"
            instructions_str += f"<instructions>\n{system_instruction}\n</instructions>"

        # Add context only if provided AND should include
        if effective_context and should_include_context:
            if instructions_str:
                instructions_str += "\n\n"
            instructions_str += f"<context>\n{effective_context}\n</context>"

        return instructions_str

    def _build_persona_instruction(self, persona: Optional[Persona] = None) -> str:
        if persona is None:
            return ""

        instruction = "<persona>\n"
        if persona.name:
            instruction += f"<name>{persona.name}</name>\n\n"
        if persona.business_context:
            instruction += (
                f"<business_context>\n{persona.business_context}" f"\n</business_context>\n\n"
            )
        if persona.tone_of_voice:
            instruction += f"<tone_of_voice>\n{persona.tone_of_voice}" f"\n</tone_of_voice>\n\n"
        instruction += "</persona>"
        return instruction


class LLM:
    def __init__(
        self,
        config: Optional[LLMConfig],
        events: List[Event],
        persona: Optional[Persona],
        context: Optional[str] = None,
        include_persona_default: bool = True,
        include_context_default: bool = True,
    ):
        self.provider: Optional[BaseLLMProvider] = None

        if config is None:
            return

        from .google import GoogleProvider
        from .openai import OpenAIProvider

        if isinstance(config, GoogleProviderConfig):
            self.provider = GoogleProvider(
                config=config,
                events=events,
                include_persona_default=include_persona_default,
                include_context_default=include_context_default,
            )
        elif isinstance(config, OpenAIProviderConfig):
            self.provider = OpenAIProvider(
                config=config,
                events=events,
                include_persona_default=include_persona_default,
                include_context_default=include_context_default,
            )

    def generate_content(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context_data: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_content(
            messages=messages,
            system_instruction=system_instruction,
            include_persona=include_persona,
            include_context=include_context,
            persona=persona,
            context_data=context_data,
            events=events,
        )

    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context_data: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> T:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_structured_content(
            messages=messages,
            schema=schema,
            system_instruction=system_instruction,
            include_persona=include_persona,
            include_context=include_context,
            persona=persona,
            context_data=context_data,
            events=events,
        )
