from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from ..types import Content, Event, GoogleConfig, LLMConfig, LLMResponse, OpenAIConfig, Persona


class BaseLLMProvider(ABC):
    def __init__(
        self,
        config: Union[GoogleConfig, OpenAIConfig],
        events: List[Event],
        persona: Optional[Persona],
    ):
        self.model = config.model
        self.api_key = config.api_key
        self.events = events
        self.persona = persona
        self.config = config

    @abstractmethod
    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        pass

    def _build_persona_instruction(self) -> str:
        if self.persona is None:
            return ""

        instruction = "<persona>\n"
        if self.persona.name:
            instruction += f"<name>{self.persona.name}</name>\n\n"
        if self.persona.business_context:
            instruction += (
                f"<business_context>\n{self.persona.business_context}" f"\n</business_context>\n\n"
            )
        if self.persona.tone_of_voice:
            instruction += (
                f"<tone_of_voice>\n{self.persona.tone_of_voice}" f"\n</tone_of_voice>\n\n"
            )
        instruction += "</persona>"
        return instruction


class LLM:
    def __init__(
        self,
        config: Optional[LLMConfig],
        events: List[Event],
        persona: Optional[Persona],
    ):
        self.provider: Optional[BaseLLMProvider] = None

        if config is None:
            return

        from .google import GoogleProvider
        from .openai import OpenAIProvider

        # Use provider attribute for reliable type checking to avoid module path issues
        # if hasattr(config, 'provider') and config.provider == 'google':
        #     self.provider = GoogleProvider(config=config, events=events, persona=persona)
        # elif hasattr(config, 'provider') and config.provider == 'openai':
        #     self.provider = OpenAIProvider(config=config, events=events, persona=persona)
        if isinstance(config, GoogleConfig):
            self.provider = GoogleProvider(config=config, events=events, persona=persona)
        elif isinstance(config, OpenAIConfig):
            self.provider = OpenAIProvider(config=config, events=events, persona=persona)

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_content(contents, system_instruction, **kwargs)

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_structured_content(
            contents, schema, system_instruction, **kwargs
        )
