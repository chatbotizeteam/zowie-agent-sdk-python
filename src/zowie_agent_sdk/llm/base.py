from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

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


class BaseLLMProvider(ABC):
    def __init__(
        self,
        config: Union[GoogleProviderConfig, OpenAIProviderConfig],
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
        self, messages: List[Message], system_instruction: str, include_persona: Optional[bool] = None, agent_include_persona_default: bool = True
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: str,
        include_persona: Optional[bool] = None,
        agent_include_persona_default: bool = True,
    ) -> T:
        pass

    @abstractmethod
    def _prepare_messages(self, messages: List[Message]) -> Any:
        """Prepare messages for the specific provider's API format.

        Each provider has different message format requirements:
        - Google: List[genai.types.ContentDict]
        - OpenAI: List[dict] with role/content

        Returns:
            Provider-specific message format
        """
        pass

    def _build_system_instruction(self, system_instruction: str, include_persona: Optional[bool] = None, agent_include_persona_default: bool = True) -> str:
        """Build system instruction combining persona and additional instructions.
        
        Args:
            system_instruction: The required system instruction
            include_persona: Override for persona inclusion (None = use agent default)
            agent_include_persona_default: Agent's default persona inclusion setting
        """
        should_include_persona = include_persona if include_persona is not None else agent_include_persona_default
        
        instructions_str = ""
        if should_include_persona:
            instructions_str = self._build_persona_instruction()
        
        instructions_str += system_instruction
        return instructions_str


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
        agent_include_persona_default: bool = True,
    ):
        self.provider: Optional[BaseLLMProvider] = None
        self.agent_include_persona_default = agent_include_persona_default

        if config is None:
            return

        from .google import GoogleProvider
        from .openai import OpenAIProvider

        if isinstance(config, GoogleProviderConfig):
            self.provider = GoogleProvider(config=config, events=events, persona=persona)
        elif isinstance(config, OpenAIProviderConfig):
            self.provider = OpenAIProvider(config=config, events=events, persona=persona)

    def generate_content(
        self, messages: List[Message], system_instruction: str, include_persona: Optional[bool] = None
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_content(messages, system_instruction, include_persona, self.agent_include_persona_default)

    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: str,
        include_persona: Optional[bool] = None,
    ) -> T:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_structured_content(messages, schema, system_instruction, include_persona, self.agent_include_persona_default)
