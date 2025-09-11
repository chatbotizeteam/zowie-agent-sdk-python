from __future__ import annotations

import json as libJson
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..domain import (
    Content,
    GoogleProviderConfig,
    LLMConfig,
    LLMResponse,
    OpenAIProviderConfig,
)
from ..protocol import (
    Event,
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
        self, contents: List[Content], system_instruction: Optional[str] = None
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Union[Dict[str, Any], str, Type[BaseModel]],
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def _prepare_contents(self, contents: List[Content]) -> Any:
        """Prepare contents for the specific provider's API format.
        
        Each provider has different content format requirements:
        - Google: List[genai.types.ContentDict] 
        - OpenAI: List[ResponseInputItemParam]
        
        Returns:
            Provider-specific content format
        """
        pass

    def _build_system_instruction(self, system_instruction: Optional[str] = None) -> str:
        """Build system instruction combining persona and additional instructions."""
        instructions_str = self._build_persona_instruction()
        if system_instruction:
            instructions_str += system_instruction
        return instructions_str

    def _parse_schema(self, schema: Union[Dict[str, Any], str, Type[BaseModel]]) -> Dict[str, Any]:
        """Parse schema from various input formats to JSON schema dict."""
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        elif isinstance(schema, dict):
            return schema
        elif isinstance(schema, str):
            try:
                parsed_schema: Dict[str, Any] = libJson.loads(schema)
                return parsed_schema
            except libJson.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema string: {e}") from e
        else:
            raise ValueError(
                "Schema must be a Pydantic model class, dict, or a valid JSON schema string"
            )

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

        if isinstance(config, GoogleProviderConfig):
            self.provider = GoogleProvider(config=config, events=events, persona=persona)
        elif isinstance(config, OpenAIProviderConfig):
            self.provider = OpenAIProvider(config=config, events=events, persona=persona)

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_content(contents, system_instruction)

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Union[Dict[str, Any], str, Type[BaseModel]],
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_structured_content(contents, schema, system_instruction)
