"""Tests for LLM base class."""

from unittest.mock import MagicMock, patch

import pytest

from zowie_agent_sdk import GoogleProviderConfig
from zowie_agent_sdk.domain import LLMResponse
from zowie_agent_sdk.llm import LLM
from zowie_agent_sdk.llm.base import BaseLLMProvider
from zowie_agent_sdk.protocol import Message


class TestLLMBase:
    """Test LLM base class."""

    def test_llm_initialization_with_none_config(self):
        """Test LLM initialization with None config."""
        events = []
        llm = LLM(config=None, events=events, persona=None)
        
        assert llm.provider is None

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_llm_initialization_with_google_config(
        self, mock_google_provider, google_config, sample_persona
    ):
        """Test LLM initialization with Google config."""
        events = []
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=sample_persona)
        
        mock_google_provider.assert_called_once_with(
            config=google_config,
            events=events,
            persona=sample_persona
        )
        assert llm.provider == mock_provider_instance

    @patch('zowie_agent_sdk.llm.openai.OpenAIProvider')
    def test_llm_initialization_with_openai_config(self, mock_openai_provider, openai_config):
        """Test LLM initialization with OpenAI config."""
        events = []
        mock_provider_instance = MagicMock()
        mock_openai_provider.return_value = mock_provider_instance
        
        llm = LLM(config=openai_config, events=events, persona=None)
        
        mock_openai_provider.assert_called_once_with(
            config=openai_config,
            events=events,
            persona=None
        )
        assert llm.provider == mock_provider_instance

    def test_generate_content_without_provider(self):
        """Test generate_content raises exception when provider is not configured."""
        events = []
        llm = LLM(config=None, events=events, persona=None)
        
        messages = [Message(author="User", content="Hello", timestamp="2024-01-15T10:00:00Z")]
        
        with pytest.raises(Exception, match="LLM provider not configured"):
            llm.generate_content(messages, "Test instruction")

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_generate_content_with_provider(self, mock_google_provider, google_config):
        """Test generate_content delegates to provider."""
        events = []
        mock_provider_instance = MagicMock()
        mock_response = LLMResponse(
            text="Test response",
            raw_response={},
            provider="google",
            model="gemini-2.0-flash",
        )
        mock_provider_instance.generate_content.return_value = mock_response
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        messages = [Message(author="User", content="Hello", timestamp="2024-01-15T10:00:00Z")]
        
        response = llm.generate_content(messages, system_instruction="Be helpful")
        
        assert response == mock_response
        mock_provider_instance.generate_content.assert_called_once_with(
            messages,
            "Be helpful",
            None,
            True
        )

    def test_generate_structured_content_without_provider(self):
        """Test generate_structured_content raises exception when provider is not configured."""
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            test: str
            
        events = []
        llm = LLM(config=None, events=events, persona=None)
        
        messages = [Message(author="User", content="Hello", timestamp="2024-01-15T10:00:00Z")]
        
        with pytest.raises(Exception, match="LLM provider not configured"):
            llm.generate_structured_content(messages, TestModel, "Test instruction")

    @patch('zowie_agent_sdk.llm.openai.OpenAIProvider')
    def test_generate_structured_content_with_provider(self, mock_openai_provider, openai_config):
        """Test generate_structured_content delegates to provider."""
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            result: str
            
        events = []
        mock_provider_instance = MagicMock()
        mock_instance = TestModel(result="structured")
        mock_provider_instance.generate_structured_content.return_value = mock_instance
        mock_openai_provider.return_value = mock_provider_instance
        
        llm = LLM(config=openai_config, events=events, persona=None)
        messages = [Message(author="User", content="Generate JSON", timestamp="2024-01-15T10:00:00Z")]
        
        response = llm.generate_structured_content(
            messages,
            TestModel,
            system_instruction="Return valid JSON"
        )
        
        assert response == mock_instance
        assert isinstance(response, TestModel)
        assert response.result == "structured"
        mock_provider_instance.generate_structured_content.assert_called_once_with(
            messages,
            TestModel,
            "Return valid JSON",
            None,
            True
        )

    def test_persona_instruction_building(self):
        """Test persona instruction building in BaseLLMProvider."""
        from zowie_agent_sdk.protocol import Persona
        
        # Create a concrete implementation for testing
        class TestProvider(BaseLLMProvider):
            def generate_content(self, messages, system_instruction=None, **kwargs):
                return None
            
            def generate_structured_content(
                self, messages, schema, system_instruction=None, **kwargs
            ):
                return None

            def _prepare_messages(self, messages):
                return messages  # Simple passthrough for test
        
        # Test with full persona
        persona = Persona(
            name="Test Bot",
            business_context="Test context",
            tone_of_voice="Friendly"
        )
        
        config = GoogleProviderConfig(api_key="test", model="test")
        provider = TestProvider(config=config, events=[], persona=persona)
        
        instruction = provider._build_persona_instruction()
        assert "<persona>" in instruction
        assert "Test Bot" in instruction
        assert "Test context" in instruction
        assert "Friendly" in instruction
        
        # Test with None persona
        provider_no_persona = TestProvider(config=config, events=[], persona=None)
        instruction = provider_no_persona._build_persona_instruction()
        assert instruction == ""
        
        # Test with partial persona
        partial_persona = Persona(
            name="Bot Name",
            business_context=None,
            tone_of_voice=None
        )
        provider_partial = TestProvider(config=config, events=[], persona=partial_persona)
        instruction = provider_partial._build_persona_instruction()
        assert "Bot Name" in instruction
        assert "business_context" not in instruction
        assert "tone_of_voice" not in instruction