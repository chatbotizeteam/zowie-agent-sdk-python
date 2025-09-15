"""Tests for structured output support with Pydantic models and dicts."""

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from zowie_agent_sdk import GoogleProviderConfig, OpenAIProviderConfig
from zowie_agent_sdk.domain import LLMResponse
from zowie_agent_sdk.llm import LLM
from zowie_agent_sdk.llm.google import GoogleProvider
from zowie_agent_sdk.llm.openai import OpenAIProvider
from zowie_agent_sdk.protocol import Message


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""
    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = []


class TestOpenAIStructuredOutput:
    """Test OpenAI provider structured output functionality."""

    def test_pydantic_model_input(self):
        """Test that Pydantic models are accepted and processed correctly."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Mock the OpenAI client for native Pydantic parsing
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            
            # Create actual model instance for parsed result
            sample_instance = SampleModel(name="Test", age=25)
            mock_response.choices[0].message.parsed = sample_instance
            mock_response.model_dump_json.return_value = (
                '{"choices": [{"message": {"parsed": {"name": "Test", "age": 25}}}]}'
            )
            mock_client.chat.completions.parse.return_value = mock_response
            
            # Test with Pydantic model (should use native parsing)
            result = provider.generate_structured_content(
                messages=[Message(author="User", content="Generate a person", timestamp="2024-01-15T10:00:00Z")],
                schema=SampleModel,
                system_instruction="Generate a test person"
            )
            
            # Verify the parse method was used for Pydantic models
            mock_client.chat.completions.parse.assert_called_once()
            call_args = mock_client.chat.completions.parse.call_args
            
            # Check that response_format is the Pydantic model class
            assert call_args.kwargs['response_format'] is SampleModel
            
            # Verify response - should return the actual model instance
            assert isinstance(result, SampleModel)
            assert result.name == "Test"
            assert result.age == 25




class TestGoogleStructuredOutput:
    """Test Google provider structured output functionality."""

    def test_pydantic_model_input(self):
        """Test that Pydantic models are accepted and processed correctly."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        provider = GoogleProvider(config=config, events=[], persona=None)
        
        # Mock the Google client
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            
            # Create actual model instance for parsed result
            sample_instance = SampleModel(name="Test", age=25)
            mock_response.parsed = sample_instance
            mock_response.model_dump_json.return_value = '{"parsed": {"name": "Test", "age": 25}}'
            mock_client.models.generate_content.return_value = mock_response
            
            # Test with Pydantic model
            result = provider.generate_structured_content(
                messages=[Message(author="User", content="Generate a person", timestamp="2024-01-15T10:00:00Z")],
                schema=SampleModel,
                system_instruction="Generate a test person"
            )
            
            # Verify the call was made with correct parameters
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            
            # Check that config contains the response_schema
            assert 'config' in call_args.kwargs
            config_arg = call_args.kwargs['config']
            assert config_arg['response_schema'] is SampleModel
            assert config_arg['response_mime_type'] == 'application/json'
            
            # Verify the model name was used
            assert call_args.kwargs['model'] == 'gemini-pro'
            
            # Verify response - should return the actual model instance
            assert isinstance(result, SampleModel)
            assert result.name == "Test"
            assert result.age == 25




class TestLLMWrapperStructuredOutput:
    """Test the LLM wrapper class with structured output."""

    def test_llm_wrapper_forwards_pydantic_model(self):
        """Test that the LLM wrapper correctly forwards Pydantic models."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        llm = LLM(config=config, events=[], persona=None)
        
        # Mock the provider's generate_structured_content  
        with patch.object(llm.provider, 'generate_structured_content') as mock_generate:
            # Return an actual model instance
            sample_instance = SampleModel(name="Test", age=30)
            mock_generate.return_value = sample_instance
            
            # Test with Pydantic model
            result = llm.generate_structured_content(
                messages=[Message(author="User", content="Test", timestamp="2024-01-15T10:00:00Z")],
                schema=SampleModel,
                system_instruction="Generate test data"
            )
            
            # Verify the provider method was called with correct args
            mock_generate.assert_called_once_with(
                [Message(author="User", content="Test", timestamp="2024-01-15T10:00:00Z")],
                SampleModel,
                "Generate test data",
                None,
                True
            )
            
            # Verify we get back the actual model instance
            assert isinstance(result, SampleModel)
            assert result.name == "Test"
            assert result.age == 30

    def test_invalid_schema_type_raises_error(self):
        """Test that invalid schema types raise appropriate errors."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Invalid schema type (string) should raise TypeError from OpenAI
        with pytest.raises(TypeError, match="Unsupported response_format type"):
            provider.generate_structured_content(
                messages=[Message(author="User", content="Test", timestamp="2024-01-15T10:00:00Z")],
                schema="invalid",  # type: ignore
                system_instruction="Test instruction"
            )

