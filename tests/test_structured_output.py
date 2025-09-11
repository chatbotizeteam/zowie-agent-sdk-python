"""Tests for structured output support with Pydantic models and dicts."""

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from zowie_agent_sdk import Content, GoogleProviderConfig, OpenAIProviderConfig
from zowie_agent_sdk.domain import LLMResponse
from zowie_agent_sdk.llm import LLM
from zowie_agent_sdk.llm.google import GoogleProvider
from zowie_agent_sdk.llm.openai import OpenAIProvider


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
            mock_response.choices[0].message.content = '{"name": "Test", "age": 25}'
            mock_response.model_dump_json.return_value = (
                '{"choices": [{"message": {"content": "{\"name\": \"Test\", \"age\": 25}"}}]}'
            )
            mock_client.chat.completions.parse.return_value = mock_response
            
            # Test with Pydantic model (should use native parsing)
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Generate a person")],
                schema=SampleModel
            )
            
            # Verify the parse method was used for Pydantic models
            mock_client.chat.completions.parse.assert_called_once()
            call_args = mock_client.chat.completions.parse.call_args
            
            # Check that response_format is the Pydantic model class
            assert call_args.kwargs['response_format'] is SampleModel
            
            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.text == '{"name": "Test", "age": 25}'


    def test_dict_schema_accepted(self):
        """Test that dict schemas are properly accepted and parsed."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Dict schema should be accepted
        dict_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        # Mock the client to avoid actual API calls
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
            mock_response.model_dump_json.return_value = (
                '{"choices": [{"message": {"content": "{\"name\": \"John\", \"age\": 30}"}}]}'
            )
            mock_client.chat.completions.create.return_value = mock_response
            
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=dict_schema
            )
            
            assert result.text == '{"name": "John", "age": 30}'
            
            # Verify the create method was used for dict schemas
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            
            # Verify all required parameters were passed
            assert 'model' in call_args.kwargs
            assert call_args.kwargs['model'] == 'gpt-4'
            assert 'messages' in call_args.kwargs
            assert len(call_args.kwargs['messages']) == 1
            assert call_args.kwargs['messages'][0]['content'] == 'Test'


class TestGoogleStructuredOutput:
    """Test Google provider structured output functionality."""

    def test_pydantic_model_input(self):
        """Test that Pydantic models are accepted and processed correctly."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        provider = GoogleProvider(config=config, events=[], persona=None)
        
        # Mock the Google client
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = '{"name": "Test", "age": 25}'
            mock_response.model_dump_json.return_value = '{"candidates": [{"content": {}}]}'
            mock_client.models.generate_content.return_value = mock_response
            
            # Test with Pydantic model
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Generate a person")],
                schema=SampleModel
            )
            
            # Verify the call was made
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            
            # Check that config contains the JSON schema
            assert 'config' in call_args.kwargs
            config_arg = call_args.kwargs['config']
            assert hasattr(config_arg, 'response_json_schema')
            # The schema should be the Pydantic model's JSON schema
            assert config_arg.response_json_schema['type'] == 'object'
            assert 'name' in config_arg.response_json_schema['properties']
            assert 'age' in config_arg.response_json_schema['properties']
            assert config_arg.response_json_schema['properties']['name']['type'] == 'string'
            assert config_arg.response_json_schema['properties']['age']['type'] == 'integer'
            assert set(config_arg.response_json_schema['required']) == {'name', 'age'}
            
            # Verify the model name was used
            assert mock_client.models.generate_content.call_args.kwargs['model'] == 'gemini-pro'
            
            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.text == '{"name": "Test", "age": 25}'


    def test_dict_schema_accepted(self):
        """Test that dict schemas are properly accepted and parsed."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        provider = GoogleProvider(config=config, events=[], persona=None)
        
        # Dict schema should be accepted
        dict_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        # Mock the client to avoid actual API calls
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = '{"name": "John", "age": 30}'
            mock_response.model_dump_json.return_value = (
                '{"candidates": [{"content": {"parts": '
                '[{"text": "{\\"name\\": \\"John\\", \\"age\\": 30}"}]}}]}'
            )
            mock_client.models.generate_content.return_value = mock_response
            
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=dict_schema
            )
            
            assert result.text == '{"name": "John", "age": 30}'
            # Verify the schema was passed correctly
            call_args = mock_client.models.generate_content.call_args
            assert call_args.kwargs['config'].response_json_schema == dict_schema
            
            # Verify all configuration was set correctly
            assert call_args.kwargs['config'].response_mime_type == 'application/json'
            assert call_args.kwargs['model'] == 'gemini-pro'
            assert 'contents' in call_args.kwargs
            # Verify content transformation
            contents = call_args.kwargs['contents']
            assert len(contents) == 1
            assert contents[0]['role'] == 'user'
            assert contents[0]['parts'][0]['text'] == 'Test'


class TestLLMWrapperStructuredOutput:
    """Test the LLM wrapper class with structured output."""

    def test_llm_wrapper_forwards_pydantic_model(self):
        """Test that the LLM wrapper correctly forwards Pydantic models."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        llm = LLM(config=config, events=[], persona=None)
        
        # Mock the provider's generate_structured_content
        with patch.object(llm.provider, 'generate_structured_content') as mock_generate:
            mock_generate.return_value = LLMResponse(
                text='{"name": "Test"}',
                raw_response=None,
                provider="openai",
                model="gpt-4"
            )
            
            # Test with Pydantic model
            result = llm.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=SampleModel
            )
            
            # Verify the provider method was called with correct args
            mock_generate.assert_called_once_with(
                [Content(role="user", text="Test")],
                SampleModel,
                None
            )
            
            assert result.text == '{"name": "Test"}'

    def test_invalid_schema_type_raises_error(self):
        """Test that invalid schema types raise appropriate errors."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Invalid schema type (string)
        with pytest.raises(ValueError, match="Schema must be a Pydantic model class or dict"):
            provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema="invalid"  # type: ignore
            )

