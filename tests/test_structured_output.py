"""Tests for structured output support with Pydantic models and JSON strings."""

import json
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
        
        # Mock the OpenAI client
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.output_text = '{"name": "Test", "age": 25}'
            mock_response.model_dump_json.return_value = '{"output_text": "test", "usage": {}}'
            mock_client.responses.create.return_value = mock_response
            
            # Test with Pydantic model
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Generate a person")],
                schema=SampleModel
            )
            
            # Verify the call was made
            mock_client.responses.create.assert_called_once()
            call_args = mock_client.responses.create.call_args
            
            # Check that text parameter contains the JSON schema config
            assert 'text' in call_args.kwargs
            text_config = call_args.kwargs['text']
            assert text_config['format']['type'] == 'json_schema'
            assert text_config['format']['name'] == 'SampleModel'
            assert 'schema' in text_config['format']
            
            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.text == '{"name": "Test", "age": 25}'

    def test_json_string_schema_input(self):
        """Test that JSON string schemas are accepted and parsed correctly."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Create a JSON schema as string
        json_schema = json.dumps({
            "type": "object",
            "title": "Person",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })
        
        # Mock the OpenAI client
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.output_text = '{"name": "Alice", "age": 30}'
            mock_response.model_dump_json.return_value = '{"output_text": "test", "usage": {}}'
            mock_client.responses.create.return_value = mock_response
            
            # Test with JSON string schema
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Generate a person")],
                schema=json_schema
            )
            
            # Verify the call was made
            mock_client.responses.create.assert_called_once()
            call_args = mock_client.responses.create.call_args
            
            # Check that text parameter contains the parsed schema
            assert 'text' in call_args.kwargs
            text_config = call_args.kwargs['text']
            assert text_config['format']['type'] == 'json_schema'
            assert text_config['format']['name'] == 'Person'
            assert text_config['format']['schema']['type'] == 'object'
            
            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.text == '{"name": "Alice", "age": 30}'

    def test_invalid_json_string_raises_error(self):
        """Test that invalid JSON strings raise appropriate errors."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Invalid JSON string
        invalid_json = "{'invalid': json}"
        
        with pytest.raises(ValueError, match="Invalid JSON schema string"):
            provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=invalid_json
            )

    def test_dict_input_raises_error(self):
        """Test that dict inputs are rejected as expected."""
        config = OpenAIProviderConfig(api_key="test-key", model="gpt-4")
        provider = OpenAIProvider(config=config, events=[], persona=None)
        
        # Dict should not be accepted
        dict_schema = {"type": "object", "properties": {}}
        
        with pytest.raises(ValueError, match="Schema must be a Pydantic model class"):
            provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=dict_schema  # type: ignore
            )


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
            
            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.text == '{"name": "Test", "age": 25}'

    def test_json_string_schema_input(self):
        """Test that JSON string schemas are accepted and parsed correctly."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        provider = GoogleProvider(config=config, events=[], persona=None)
        
        # Create a JSON schema as string
        json_schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        })
        
        # Mock the Google client
        with patch.object(provider, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.candidates = [MagicMock()]
            mock_response.candidates[0].content.parts = [MagicMock()]
            mock_response.candidates[0].content.parts[0].text = '{"name": "Alice", "age": 30}'
            mock_response.model_dump_json.return_value = '{"candidates": [{"content": {}}]}'
            mock_client.models.generate_content.return_value = mock_response
            
            # Test with JSON string schema
            result = provider.generate_structured_content(
                contents=[Content(role="user", text="Generate a person")],
                schema=json_schema
            )
            
            # Verify the call was made
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            
            # Check that config contains the parsed schema
            assert 'config' in call_args.kwargs
            config_arg = call_args.kwargs['config']
            assert hasattr(config_arg, 'response_json_schema')
            assert config_arg.response_json_schema['type'] == 'object'
            
            # Verify response
            assert isinstance(result, LLMResponse)
            assert result.text == '{"name": "Alice", "age": 30}'

    def test_invalid_json_string_raises_error(self):
        """Test that invalid JSON strings raise appropriate errors."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        provider = GoogleProvider(config=config, events=[], persona=None)
        
        # Invalid JSON string
        invalid_json = "not valid json"
        
        with pytest.raises(ValueError, match="Invalid JSON schema string"):
            provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=invalid_json
            )

    def test_dict_input_raises_error(self):
        """Test that dict inputs are rejected as expected."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        provider = GoogleProvider(config=config, events=[], persona=None)
        
        # Dict should not be accepted
        dict_schema = {"type": "object", "properties": {}}
        
        with pytest.raises(ValueError, match="Schema must be a Pydantic model class"):
            provider.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=dict_schema  # type: ignore
            )


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

    def test_llm_wrapper_forwards_json_string(self):
        """Test that the LLM wrapper correctly forwards JSON strings."""
        config = GoogleProviderConfig(api_key="test-key", model="gemini-pro")
        llm = LLM(config=config, events=[], persona=None)
        
        json_schema = json.dumps({"type": "object"})
        
        # Mock the provider's generate_structured_content
        with patch.object(llm.provider, 'generate_structured_content') as mock_generate:
            mock_generate.return_value = LLMResponse(
                text='{}',
                raw_response=None,
                provider="google",
                model="gemini-pro"
            )
            
            # Test with JSON string
            result = llm.generate_structured_content(
                contents=[Content(role="user", text="Test")],
                schema=json_schema
            )
            
            # Verify the provider method was called with correct args
            mock_generate.assert_called_once_with(
                [Content(role="user", text="Test")],
                json_schema,
                None
            )
            
            assert result.text == '{}'