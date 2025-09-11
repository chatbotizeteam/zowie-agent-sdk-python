"""Tests for LLM provider error handling."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from zowie_agent_sdk import Content
from zowie_agent_sdk.domain import LLMResponse
from zowie_agent_sdk.llm import LLM
from zowie_agent_sdk.protocol import LLMCallEvent


class TestLLMErrorHandling:
    """Test LLM error handling scenarios."""

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_llm_api_timeout(self, mock_google_provider, google_config):
        """Test handling of LLM API timeout."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Simulate timeout
        mock_provider_instance.generate_content.side_effect = requests.exceptions.Timeout(
            "Request to Gemini API timed out"
        )
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        with pytest.raises(requests.exceptions.Timeout):
            llm.generate_content(contents)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_llm_api_rate_limit(self, mock_google_provider, google_config):
        """Test handling of rate limiting from LLM API."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Simulate rate limit error
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_error = requests.HTTPError(response=mock_response)
        mock_provider_instance.generate_content.side_effect = mock_error
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        with pytest.raises(requests.HTTPError):
            llm.generate_content(contents)

    @patch('zowie_agent_sdk.llm.openai.OpenAIProvider')
    def test_invalid_api_key_error(self, mock_openai_provider, openai_config):
        """Test handling of invalid API key."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Simulate authentication error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_error = requests.HTTPError(response=mock_response)
        mock_provider_instance.generate_content.side_effect = mock_error
        mock_openai_provider.return_value = mock_provider_instance
        
        llm = LLM(config=openai_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        with pytest.raises(requests.HTTPError):
            llm.generate_content(contents)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_malformed_llm_response(self, mock_google_provider, google_config):
        """Test handling of malformed response from LLM provider."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Return invalid response structure
        mock_provider_instance.generate_content.side_effect = ValueError(
            "Invalid response format from API"
        )
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        with pytest.raises(ValueError):
            llm.generate_content(contents)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_network_connection_error(self, mock_google_provider, google_config):
        """Test handling of network connection errors."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Simulate connection error
        mock_provider_instance.generate_content.side_effect = requests.ConnectionError(
            "Failed to establish connection"
        )
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        with pytest.raises(requests.ConnectionError):
            llm.generate_content(contents)

    @patch('zowie_agent_sdk.llm.openai.OpenAIProvider')
    def test_structured_content_json_error(self, mock_openai_provider, openai_config):
        """Test handling of JSON parsing errors in structured content."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Return invalid JSON in structured response
        mock_provider_instance.generate_structured_content.return_value = LLMResponse(
            text="This is not valid JSON {",
            raw_response={},
            provider="openai",
            model="gpt-4",
        )
        mock_openai_provider.return_value = mock_provider_instance
        
        llm = LLM(config=openai_config, events=events, persona=None)
        contents = [Content(role="user", text="Generate JSON")]
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        
        # Should return the response even if it's not valid JSON
        # The validation is up to the caller
        response = llm.generate_structured_content(contents, schema)
        assert response.text == "This is not valid JSON {"

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_empty_response_from_llm(self, mock_google_provider, google_config):
        """Test handling of empty response from LLM."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Return empty response
        mock_provider_instance.generate_content.return_value = LLMResponse(
            text="",
            raw_response={},
            provider="google",
            model="gemini-2.0-flash",
        )
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        response = llm.generate_content(contents)
        assert response.text == ""

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_llm_event_recording_on_error(self, mock_google_provider, google_config):
        """Test that events are still recorded when LLM calls fail."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Simulate an error that should still record an event
        def side_effect_with_event(*args, **kwargs):
            # Add a mock event before raising
            from zowie_agent_sdk.protocol import LLMCallEventPayload
            events.append(
                LLMCallEvent(
                    payload=LLMCallEventPayload(
                        prompt="Test prompt",
                        response="Error occurred",
                        model="gemini-2.0-flash",
                        durationInMillis=100
                    )
                )
            )
            raise RuntimeError("LLM processing failed")
        
        mock_provider_instance.generate_content.side_effect = side_effect_with_event
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        contents = [Content(role="user", text="Test message")]
        
        with pytest.raises(RuntimeError):
            llm.generate_content(contents)
        
        # Event should still be recorded
        assert len(events) == 1
        assert events[0].payload.response == "Error occurred"

    def test_unsupported_provider_config(self):
        """Test handling of unsupported provider configuration."""
        events = []
        
        # Create a mock config that's not Google or OpenAI
        class UnsupportedConfig:
            provider = "unsupported"
            api_key = "test"
            model = "test"
        
        config = UnsupportedConfig()
        
        # This should initialize with provider=None
        llm = LLM(config=config, events=events, persona=None)
        assert llm.provider is None
        
        # And raise error when trying to use it
        contents = [Content(role="user", text="Test")]
        with pytest.raises(Exception, match="LLM provider not configured"):
            llm.generate_content(contents)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_very_long_prompt_handling(self, mock_google_provider, google_config):
        """Test handling of very long prompts."""
        events = []
        mock_provider_instance = MagicMock()
        
        # Return a response for long prompt
        mock_provider_instance.generate_content.return_value = LLMResponse(
            text="Response to long prompt",
            raw_response={},
            provider="google",
            model="gemini-2.0-flash",
        )
        mock_google_provider.return_value = mock_provider_instance
        
        llm = LLM(config=google_config, events=events, persona=None)
        
        # Create a very long prompt (10k+ characters)
        long_text = "This is a test. " * 1000
        contents = [Content(role="user", text=long_text)]
        
        response = llm.generate_content(contents)
        assert response.text == "Response to long prompt"
