"""Tests for error handling and edge cases."""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    AgentResponseContinue,
    Context,
    GoogleConfig,
)


class ErrorAgent(Agent):
    """Agent that raises errors for testing."""
    
    def handle(self, context: Context):
        if context.messages and "runtime" in context.messages[-1].content.lower():
            raise RuntimeError("Runtime error for testing")
        if context.messages and "error" in context.messages[-1].content.lower():
            raise ValueError("Intentional error for testing")
        return AgentResponseContinue(message="Success")


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_malformed_json_request(self, mock_google_provider):
        """Test handling of malformed JSON in request."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        # Send invalid JSON
        response = client.post(
            "/",
            content='{"invalid": json}',
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_missing_required_metadata(self, mock_google_provider):
        """Test request with missing metadata field."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        # Missing metadata entirely
        request_data = {
            "messages": [
                {
                    "author": "User",
                    "content": "Test",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # This will raise a KeyError and FastAPI will re-raise it
        with pytest.raises(Exception):  # Could be KeyError or wrapped exception
            response = client.post("/", json=request_data)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_missing_required_metadata_fields(self, mock_google_provider):
        """Test request with incomplete metadata."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        # Missing requestId in metadata
        request_data = {
            "metadata": {
                "chatbotId": "bot-123",
                "conversationId": "conv-456",
            },
            "messages": [],
        }
        
        # This will raise a KeyError for missing requestId
        with pytest.raises(Exception):
            response = client.post("/", json=request_data)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_agent_raises_exception(self, mock_google_provider):
        """Test when agent's handle method raises an exception."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Trigger an error please",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # Agent raises ValueError, FastAPI will re-raise it in test mode
        with pytest.raises(ValueError, match="Intentional error for testing"):
            response = client.post("/", json=request_data)

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_empty_messages_array(self, mock_google_provider):
        """Test handling of empty messages array."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [],
        }
        
        response = client.post("/", json=request_data)
        # Should handle empty messages gracefully
        assert response.status_code == 200
        result = response.json()
        assert result["command"]["type"] == "send_message"
        assert result["command"]["payload"]["message"] == "Success"

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_very_large_message_history(self, mock_google_provider):
        """Test handling of very large message history."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        # Create 100+ messages
        large_messages = []
        for i in range(150):
            large_messages.append({
                "author": "User" if i % 2 == 0 else "Chatbot",
                "content": f"Message {i}",
                "timestamp": f"2025-01-01T10:{i:02d}:00.000Z",
            })
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": large_messages,
        }
        
        response = client.post("/", json=request_data)
        assert response.status_code == 200
        # Verify all messages were processed
        result = response.json()
        assert result["command"]["type"] == "send_message"

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_unicode_and_emoji_in_messages(self, mock_google_provider):
        """Test handling of Unicode and emoji characters."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Hello 你好 مرحبا 🌍🎉😊",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                },
                {
                    "author": "Chatbot",
                    "content": "Response with émojis 🤖 and spëcial çhars ñ",
                    "timestamp": "2025-01-01T10:00:01.000Z",
                }
            ],
        }
        
        response = client.post("/", json=request_data)
        assert response.status_code == 200
        result = response.json()
        assert result["command"]["type"] == "send_message"

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_null_values_in_optional_fields(self, mock_google_provider):
        """Test handling of null values in optional fields."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
                "interactionId": None,  # Explicitly null
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Test",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
            "context": None,  # Explicitly null
            "persona": None,  # Explicitly null
        }
        
        response = client.post("/", json=request_data)
        assert response.status_code == 200

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_invalid_message_author(self, mock_google_provider):
        """Test handling of invalid message author field."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [
                {
                    "author": "InvalidAuthor",  # Not "User" or "Chatbot"
                    "content": "Test message",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # Should still process - we don't validate author values strictly
        response = client.post("/", json=request_data)
        assert response.status_code == 200

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_different_runtime_errors(self, mock_google_provider):
        """Test handling of different types of runtime errors."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ErrorAgent(llm_config=GoogleConfig(api_key="test", model="test"))
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Trigger runtime error",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # Agent raises RuntimeError, FastAPI will re-raise it in test mode
        with pytest.raises(RuntimeError, match="Runtime error for testing"):
            response = client.post("/", json=request_data)