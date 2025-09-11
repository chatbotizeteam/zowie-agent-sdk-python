"""Integration tests for the complete SDK."""

import base64
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    APIKeyAuth,
    BasicAuth,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    TransferToBlockResponse,
)


class IntegrationTestAgent(Agent):
    """Test agent for integration testing."""

    def handle(self, context: Context):
        """Handle based on message content."""
        last_message = context.messages[-1].content.lower() if context.messages else ""
        
        # Store some values
        context.store_value("processed", True)
        context.store_value("message_count", len(context.messages))
        
        if "finish" in last_message:
            return TransferToBlockResponse(
                message="Task completed",
                next_block="completion_block"
            )
        elif "error" in last_message:
            raise ValueError("Simulated error")
        else:
            return ContinueConversationResponse(
                message=f"Processed {len(context.messages)} messages"
            )


class TestIntegration:
    """Integration tests for the complete flow."""

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_full_request_cycle_no_auth(self, mock_google_provider):
        """Test complete request cycle without authentication."""
        # Setup mock provider
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        # Create agent
        agent = IntegrationTestAgent(
            llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.0-flash")
        )
        client = TestClient(agent.app)
        
        # Prepare request
        request_data = {
            "metadata": {
                "requestId": "int-test-001",
                "chatbotId": "bot-001",
                "conversationId": "conv-001",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Hello bot",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # Send request
        response = client.post("/", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        
        assert result["command"]["type"] == "send_message"
        assert result["command"]["payload"]["message"] == "Processed 1 messages"
        assert result["valuesToSave"]["processed"] is True
        assert result["valuesToSave"]["message_count"] == 1

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_full_request_cycle_with_api_key_auth(self, mock_google_provider):
        """Test complete request cycle with API key authentication."""
        # Setup mock provider
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        # Create agent with auth
        auth_config = APIKeyAuth(header_name="X-API-Key", api_key="secret-key")
        agent = IntegrationTestAgent(
            llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.0-flash"),
            auth_config=auth_config
        )
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "int-test-002",
                "chatbotId": "bot-001",
                "conversationId": "conv-001",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Test message",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # Request without auth should fail
        response = client.post("/", json=request_data)
        assert response.status_code == 401
        
        # Request with correct auth should succeed
        headers = {"X-API-Key": "secret-key"}
        response = client.post("/", json=request_data, headers=headers)
        assert response.status_code == 200

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_full_request_cycle_with_basic_auth(self, mock_google_provider):
        """Test complete request cycle with Basic authentication."""
        # Setup mock provider
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        # Create agent with basic auth
        auth_config = BasicAuth(username="admin", password="secret")
        agent = IntegrationTestAgent(
            llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.0-flash"),
            auth_config=auth_config
        )
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "int-test-003",
                "chatbotId": "bot-001",
                "conversationId": "conv-001",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Hello",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        # Request with correct basic auth
        credentials = base64.b64encode(b"admin:secret").decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}
        response = client.post("/", json=request_data, headers=headers)
        assert response.status_code == 200

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_finish_response_flow(self, mock_google_provider):
        """Test flow that results in a finish response."""
        # Setup mock provider
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = IntegrationTestAgent(
            llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.0-flash")
        )
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "int-test-004",
                "chatbotId": "bot-001",
                "conversationId": "conv-001",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Please finish the task",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        response = client.post("/", json=request_data)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["command"]["type"] == "go_to_next_block"
        assert result["command"]["payload"]["message"] == "Task completed"
        assert result["command"]["payload"]["nextBlockReferenceKey"] == "completion_block"

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_persona_handling(self, mock_google_provider):
        """Test that persona is properly passed through the system."""
        # Setup mock provider
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = IntegrationTestAgent(
            llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.0-flash")
        )
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "int-test-005",
                "chatbotId": "bot-001",
                "conversationId": "conv-001",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Test with persona",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
            "persona": {
                "name": "Assistant Bot",
                "businessContext": "You are a helpful assistant",
                "toneOfVoice": "Professional",
            }
        }
        
        response = client.post("/", json=request_data)
        assert response.status_code == 200
        
        # Verify persona was passed to provider
        mock_google_provider.assert_called_once()
        call_args = mock_google_provider.call_args
        assert call_args[1]["persona"] is not None
        assert call_args[1]["persona"].name == "Assistant Bot"

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_empty_values_handling(self, mock_google_provider):
        """Test handling of empty values in response."""
        # Setup mock provider
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        class EmptyValuesAgent(Agent):
            def handle(self, context: Context):
                # Don't store any values
                return ContinueConversationResponse(message="No values stored")
        
        agent = EmptyValuesAgent(
            llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.0-flash")
        )
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "int-test-006",
                "chatbotId": "bot-001",
                "conversationId": "conv-001",
            },
            "messages": [],
        }
        
        response = client.post("/", json=request_data)
        assert response.status_code == 200
        
        result = response.json()
        # valuesToSave should be None when no values are stored
        assert result["valuesToSave"] is None
        # events should be None when no events occurred
        assert result["events"] is None