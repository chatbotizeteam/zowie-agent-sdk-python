"""Simplified integration tests that work with the actual SDK implementation."""

from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    AgentResponse,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    TransferToBlockResponse,
)


class SimpleEchoAgent(Agent):
    """Simple test agent that echoes messages."""

    def handle(self, context: Context) -> AgentResponse:
        # Handle empty conversation start
        if len(context.messages) == 0:
            return ContinueConversationResponse(message="Hello! Send me a message.")

        last_message = context.messages[-1].content
        context.store_value("echoed_message", last_message)
        return ContinueConversationResponse(message=f"Echo: {last_message}")


class SimpleTransferAgent(Agent):
    """Simple test agent that transfers on keyword."""

    def handle(self, context: Context) -> AgentResponse:
        # Handle empty conversation start
        if len(context.messages) == 0:
            return ContinueConversationResponse(message="How can I help?")

        last_message = context.messages[-1].content.lower()
        if "transfer" in last_message:
            return TransferToBlockResponse(message="Transferring you now...", next_block="support")

        return ContinueConversationResponse(message="I can help with that!")


def test_simple_agent_creation() -> None:
    """Test that we can create agents successfully."""
    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    assert agent is not None
    assert agent.app is not None
    assert hasattr(agent, "handle")


def test_agent_health_endpoint() -> None:
    """Test the health endpoint works."""
    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["agent"] == "SimpleEchoAgent"


def test_echo_agent_workflow() -> None:
    """Test complete echo agent workflow."""
    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    # Test with empty messages
    request_data: Dict[str, Any] = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"
    assert "Hello!" in data["command"]["payload"]["message"]

    # Test with a message
    request_data["messages"] = [
        {"author": "User", "content": "Test message", "timestamp": "2024-01-01T00:00:00Z"}
    ]

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"
    assert "Echo: Test message" in data["command"]["payload"]["message"]
    assert data["valuesToSave"]["echoed_message"] == "Test message"


def test_transfer_agent_workflow() -> None:
    """Test transfer agent workflow."""
    agent = SimpleTransferAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    # Test normal message
    request_data = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [
            {"author": "User", "content": "I need help", "timestamp": "2024-01-01T00:00:00Z"}
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"
    assert "I can help" in data["command"]["payload"]["message"]

    # Test transfer keyword
    request_data["messages"] = [
        {"author": "User", "content": "Please transfer me", "timestamp": "2024-01-01T00:00:00Z"}
    ]

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "go_to_next_block"
    assert data["command"]["payload"]["nextBlockReferenceKey"] == "support"


def test_agent_with_persona_and_context() -> None:
    """Test agent with persona and context data."""
    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    request_data = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [{"author": "User", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"}],
        "persona": {
            "name": "Assistant",
            "businessContext": "Customer support",
            "toneOfVoice": "Friendly",
        },
        "context": "Customer needs help",
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "Echo: Hello" in data["command"]["payload"]["message"]


def test_invalid_request_validation() -> None:
    """Test request validation with invalid data."""
    from pydantic import ValidationError

    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    # Missing required fields - in this SDK implementation,
    # Pydantic validation errors are raised as exceptions
    invalid_data = {"invalid": "data"}

    # The validation error will be raised as a ValidationError
    with pytest.raises(ValidationError):
        client.post("/", json=invalid_data)


@patch("requests.request")
def test_http_client_integration(mock_requests: Mock) -> None:
    """Test HTTP client integration with mocked requests."""
    # Mock the requests.get call
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = '{"status": "success"}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_requests.return_value = mock_response

    class HttpTestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            # Make HTTP request
            response = context.http.get(
                url="https://api.example.com/test", headers={"Authorization": "Bearer token"}
            )

            if response.status_code == 200:
                return ContinueConversationResponse(message="API call successful")
            else:
                return ContinueConversationResponse(message="API call failed")

    agent = HttpTestAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    request_data = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [
            {"author": "User", "content": "Test API", "timestamp": "2024-01-01T00:00:00Z"}
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "API call successful" in data["command"]["payload"]["message"]

    # Verify the mock was called
    mock_requests.assert_called_once()


def test_agent_error_handling() -> None:
    """Test agent error handling."""

    class ErrorAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            raise ValueError("Test error")

    agent = ErrorAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    request_data = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [],
    }

    # Should raise the exception (FastAPI will handle it)
    with pytest.raises(ValueError):
        client.post("/", json=request_data)


def test_context_object_functionality() -> None:
    """Test the Context object functionality."""

    class ContextTestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            # Test context properties
            assert context.metadata is not None
            assert hasattr(context, "messages")
            assert hasattr(context, "store_value")
            assert hasattr(context, "http")
            assert hasattr(context, "llm")

            # Test store_value
            context.store_value("test_key", "test_value")
            context.store_value("message_count", len(context.messages))

            return ContinueConversationResponse(message="Context test passed")

    agent = ContextTestAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    request_data = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [{"author": "User", "content": "Test", "timestamp": "2024-01-01T00:00:00Z"}],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "Context test passed" in data["command"]["payload"]["message"]
    assert data["valuesToSave"]["test_key"] == "test_value"
    assert data["valuesToSave"]["message_count"] == 1


def test_multi_turn_conversation() -> None:
    """Test agent with multi-turn conversation."""
    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )
    client = TestClient(agent.app)

    # Multi-turn conversation
    request_data = {
        "metadata": {"requestId": "test-1", "chatbotId": "test-bot", "conversationId": "test-conv"},
        "messages": [
            {"author": "User", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
            {"author": "Chatbot", "content": "Echo: Hello", "timestamp": "2024-01-01T00:00:01Z"},
            {"author": "User", "content": "How are you?", "timestamp": "2024-01-01T00:00:02Z"},
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Should echo the last user message
    assert "Echo: How are you?" in data["command"]["payload"]["message"]


def test_agent_configuration_options() -> None:
    """Test various agent configuration options."""
    # Test with different timeout
    agent = SimpleEchoAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"),
        http_timeout_seconds=30.0,
        include_persona_by_default=False,
        include_context_by_default=False,
        log_level="ERROR",
    )

    assert agent.http_timeout_seconds == 30.0
    assert agent.include_persona_by_default is False
    assert agent.include_context_by_default is False

    # Should still work
    client = TestClient(agent.app)
    response = client.get("/health")
    assert response.status_code == 200
