"""Shared pytest fixtures and configuration."""

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from zowie_agent_sdk import (
    APIKeyAuth,
    BasicAuth,
    BearerTokenAuth,
    GoogleProviderConfig,
    OpenAIProviderConfig,
)
from zowie_agent_sdk.protocol import (
    Message,
    Metadata,
    Persona,
)


@pytest.fixture
def sample_metadata() -> Metadata:
    """Sample metadata for testing."""
    return Metadata(
        requestId="test-request-123",
        chatbotId="test-chatbot-456",
        conversationId="test-conversation-789",
        interactionId="test-interaction-001",
    )


@pytest.fixture
def sample_messages() -> List[Message]:
    """Sample messages for testing."""
    return [
        Message(
            author="User",
            content="Hello, I need help",
            timestamp="2025-01-01T10:00:00.000Z",
        ),
        Message(
            author="Chatbot",
            content="I'm here to help you!",
            timestamp="2025-01-01T10:00:01.000Z",
        ),
    ]


@pytest.fixture
def sample_persona() -> Persona:
    """Sample persona for testing."""
    return Persona(
        name="Test Assistant",
        business_context="You are a helpful test assistant",
        tone_of_voice="Professional and friendly",
    )


@pytest.fixture
def google_config() -> GoogleProviderConfig:
    """Google LLM configuration for testing."""
    return GoogleProviderConfig(
        api_key="test-google-api-key",
        model="gemini-2.0-flash",
    )


@pytest.fixture
def openai_config() -> OpenAIProviderConfig:
    """OpenAI LLM configuration for testing."""
    return OpenAIProviderConfig(
        api_key="test-openai-api-key",
        model="gpt-4",
    )


@pytest.fixture
def api_key_auth() -> APIKeyAuth:
    """API Key authentication configuration."""
    return APIKeyAuth(
        header_name="X-API-Key",
        api_key="test-api-key-123",
    )


@pytest.fixture
def basic_auth() -> BasicAuth:
    """Basic authentication configuration."""
    return BasicAuth(
        username="testuser",
        password="testpass123",
    )


@pytest.fixture
def bearer_auth() -> BearerTokenAuth:
    """Bearer token authentication configuration."""
    return BearerTokenAuth(
        token="test-bearer-token-xyz",
    )


@pytest.fixture
def mock_request() -> Mock:
    """Mock FastAPI Request object."""
    request = Mock()
    request.headers = {}
    return request


@pytest.fixture
def sample_request_json() -> Dict[str, Any]:
    """Sample request JSON matching SPEC.md format."""
    return {
        "metadata": {
            "requestId": "test-request-123",
            "chatbotId": "test-chatbot-456",
            "conversationId": "test-conversation-789",
            "interactionId": "test-interaction-001",
        },
        "messages": [
            {
                "author": "User",
                "content": "Hello, I need help",
                "timestamp": "2025-01-01T10:00:00.000Z",
            },
            {
                "author": "Chatbot",
                "content": "I'm here to help you!",
                "timestamp": "2025-01-01T10:00:01.000Z",
            },
        ],
        "context": "Test context",
        "persona": {
            "name": "Test Assistant",
            "businessContext": "You are a helpful test assistant",
            "toneOfVoice": "Professional and friendly",
        },
    }


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Mock LLM response."""
    return {
        "text": "This is a test response from the LLM.",
        "raw_response": {"test": "data"},
        "provider": "google",
        "model": "gemini-2.0-flash",
    }