"""Pytest configuration and shared fixtures for Zowie Agent SDK tests."""

import pytest
from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    AgentResponse,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
)


class TestAgent(Agent):
    """Simple test agent for integration testing."""

    def handle(self, context: Context) -> AgentResponse:
        """Simple echo handler for testing."""
        last_message = context.messages[-1].content if context.messages else "No messages"
        return ContinueConversationResponse(message=f"Echo: {last_message}")


class LLMTestAgent(Agent):
    """Test agent that uses LLM functionality."""

    def handle(self, context: Context) -> AgentResponse:
        """Handler that uses LLM to generate content."""
        response = context.llm.generate_content(
            messages=context.messages,
            system_instruction="You are a helpful test assistant. Respond briefly.",
        )
        return ContinueConversationResponse(message=response)


@pytest.fixture
def test_agent() -> TestAgent:
    """Create a test agent instance."""
    return TestAgent(
        llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.5-flash"),
        log_level="ERROR",  # Suppress logs during testing
    )


@pytest.fixture
def llm_test_agent() -> LLMTestAgent:
    """Create a test agent with LLM functionality."""
    return LLMTestAgent(
        llm_config=GoogleProviderConfig(api_key="test-key", model="gemini-2.5-flash"),
        log_level="ERROR",
    )


@pytest.fixture
def test_client(test_agent: TestAgent) -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(test_agent.app)


# Test data creation utilities are available in tests/utils.py:
# - create_test_metadata(): Create customizable Metadata objects
# - create_test_message(): Create customizable Message objects
# - create_mock_http_response(): Create mock HTTP responses
# - assert_valid_agent_response(): Validate response structure
# - assert_events_recorded(): Check events in responses
