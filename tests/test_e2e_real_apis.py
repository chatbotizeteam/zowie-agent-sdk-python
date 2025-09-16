"""
Real end-to-end tests that call actual external APIs.

These tests are OPTIONAL and only run when real API keys are provided.
They test the complete integration with real LLM providers and external services.

To run these tests, set environment variables:
- GOOGLE_API_KEY (for Gemini tests)
- OPENAI_API_KEY (for OpenAI tests)

Usage:
    # Run with real API keys
    GOOGLE_API_KEY=your_key OPENAI_API_KEY=your_key poetry run pytest tests/test_e2e_real_apis.py -v

    # Skip if no API keys
    poetry run pytest tests/test_e2e_real_apis.py -v  # Will skip tests
"""

import os

import pytest
from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    AgentResponse,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    OpenAIProviderConfig,
)

# Skip all tests in this file if no API keys are provided
pytestmark = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="No API keys provided - set GOOGLE_API_KEY or OPENAI_API_KEY to run real E2E tests",
)


class RealLLMTestAgent(Agent):
    """Test agent that makes real LLM calls."""

    def handle(self, context: Context) -> AgentResponse:
        if not context.messages:
            return ContinueConversationResponse(
                message="Hello! I'm a real AI assistant. What can I help you with?"
            )

        # Make a real LLM call
        response = context.llm.generate_content(
            messages=context.messages,
            system_instruction="You are a helpful assistant. Keep your response to 1-2 sentences.",
        )

        # Store the fact that we made a real API call
        context.store_value("real_llm_call_made", True)
        context.store_value("response_length", len(response))

        return ContinueConversationResponse(message=response)


class RealHTTPTestAgent(Agent):
    """Test agent that makes real HTTP calls."""

    def handle(self, context: Context) -> AgentResponse:
        # Handle empty conversation start
        if len(context.messages) == 0:
            return ContinueConversationResponse(message="Send me a request to test HTTP calls.")

        # Make a real HTTP call to a public API
        try:
            response = context.http.get(
                url="https://httpbin.org/json", headers={"User-Agent": "ZowieSDK-E2E-Test/1.0"}
            )

            if response.status_code == 200:
                data = response.json()
                context.store_value("real_http_call_made", True)
                context.store_value("api_response_keys", list(data.keys()))

                return ContinueConversationResponse(
                    message=f"Successfully called httpbin.org and got response with keys: {list(data.keys())}"  # noqa: E501
                )
            else:
                return ContinueConversationResponse(
                    message=f"HTTP call failed with status {response.status_code}"
                )

        except Exception as e:
            context.store_value("http_error", str(e))
            return ContinueConversationResponse(message=f"HTTP call failed with error: {str(e)}")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_real_google_gemini_integration() -> None:
    """Test real integration with Google Gemini API."""
    agent = RealLLMTestAgent(
        llm_config=GoogleProviderConfig(
            api_key=os.environ["GOOGLE_API_KEY"], model="gemini-2.5-flash"
        ),
        log_level="INFO",
    )

    client = TestClient(agent.app)

    request_data = {
        "metadata": {
            "requestId": "real-google-test-1",
            "chatbotId": "e2e-test-bot",
            "conversationId": "real-google-conv",
        },
        "messages": [
            {
                "author": "User",
                "content": "What is 2+2? Just give me the number.",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"

    # Verify we got a real response
    message = data["command"]["payload"]["message"]
    assert len(message) > 0
    assert isinstance(message, str)

    # Verify our tracking worked
    assert data["valuesToSave"]["real_llm_call_made"] is True
    assert data["valuesToSave"]["response_length"] > 0

    # Verify events were tracked
    assert data["events"] is not None
    assert len(data["events"]) > 0
    assert data["events"][0]["type"] == "llm_call"

    print(f"✅ Real Gemini response: {message}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_real_openai_gpt_integration() -> None:
    """Test real integration with OpenAI GPT API."""
    agent = RealLLMTestAgent(
        llm_config=OpenAIProviderConfig(api_key=os.environ["OPENAI_API_KEY"], model="gpt-5-mini"),
        log_level="INFO",
    )

    print(f"Agent: {agent.llm_config.api_key}")

    client = TestClient(agent.app)

    request_data = {
        "metadata": {
            "requestId": "real-openai-test-1",
            "chatbotId": "e2e-test-bot",
            "conversationId": "real-openai-conv",
        },
        "messages": [
            {
                "author": "User",
                "content": "What is the capital of France? One word answer only.",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"

    # Verify we got a real response
    message = data["command"]["payload"]["message"]
    assert len(message) > 0
    assert isinstance(message, str)

    # Verify our tracking worked
    assert data["valuesToSave"]["real_llm_call_made"] is True
    assert data["valuesToSave"]["response_length"] > 0

    # Verify events were tracked
    assert data["events"] is not None
    assert len(data["events"]) > 0
    assert data["events"][0]["type"] == "llm_call"

    print(f"✅ Real OpenAI response: {message}")


def test_real_http_api_integration() -> None:
    """Test real HTTP API integration with httpbin.org."""
    agent = RealHTTPTestAgent(
        llm_config=GoogleProviderConfig(
            api_key="dummy-key-for-http-test", model="gemini-2.5-flash"
        ),
        log_level="INFO",
    )

    client = TestClient(agent.app)

    request_data = {
        "metadata": {
            "requestId": "real-http-test-1",
            "chatbotId": "e2e-test-bot",
            "conversationId": "real-http-conv",
        },
        "messages": [
            {"author": "User", "content": "Test the HTTP API", "timestamp": "2024-01-01T00:00:00Z"}
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"

    # Verify we got a real response
    message = data["command"]["payload"]["message"]
    assert "httpbin.org" in message
    assert "Successfully called" in message

    # Verify our tracking worked
    assert data["valuesToSave"]["real_http_call_made"] is True
    assert "api_response_keys" in data["valuesToSave"]

    # Verify events were tracked for the HTTP call
    assert data["events"] is not None
    assert len(data["events"]) > 0

    # Find the API call event
    api_events = [e for e in data["events"] if e["type"] == "api_call"]
    assert len(api_events) > 0

    api_event = api_events[0]
    assert api_event["payload"]["url"] == "https://httpbin.org/json"
    assert api_event["payload"]["requestMethod"] == "GET"
    assert api_event["payload"]["responseStatusCode"] == 200

    print(f"✅ Real HTTP response: {message}")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_real_structured_content_generation() -> None:
    """Test real structured content generation with Gemini."""
    from pydantic import BaseModel

    class UserRequest(BaseModel):
        intent: str
        urgency: int  # 1-10
        needs_human: bool

    class StructuredTestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            # Handle empty conversation start
            if len(context.messages) == 0:
                return ContinueConversationResponse(message="Send me a message to analyze.")

            # Make real structured LLM call
            analysis = context.llm.generate_structured_content(
                messages=context.messages,
                schema=UserRequest,
                system_instruction="Analyze the user's message. Rate urgency 1-10 where 10 is emergency.",  # noqa: E501
            )

            context.store_value("detected_intent", analysis.intent)
            context.store_value("urgency_score", analysis.urgency)
            context.store_value("needs_human", analysis.needs_human)

            return ContinueConversationResponse(
                message=f"I analyzed your message: Intent={analysis.intent}, Urgency={analysis.urgency}/10, Needs human={analysis.needs_human}"  # noqa: E501
            )

    agent = StructuredTestAgent(
        llm_config=GoogleProviderConfig(
            api_key=os.environ["GOOGLE_API_KEY"], model="gemini-2.5-flash"
        ),
        log_level="INFO",
    )

    client = TestClient(agent.app)

    request_data = {
        "metadata": {
            "requestId": "real-structured-test-1",
            "chatbotId": "e2e-test-bot",
            "conversationId": "real-structured-conv",
        },
        "messages": [
            {
                "author": "User",
                "content": "I need help resetting my password please",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ],
    }

    response = client.post("/", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert data["command"]["type"] == "send_message"

    # Verify structured analysis worked
    assert "detected_intent" in data["valuesToSave"]
    assert "urgency_score" in data["valuesToSave"]
    assert "needs_human" in data["valuesToSave"]

    # Verify the analysis makes sense
    intent = data["valuesToSave"]["detected_intent"]
    urgency = data["valuesToSave"]["urgency_score"]

    assert isinstance(intent, str)
    assert len(intent) > 0
    assert isinstance(urgency, int)
    assert 1 <= urgency <= 10

    print(f"✅ Real structured analysis: Intent='{intent}', Urgency={urgency}/10")


@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and os.getenv("OPENAI_API_KEY")),
    reason="Both GOOGLE_API_KEY and OPENAI_API_KEY required for provider comparison",
)
def test_real_provider_comparison() -> None:
    """Compare responses from both real providers."""

    class ComparisonAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            # Handle empty conversation start
            if len(context.messages) == 0:
                return ContinueConversationResponse(message="Send a message to get AI responses.")

            response = context.llm.generate_content(
                messages=context.messages,
                system_instruction="Give a very brief, one sentence answer.",
            )

            provider_name = "Google Gemini" if "gemini" in self.llm_config.model else "OpenAI GPT"
            context.store_value("provider_used", provider_name)
            context.store_value("model_used", self.llm_config.model)

            return ContinueConversationResponse(message=f"[{provider_name}]: {response}")

    # Test with Google
    google_agent = ComparisonAgent(
        llm_config=GoogleProviderConfig(
            api_key=os.environ["GOOGLE_API_KEY"], model="gemini-2.5-flash"
        ),
        log_level="ERROR",
    )

    # Test with OpenAI
    openai_agent = ComparisonAgent(
        llm_config=OpenAIProviderConfig(api_key=os.environ["OPENAI_API_KEY"], model="gpt-5-mini"),
        log_level="ERROR",
    )

    test_message = {
        "metadata": {
            "requestId": "comparison-test",
            "chatbotId": "e2e-test-bot",
            "conversationId": "comparison-conv",
        },
        "messages": [
            {
                "author": "User",
                "content": "What is artificial intelligence?",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ],
    }

    # Get responses from both providers
    google_client = TestClient(google_agent.app)
    openai_client = TestClient(openai_agent.app)

    google_response = google_client.post("/", json=test_message)
    openai_response = openai_client.post("/", json=test_message)

    assert google_response.status_code == 200
    assert openai_response.status_code == 200

    google_data = google_response.json()
    openai_data = openai_response.json()

    google_message = google_data["command"]["payload"]["message"]
    openai_message = openai_data["command"]["payload"]["message"]

    # Both should have responses
    assert len(google_message) > 0
    assert len(openai_message) > 0

    # Verify provider tracking
    assert google_data["valuesToSave"]["provider_used"] == "Google Gemini"
    assert openai_data["valuesToSave"]["provider_used"] == "OpenAI GPT"

    print(f"✅ Google response: {google_message}")
    print(f"✅ OpenAI response: {openai_message}")

    # They should be different (very unlikely to be identical)
    assert google_message != openai_message
