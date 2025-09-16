"""Tests for LLM integration with proper mocking."""

from datetime import datetime
from typing import Any, List
from unittest.mock import Mock, patch

from pydantic import BaseModel

from zowie_agent_sdk import (
    Agent,
    AgentResponse,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    Message,
    OpenAIProviderConfig,
)
from zowie_agent_sdk.llm.base import LLM
from zowie_agent_sdk.protocol import Event, LLMCallEvent, LLMCallEventPayload


class LLMTestStructuredResponse(BaseModel):
    """Test schema for structured responses."""

    intent: str
    confidence: float
    needs_escalation: bool


@patch("zowie_agent_sdk.llm.google.GoogleProvider.generate_content")
def test_google_llm_integration(mock_generate: Mock) -> None:
    """Test Google LLM integration with proper mocking."""
    # Mock the LLM response
    mock_generate.return_value = "I can help you with that!"

    class LLMTestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            response = context.llm.generate_content(
                messages=context.messages, system_instruction="You are helpful."
            )
            return ContinueConversationResponse(message=response)

    agent = LLMTestAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context manually to test LLM
    from zowie_agent_sdk import Metadata

    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")

    messages = [Message(author="User", content="Hello", timestamp=datetime(2024, 1, 1, 0, 0, 0))]

    events: List[Event] = []
    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: None,
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)
    assert "I can help you with that!" in result.message


@patch("zowie_agent_sdk.llm.google.GoogleProvider.generate_structured_content")
def test_google_structured_llm_integration(mock_generate_structured: Mock) -> None:
    """Test Google structured LLM integration."""

    # Mock the structured response
    mock_generate_structured.return_value = LLMTestStructuredResponse(
        intent="help_request", confidence=0.95, needs_escalation=False
    )

    class StructuredLLMAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            analysis = context.llm.generate_structured_content(
                messages=context.messages,
                schema=LLMTestStructuredResponse,
                system_instruction="Analyze the user's intent.",
            )

            if analysis.needs_escalation:
                return ContinueConversationResponse(message="Escalating...")
            else:
                return ContinueConversationResponse(
                    message=f"Intent: {analysis.intent} (confidence: {analysis.confidence})"
                )

    agent = StructuredLLMAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Test the agent
    from zowie_agent_sdk import Metadata

    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")

    messages = [
        Message(author="User", content="I need help", timestamp=datetime(2024, 1, 1, 0, 0, 0))
    ]

    events: List[Event] = []
    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: None,
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)
    assert "Intent: help_request" in result.message
    assert "confidence: 0.95" in result.message


@patch("zowie_agent_sdk.llm.openai.OpenAIProvider.generate_content")
def test_openai_llm_integration(mock_generate: Mock) -> None:
    """Test OpenAI LLM integration."""
    # Mock the LLM response
    mock_generate.return_value = "OpenAI response here!"

    class OpenAITestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            response = context.llm.generate_content(
                messages=context.messages, system_instruction="You are an OpenAI assistant."
            )
            return ContinueConversationResponse(message=response)

    agent = OpenAITestAgent(
        llm_config=OpenAIProviderConfig(api_key="test", model="gpt-5-mini"), log_level="ERROR"
    )

    # Create context
    from zowie_agent_sdk import Metadata

    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")

    messages = [
        Message(author="User", content="Hello OpenAI", timestamp=datetime(2024, 1, 1, 0, 0, 0))
    ]

    events: List[Event] = []
    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: None,
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)
    assert "OpenAI response here!" in result.message


def test_llm_provider_configuration() -> None:
    """Test LLM provider configuration."""

    # Test Google config
    google_config = GoogleProviderConfig(api_key="google-test", model="gemini-2.5-flash")
    llm = LLM(config=google_config, events=[], persona=None, context=None)
    assert llm.provider is not None
    assert llm.provider.model == "gemini-2.5-flash"

    # Test OpenAI config
    openai_config = OpenAIProviderConfig(api_key="openai-test", model="gpt-5-mini")
    llm = LLM(config=openai_config, events=[], persona=None, context=None)
    assert llm.provider is not None
    assert llm.provider.model == "gpt-5-mini"


def test_llm_error_handling() -> None:
    """Test LLM error handling."""

    class ErrorLLMAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            try:
                # This will fail because LLM provider is not configured
                response = context.llm.generate_content(
                    messages=context.messages, system_instruction="Test"
                )
                return ContinueConversationResponse(message=response)
            except Exception as e:
                return ContinueConversationResponse(message=f"Error: {str(e)}")

    # Create agent with None config (no LLM provider)
    agent = ErrorLLMAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Manually set provider to None to simulate error
    agent._base_llm.provider = None

    # Create context
    from zowie_agent_sdk import Metadata

    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")

    messages = [Message(author="User", content="Test", timestamp=datetime(2024, 1, 1, 0, 0, 0))]

    events: List[Event] = []
    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: None,
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)
    assert "Error:" in result.message


@patch("zowie_agent_sdk.llm.google.GoogleProvider.generate_content")
def test_llm_with_persona_and_context(mock_generate: Mock) -> None:
    """Test LLM with persona and context."""
    from zowie_agent_sdk import Metadata, Persona

    # Mock the LLM response
    mock_generate.return_value = "Response with persona"

    class PersonaAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            response = context.llm.generate_content(
                messages=context.messages,
                system_instruction="Help the user",
                include_persona=True,
                include_context=True,
            )
            return ContinueConversationResponse(message=response)

    agent = PersonaAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context with persona
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")

    messages = [Message(author="User", content="Help me", timestamp=datetime(2024, 1, 1, 0, 0, 0))]

    persona = Persona(
        name="Assistant", business_context="Customer support", tone_of_voice="Friendly"
    )

    events: List[Event] = []
    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: None,
        llm=agent._base_llm,
        http=agent._base_http_client,
        persona=persona,
        context="Important customer",
        events=events,
    )

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)
    assert "Response with persona" in result.message

    # Verify the mock was called with persona and context
    mock_generate.assert_called_once()
    call_kwargs = mock_generate.call_args[1]
    assert call_kwargs["persona"] == persona
    assert call_kwargs["context"] == "Important customer"


def test_llm_timeout_handling() -> None:
    """Test LLM handling with FakeLLMProvider (without actual delays)."""
    from tests.fake_providers import FakeLLMProvider
    from zowie_agent_sdk import Metadata

    class TimeoutTestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            try:
                # This should timeout if provider takes too long
                response = context.llm.generate_content(
                    messages=context.messages, system_instruction="Respond quickly"
                )
                context.store_value("response_received", True)
                return ContinueConversationResponse(message=response)
            except Exception as e:
                context.store_value("timeout_error", str(e))
                return ContinueConversationResponse(message="Request timed out")

    # Create agent with a fake provider that simulates delay
    agent = TimeoutTestAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Replace the LLM provider with our fake one that has a delay
    fake_provider = FakeLLMProvider(
        config=agent.llm_config,
        events=[],
        responses=["Delayed response"],
        simulate_delay=0.0,  # No actual delay for speed
    )
    agent._base_llm.provider = fake_provider

    # Create context
    metadata = Metadata(requestId="timeout-test", chatbotId="test", conversationId="test")

    messages = [
        Message(author="User", content="Test timeout", timestamp=datetime(2024, 1, 1, 0, 0, 0))
    ]

    events: List[Event] = []
    values_stored = {}

    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: values_stored.update({k: v}),
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    # Test the response handling
    result = agent.handle(context)
    
    assert isinstance(result, ContinueConversationResponse)
    assert values_stored.get("response_received") is True
    assert "Delayed response" in result.message
    
    # Verify the fake provider was called
    assert fake_provider.call_count == 1
    assert fake_provider.assert_called_with(messages=messages)


def test_llm_timeout_with_error() -> None:
    """Test LLM timeout error scenario."""
    from tests.fake_providers import FakeLLMProvider
    from zowie_agent_sdk import Metadata

    class TimeoutErrorAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            try:
                response = context.llm.generate_content(
                    messages=context.messages, system_instruction="This will timeout"
                )
                return ContinueConversationResponse(message=response)
            except TimeoutError:
                context.store_value("timeout_occurred", True)
                return ContinueConversationResponse(message="Request timed out")
            except Exception as e:
                context.store_value("other_error", str(e))
                return ContinueConversationResponse(message=f"Error: {str(e)}")

    agent = TimeoutErrorAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Configure fake provider to raise timeout error
    fake_provider = FakeLLMProvider(
        config=agent.llm_config, events=[], raise_error=TimeoutError("LLM request timed out")
    )
    agent._base_llm.provider = fake_provider

    # Create context
    metadata = Metadata(requestId="timeout-error-test", chatbotId="test", conversationId="test")

    messages = [
        Message(
            author="User", content="Test timeout error", timestamp=datetime(2024, 1, 1, 0, 0, 0)
        )
    ]

    events: List[Event] = []
    values_stored = {}

    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: values_stored.update({k: v}),
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)
    assert "Request timed out" in result.message
    assert values_stored.get("timeout_occurred") is True


@patch("zowie_agent_sdk.llm.google.GoogleProvider.generate_content")
def test_llm_events_tracking(mock_generate: Mock) -> None:
    """Test that LLM calls generate events."""
    from zowie_agent_sdk import Metadata

    # Mock the LLM response
    mock_generate.return_value = "Test response"

    class EventTrackingAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            initial_event_count = len(context.events)

            response = context.llm.generate_content(
                messages=context.messages, system_instruction="Test"
            )

            # Store event count for verification
            context.store_value("initial_events", initial_event_count)
            context.store_value("final_events", len(context.events))

            return ContinueConversationResponse(message=response)

    agent = EventTrackingAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")

    messages = [Message(author="User", content="Test", timestamp=datetime(2024, 1, 1, 0, 0, 0))]

    events: List[Event] = []
    context = Context(
        metadata=metadata,
        messages=messages,
        store_value=lambda k, v: None,
        llm=agent._base_llm,
        http=agent._base_http_client,
        events=events,
    )

    # Mock the events being added by the provider
    def mock_generate_with_events(*args: Any, **kwargs: Any) -> str:
        # Simulate adding an event
        events.append(
            LLMCallEvent(
                type="llm_call",
                payload=LLMCallEventPayload(
                    model="gemini-2.5-flash",
                    prompt="test prompt",
                    response="test response",
                    durationInMillis=100,
                ),
            )
        )
        return "Test response"

    mock_generate.side_effect = mock_generate_with_events

    result = agent.handle(context)
    assert isinstance(result, ContinueConversationResponse)

    # Should have generated events
    assert len(events) > 0
    assert any(event.type == "llm_call" for event in events)
