"""Fake LLM providers for deterministic testing."""

import time
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

from pydantic import BaseModel

from zowie_agent_sdk.llm.base import BaseLLMProvider
from zowie_agent_sdk.protocol import (
    Event,
    LLMCallEvent,
    LLMCallEventPayload,
    Message,
    Persona,
)

T = TypeVar("T", bound=BaseModel)


class FakeLLMProvider(BaseLLMProvider):
    """
    Fake LLM provider for testing with deterministic responses.

    Features:
    - Configurable canned responses
    - Optional simulated delay
    - Error simulation
    - Call tracking for assertions
    """

    def __init__(
        self,
        config: Any,
        events: List[Event],
        include_persona_default: bool = True,
        include_context_default: bool = True,
        responses: Optional[List[str]] = None,
        structured_responses: Optional[List[Any]] = None,
        simulate_delay: float = 0.0,
        raise_error: Optional[Exception] = None,
    ):
        super().__init__(config, events, include_persona_default, include_context_default)
        self.responses = responses or ["Default test response"]
        self.structured_responses = structured_responses or []
        self.simulate_delay = simulate_delay
        self.raise_error = raise_error
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
        self.structured_call_history: List[Dict[str, Any]] = []

    def generate_content(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> str:
        """Generate deterministic content for testing."""
        # Track the call
        self.call_history.append(
            {
                "messages": messages,
                "system_instruction": system_instruction,
                "include_persona": include_persona,
                "include_context": include_context,
                "persona": persona,
                "context": context,
            }
        )

        # Simulate delay if configured
        if self.simulate_delay > 0:
            time.sleep(self.simulate_delay)

        # Raise error if configured
        if self.raise_error:
            raise self.raise_error

        # Record event if events list provided
        response_text = self.responses[self.call_count % len(self.responses)]
        if events is not None:
            event = LLMCallEvent(
                type="llm_call",
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=system_instruction or "No instruction",
                    response=response_text,
                    durationInMillis=int(self.simulate_delay * 1000),
                ),
            )
            events.append(event)
        elif self.events is not None:
            event = LLMCallEvent(
                type="llm_call",
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=system_instruction or "No instruction",
                    response=response_text,
                    durationInMillis=int(self.simulate_delay * 1000),
                ),
            )
            self.events.append(event)

        # Return next response in sequence
        self.call_count += 1
        return response_text

    def generate_structured_content(
        self,
        messages: List[Message],
        schema: Type[T],
        system_instruction: Optional[str] = None,
        include_persona: Optional[bool] = None,
        include_context: Optional[bool] = None,
        persona: Optional[Persona] = None,
        context: Optional[str] = None,
        events: Optional[List[Event]] = None,
    ) -> T:
        """Generate deterministic structured content for testing."""
        # Track the call
        self.structured_call_history.append(
            {
                "messages": messages,
                "schema": schema,
                "system_instruction": system_instruction,
                "include_persona": include_persona,
                "include_context": include_context,
                "persona": persona,
                "context": context,
            }
        )

        # Simulate delay if configured
        if self.simulate_delay > 0:
            time.sleep(self.simulate_delay)

        # Raise error if configured
        if self.raise_error:
            raise self.raise_error

        # Return structured response if available
        if self.structured_responses:
            response = self.structured_responses[self.call_count % len(self.structured_responses)]
            self.call_count += 1
            # The response should already be of the correct type T
            # as it's provided by the test setup
            # We cast it to T for type safety
            return cast(T, response)

        # Create a default instance of the schema
        # This is a simple fallback - in real tests, structured_responses should be provided
        self.call_count += 1
        # For testing purposes, raise an error if no structured responses are provided
        # Tests should always provide appropriate structured_responses
        raise ValueError(
            f"No structured responses provided for schema {schema.__name__}. "
            "Tests should provide structured_responses matching the expected schema."
        )

    def _prepare_messages(self, messages: List[Message]) -> Any:
        """Prepare messages for the fake provider."""
        # Just return messages as-is for the fake provider
        return messages

    def reset(self) -> None:
        """Reset call tracking for fresh test state."""
        self.call_count = 0
        self.call_history = []
        self.structured_call_history = []

    def assert_called_with(
        self,
        messages: Optional[List[Message]] = None,
        system_instruction: Optional[str] = None,
        persona: Optional[Persona] = None,
        context: Optional[str] = None,
    ) -> bool:
        """Assert that the provider was called with specific arguments."""
        for call in self.call_history:
            matches = True
            if messages is not None and call["messages"] != messages:
                matches = False
            if system_instruction is not None and call["system_instruction"] != system_instruction:
                matches = False
            if persona is not None and call["persona"] != persona:
                matches = False
            if context is not None and call["context"] != context:
                matches = False
            if matches:
                return True
        return False


class FakeLLMProviderSequence(FakeLLMProvider):
    """
    Fake provider that returns different responses in sequence.
    Useful for testing multi-turn conversations.
    """

    def __init__(
        self, config: Any, events: List[Event], response_sequence: List[str], **kwargs: Any
    ) -> None:
        super().__init__(config=config, events=events, responses=response_sequence, **kwargs)
        self.response_sequence = response_sequence

    def generate_content(self, *args: Any, **kwargs: Any) -> str:
        """Generate content from the sequence."""
        if self.call_count >= len(self.response_sequence):
            # Repeat last response if we run out
            response_text = self.response_sequence[-1]
        else:
            response_text = self.response_sequence[self.call_count]

        # Store the response we'll return
        self.responses = [response_text]

        return super().generate_content(*args, **kwargs)


class FakeLLMProviderWithPatterns(FakeLLMProvider):
    """
    Fake provider that returns responses based on input patterns.
    Useful for testing conditional logic.
    """

    def __init__(
        self,
        config: Any,
        events: List[Event],
        pattern_responses: Dict[str, str],
        default_response: str = "No pattern matched",
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, events=events, responses=[default_response], **kwargs)
        self.pattern_responses = pattern_responses
        self.default_response = default_response

    def generate_content(self, messages: List[Message], *args: Any, **kwargs: Any) -> str:
        """Generate content based on message patterns."""
        # Check last message for patterns
        if messages:
            last_message = messages[-1].content.lower()
            for pattern, response in self.pattern_responses.items():
                if pattern.lower() in last_message:
                    self.responses = [response]
                    break
            else:
                self.responses = [self.default_response]

        return super().generate_content(messages, *args, **kwargs)
