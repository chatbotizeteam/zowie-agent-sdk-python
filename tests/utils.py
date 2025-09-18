"""Test utilities for Zowie Agent SDK tests."""

import json
from datetime import datetime
from typing import Any, Dict, Literal, Optional
from unittest.mock import Mock

from zowie_agent_sdk import Message, Metadata


def create_test_metadata(
    request_id: str = "test-request",
    chatbot_id: str = "test-chatbot",
    conversation_id: str = "test-conversation",
    interaction_id: str = "test-interaction",
) -> Metadata:
    """Create test metadata with optional custom values."""
    return Metadata(
        requestId=request_id,
        chatbotId=chatbot_id,
        conversationId=conversation_id,
        interactionId=interaction_id,
    )


def create_test_message(
    author: Literal["User", "Chatbot"] = "User",
    content: str = "Test message",
    timestamp: Optional[datetime] = None,
) -> Message:
    """Create a test message with optional custom values."""
    if timestamp is None:
        timestamp = datetime.now()

    return Message(author=author, content=content, timestamp=timestamp)


def create_mock_http_response(
    status_code: int = 200, json_data: Optional[Dict[str, Any]] = None, text: Optional[str] = None
) -> Mock:
    """Create a mock HTTP response for testing."""
    mock_response = Mock()
    mock_response.status_code = status_code

    if json_data is None:
        json_data = {"status": "success", "data": "test"}

    mock_response.json.return_value = json_data

    if text is None:
        text = json.dumps(json_data)

    mock_response.text = text
    mock_response.headers = {"Content-Type": "application/json"}
    return mock_response


def assert_valid_agent_response(response_data: Dict[str, Any]) -> None:
    """Assert that response data has valid agent response structure."""
    assert "command" in response_data
    assert "type" in response_data["command"]
    assert "payload" in response_data["command"]

    command_type = response_data["command"]["type"]
    assert command_type in ["send_message", "go_to_next_block"]

    if command_type == "send_message":
        assert "message" in response_data["command"]["payload"]
        assert isinstance(response_data["command"]["payload"]["message"], str)

    elif command_type == "go_to_next_block":
        assert "nextBlockReferenceKey" in response_data["command"]["payload"]
        assert isinstance(response_data["command"]["payload"]["nextBlockReferenceKey"], str)


def assert_events_recorded(response_data: Dict[str, Any]) -> None:
    """Assert that events were properly recorded in response."""
    # Events can be None (no events) or a list
    assert "events" in response_data
    if response_data["events"] is not None:
        assert isinstance(response_data["events"], list)
