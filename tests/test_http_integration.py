"""Tests for HTTP client integration with proper mocking."""

import json
from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

from tests.utils import create_mock_http_response, create_test_message, create_test_metadata
from zowie_agent_sdk import (
    Agent,
    AgentResponse,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    Message,
    Metadata,
)
from zowie_agent_sdk.protocol import Event


@patch("requests.request")
def test_http_client_get_integration(mock_request: Mock) -> None:
    """Test HTTP client GET integration."""

    # Mock requests response
    mock_response = create_mock_http_response(json_data={"data": "success"})
    mock_request.return_value = mock_response

    class HttpGetAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            response = context.http.get(
                url="https://api.example.com/data", headers={"Authorization": "Bearer token"}
            )

            context.store_value("api_status", response.status_code)
            context.store_value("api_data", response.json())

            return ContinueConversationResponse(message=f"API returned: {response.json()['data']}")

    agent = HttpGetAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context
    metadata = create_test_metadata(request_id="test", chatbot_id="test", conversation_id="test")
    messages = [create_test_message(content="Check API", timestamp=datetime(2025, 9, 15, 0, 0, 0))]
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
    assert "API returned: success" in result.message

    # Verify the request was made correctly
    mock_request.assert_called_once_with(
        method="GET",
        url="https://api.example.com/data",
        headers={"Authorization": "Bearer token"},
        json=None,
        timeout=10.0,  # Default timeout
    )

    # Verify events were recorded
    assert len(events) == 1
    assert events[0].type == "api_call"
    assert events[0].payload.requestMethod == "GET"
    assert events[0].payload.responseStatusCode == 200


@patch("requests.request")
def test_http_client_post_integration(mock_request: Mock) -> None:
    """Test HTTP client POST integration."""

    # Mock requests response
    mock_response = Mock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": "123", "status": "created"}
    mock_response.text = '{"id": "123", "status": "created"}'
    mock_response.headers = {"Content-Type": "application/json"}
    mock_request.return_value = mock_response

    class HttpPostAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            data = {"name": "test", "value": 42}

            response = context.http.post(
                url="https://api.example.com/create",
                json=data,
                headers={"Content-Type": "application/json"},
            )

            return ContinueConversationResponse(
                message=f"Created resource with ID: {response.json()['id']}"
            )

    agent = HttpPostAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")
    messages = [
        Message(author="User", content="Create resource", timestamp=datetime(2025, 9, 15, 0, 0, 0))
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
    assert "Created resource with ID: 123" in result.message

    # Verify the request was made correctly
    mock_request.assert_called_once_with(
        method="POST",
        url="https://api.example.com/create",
        headers={"Content-Type": "application/json"},
        json={"name": "test", "value": 42},
        timeout=10.0,
    )


@patch("requests.request")
def test_http_client_error_handling(mock_request: Mock) -> None:
    """Test HTTP client error handling."""

    # Mock requests to raise a requests exception
    from requests.exceptions import ConnectionError

    mock_request.side_effect = ConnectionError("Network error")

    class HttpErrorAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            try:
                context.http.get(url="https://api.example.com/fail", headers={})
                return ContinueConversationResponse(message="Unexpected success")
            except Exception as e:
                context.store_value("error_message", str(e))
                return ContinueConversationResponse(message="API call failed as expected")

    agent = HttpErrorAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")
    messages = [
        Message(author="User", content="Test error", timestamp=datetime(2025, 9, 15, 0, 0, 0))
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
    assert "API call failed as expected" in result.message

    # Events should still be recorded even on error
    assert len(events) == 1
    assert events[0].type == "api_call"


@patch("requests.request")
def test_http_client_different_methods(mock_request: Mock) -> None:
    """Test HTTP client with different HTTP methods."""

    # Mock requests response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"method": "success"}
    mock_response.text = '{"method": "success"}'
    mock_response.headers = {}
    mock_request.return_value = mock_response

    class MultiMethodAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            last_message = context.messages[-1].content.lower()

            if "get" in last_message:
                context.http.get("https://api.example.com/get", {})
                method = "GET"
            elif "post" in last_message:
                context.http.post("https://api.example.com/post", {}, {})
                method = "POST"
            elif "put" in last_message:
                context.http.put("https://api.example.com/put", {}, {})
                method = "PUT"
            elif "patch" in last_message:
                context.http.patch("https://api.example.com/patch", {}, {})
                method = "PATCH"
            elif "delete" in last_message:
                context.http.delete("https://api.example.com/delete", {})
                method = "DELETE"
            else:
                return ContinueConversationResponse(message="Unknown method")

            context.store_value("method_used", method)
            return ContinueConversationResponse(message=f"{method} request completed")

    agent = MultiMethodAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Test different methods
    methods = ["get", "post", "put", "patch", "delete"]

    for method in methods:
        mock_request.reset_mock()

        metadata = Metadata(requestId=f"test-{method}", chatbotId="test", conversationId="test")
        messages = [
            Message(
                author="User", content=f"Test {method}", timestamp=datetime(2025, 9, 15, 0, 0, 0)
            )
        ]  # noqa: E501
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
        assert f"{method.upper()} request completed" in result.message

        # Verify correct HTTP method was used
        mock_request.assert_called_once()
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["method"] == method.upper()

        # Verify event was recorded
        assert len(events) == 1
        assert events[0].type == "api_call"
        assert events[0].payload.requestMethod == method.upper()


@patch("requests.request")
def test_http_client_timeout_configuration(mock_request: Mock) -> None:
    """Test HTTP client timeout configuration."""

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_response.text = "{}"
    mock_response.headers = {}
    mock_request.return_value = mock_response

    class TimeoutAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            # Test custom timeout
            context.http.get(url="https://api.example.com/slow", headers={}, timeout_seconds=30.0)

            return ContinueConversationResponse(message="Request completed")

    agent = TimeoutAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"),
        http_timeout_seconds=5.0,  # Agent default timeout
        log_level="ERROR",
    )

    # Create context
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")
    messages = [
        Message(author="User", content="Test timeout", timestamp=datetime(2025, 9, 15, 0, 0, 0))
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

    # Verify custom timeout was used (not agent default)
    mock_request.assert_called_once()
    call_kwargs = mock_request.call_args[1]
    assert call_kwargs["timeout"] == 30.0


@patch("requests.request")
def test_http_client_header_inclusion(mock_request: Mock) -> None:
    """Test HTTP client header inclusion/exclusion."""

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_response.text = "{}"
    mock_response.headers = {"Content-Type": "application/json"}
    mock_request.return_value = mock_response

    class HeaderTestAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            # Test with headers excluded
            context.http.get(
                url="https://api.example.com/test",
                headers={"Authorization": "Bearer secret"},
                include_headers=False,
            )

            return ContinueConversationResponse(message="Headers test")

    agent = HeaderTestAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"),
        include_http_headers_by_default=True,  # Agent default
        log_level="ERROR",
    )

    # Create context
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")
    messages = [
        Message(author="User", content="Test headers", timestamp=datetime(2025, 9, 15, 0, 0, 0))
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

    # Verify event was recorded but headers were excluded
    assert len(events) == 1
    event = events[0]
    assert event.type == "api_call"
    # Headers should be excluded due to include_headers=False
    assert event.payload.requestHeaders == {}
    assert event.payload.responseHeaders == {}


@patch("requests.request")
def test_http_client_json_serialization(mock_request: Mock) -> None:
    """Test HTTP client JSON serialization in events."""

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "ok"}
    mock_response.text = '{"result": "ok"}'
    mock_response.headers = {}
    mock_request.return_value = mock_response

    class JsonAgent(Agent):
        def handle(self, context: Context) -> AgentResponse:
            complex_data = {
                "user": {"id": 123, "name": "Test"},
                "items": [{"id": 1, "qty": 2}],
                "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
            }

            context.http.post(
                url="https://api.example.com/complex",
                json=complex_data,
                headers={"Content-Type": "application/json"},
            )

            return ContinueConversationResponse(message="JSON test completed")

    agent = JsonAgent(
        llm_config=GoogleProviderConfig(api_key="test", model="gemini-2.5-flash"), log_level="ERROR"
    )

    # Create context
    metadata = Metadata(requestId="test", chatbotId="test", conversationId="test")
    messages = [
        Message(author="User", content="Test JSON", timestamp=datetime(2025, 9, 15, 0, 0, 0))
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

    # Verify event was recorded with proper JSON serialization
    assert len(events) == 1
    event = events[0]
    assert event.type == "api_call"

    # Request body should be valid JSON string
    request_body = event.payload.requestBody
    assert isinstance(request_body, str)
    parsed_body = json.loads(request_body)
    assert parsed_body["user"]["id"] == 123
    assert len(parsed_body["items"]) == 1
