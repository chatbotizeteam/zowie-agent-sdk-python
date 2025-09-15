"""Tests for HTTPClient class."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from zowie_agent_sdk.http import HTTPClient
from zowie_agent_sdk.protocol import APICallEvent


class TestHTTPClient:
    """Test HTTPClient class."""

    def test_http_client_initialization(self):
        """Test HTTPClient initialization."""
        events = []
        client = HTTPClient(events=events)
        
        assert client.events is events
        assert client.default_timeout_seconds == 10.0
        assert client.include_headers_by_default is True

    def test_http_client_with_custom_timeout(self):
        """Test HTTPClient with custom timeout."""
        events = []
        client = HTTPClient(events=events, default_timeout_seconds=30.0)
        
        assert client.default_timeout_seconds == 30.0

    def test_http_client_with_headers_disabled(self):
        """Test HTTPClient with headers disabled by default."""
        events = []
        client = HTTPClient(events=events, include_headers_by_default=False)
        
        assert client.include_headers_by_default is False

    @patch('requests.request')
    def test_get_request(self, mock_request):
        """Test GET request."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"result": "success"}'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        response = client.get(
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer token"}
        )
        
        assert response == mock_response
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer token"},
            json=None,
            timeout=10.0
        )
        
        # Check event was recorded with complete details
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, APICallEvent)
        assert event.payload.url == "https://api.example.com/test"
        assert event.payload.requestMethod == "GET"
        assert event.payload.responseStatusCode == 200
        
        # Verify complete event payload
        assert event.payload.requestHeaders == {"Authorization": "Bearer token"}
        assert event.payload.requestBody is None  # GET request has no body
        assert event.payload.responseHeaders == {"Content-Type": "application/json"}
        assert event.payload.responseBody == '{"result": "success"}'
        assert event.payload.durationInMillis >= 0  # May be 0 when mocked
        assert event.type == "api_call"

    @patch('requests.request')
    def test_post_request(self, mock_request):
        """Test POST request."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.text = '{"id": 123}'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        request_data = {"name": "test", "value": 42}
        response = client.post(
            url="https://api.example.com/create",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response == mock_response
        mock_request.assert_called_once_with(
            method="POST",
            url="https://api.example.com/create",
            headers={"Content-Type": "application/json"},
            json=request_data,
            timeout=10.0
        )
        
        # Check event was recorded with complete POST details
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, APICallEvent)
        assert event.payload.requestMethod == "POST"
        assert event.payload.requestBody == json.dumps(request_data)
        
        # Verify complete POST event payload
        assert event.payload.url == "https://api.example.com/create"
        assert event.payload.requestHeaders == {"Content-Type": "application/json"}
        assert event.payload.responseStatusCode == 201
        assert event.payload.responseBody == '{"id": 123}'
        assert event.payload.responseHeaders == {"Content-Type": "application/json"}
        assert event.payload.durationInMillis >= 0  # May be 0 when mocked

    @patch('requests.request')
    def test_put_request(self, mock_request):
        """Test PUT request."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"updated": true}'
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        update_data = {"name": "updated"}
        response = client.put(
            url="https://api.example.com/update/123",
            json=update_data,
            headers={"Authorization": "Bearer token"}
        )
        
        assert response == mock_response
        mock_request.assert_called_once_with(
            method="PUT",
            url="https://api.example.com/update/123",
            headers={"Authorization": "Bearer token"},
            json=update_data,
            timeout=10.0
        )
        
        # Check event with complete PUT details
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, APICallEvent)
        assert event.payload.requestMethod == "PUT"
        assert event.payload.url == "https://api.example.com/update/123"
        assert event.payload.responseStatusCode == 200
        assert event.payload.responseBody == '{"updated": true}'
        assert event.payload.requestBody == json.dumps(update_data)
        assert event.payload.durationInMillis >= 0  # May be 0 when mocked

    @patch('requests.request')
    def test_patch_request(self, mock_request):
        """Test PATCH request."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"patched": true}'
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        patch_data = {"status": "active"}
        response = client.patch(
            url="https://api.example.com/patch/456",
            json=patch_data,
            headers={"Authorization": "Bearer token"}
        )
        
        assert response == mock_response
        mock_request.assert_called_once_with(
            method="PATCH",
            url="https://api.example.com/patch/456",
            headers={"Authorization": "Bearer token"},
            json=patch_data,
            timeout=10.0
        )
        
        # Check event with complete PATCH details
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, APICallEvent)
        assert event.payload.requestMethod == "PATCH"
        assert event.payload.url == "https://api.example.com/patch/456"
        assert event.payload.responseStatusCode == 200
        assert event.payload.responseBody == '{"patched": true}'
        assert event.payload.requestBody == json.dumps(patch_data)
        assert event.payload.durationInMillis >= 0  # May be 0 when mocked

    @patch('requests.request')
    def test_delete_request(self, mock_request):
        """Test DELETE request."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.text = ''
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        response = client.delete(
            url="https://api.example.com/delete/123",
            headers={"Authorization": "Bearer token"}
        )
        
        assert response == mock_response
        mock_request.assert_called_once_with(
            method="DELETE",
            url="https://api.example.com/delete/123",
            headers={"Authorization": "Bearer token"},
            json=None,
            timeout=10.0
        )
        
        # Check event with complete DELETE details
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, APICallEvent)
        assert event.payload.requestMethod == "DELETE"
        assert event.payload.responseStatusCode == 204
        assert event.payload.url == "https://api.example.com/delete/123"
        assert event.payload.requestBody is None  # DELETE typically has no body
        assert event.payload.responseBody == ''  # 204 has no content
        assert event.payload.requestHeaders == {"Authorization": "Bearer token"}
        assert event.payload.durationInMillis >= 0  # May be 0 when mocked

    @patch('requests.request')
    def test_custom_timeout(self, mock_request):
        """Test request with custom timeout."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        client.get(
            url="https://api.example.com/test",
            headers={},
            timeout_seconds=60.0
        )
        
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/test",
            headers={},
            json=None,
            timeout=60.0
        )

    @patch('requests.request')
    def test_timeout_exception(self, mock_request):
        """Test handling of timeout exception."""
        events = []
        client = HTTPClient(events=events)
        
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(requests.exceptions.Timeout):
            client.get(
                url="https://api.example.com/slow",
                headers={}
            )
        
        # Check event was recorded for timeout
        assert len(events) == 1
        event = events[0]
        assert event.payload.responseStatusCode == 504
        assert "Request timed out" in event.payload.responseBody

    @patch('requests.request')
    def test_request_exception(self, mock_request):
        """Test handling of general request exception."""
        events = []
        client = HTTPClient(events=events)
        
        mock_request.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(requests.RequestException):
            client.post(
                url="https://api.example.com/error",
                json={"test": "data"},
                headers={}
            )
        
        # Check event was recorded for error
        assert len(events) == 1
        event = events[0]
        assert event.payload.responseStatusCode == 0
        assert "Connection error" in event.payload.responseBody

    @patch('requests.request')
    def test_event_duration(self, mock_request):
        """Test that event duration is recorded."""
        events = []
        client = HTTPClient(events=events)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        client.get(
            url="https://api.example.com/test",
            headers={}
        )
        
        assert len(events) == 1
        event = events[0]
        # Duration should be recorded (greater than 0)
        assert event.payload.durationInMillis >= 0

    @patch('requests.request')
    def test_headers_included_by_default(self, mock_request):
        """Test that headers are included in events by default."""
        events = []
        client = HTTPClient(events=events, include_headers_by_default=True)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        client.get(
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer secret-token"}
        )
        
        assert len(events) == 1
        event = events[0]
        assert event.payload.requestHeaders == {"Authorization": "Bearer secret-token"}
        assert event.payload.responseHeaders == {"Content-Type": "application/json"}

    @patch('requests.request')
    def test_headers_excluded_by_default(self, mock_request):
        """Test that headers are excluded from events when disabled."""
        events = []
        client = HTTPClient(events=events, include_headers_by_default=False)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        client.get(
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer secret-token"}
        )
        
        assert len(events) == 1
        event = events[0]
        assert event.payload.requestHeaders == {}
        assert event.payload.responseHeaders == {}

    @patch('requests.request')
    def test_headers_override_per_call(self, mock_request):
        """Test per-call override of header inclusion."""
        events = []
        client = HTTPClient(events=events, include_headers_by_default=False)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'OK'
        mock_response.headers = {"Content-Type": "application/json"}
        mock_request.return_value = mock_response
        
        # Override to include headers for this specific call
        client.get(
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer secret-token"},
            include_headers=True
        )
        
        assert len(events) == 1
        event = events[0]
        assert event.payload.requestHeaders == {"Authorization": "Bearer secret-token"}
        assert event.payload.responseHeaders == {"Content-Type": "application/json"}