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

    def test_http_client_with_custom_timeout(self):
        """Test HTTPClient with custom timeout."""
        events = []
        client = HTTPClient(events=events, default_timeout_seconds=30.0)
        
        assert client.default_timeout_seconds == 30.0

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
        
        # Check event was recorded
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, APICallEvent)
        assert event.payload.url == "https://api.example.com/test"
        assert event.payload.requestMethod == "GET"
        assert event.payload.responseStatusCode == 200

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
        
        # Check event was recorded
        assert len(events) == 1
        event = events[0]
        assert event.payload.requestMethod == "POST"
        assert event.payload.requestBody == json.dumps(request_data)

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