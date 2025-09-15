"""Tests for logging functionality across SDK components."""

import logging
from unittest.mock import Mock, patch

import pytest
import requests

from zowie_agent_sdk.auth import AuthValidator
from zowie_agent_sdk.domain import APIKeyAuth
from zowie_agent_sdk.http import HTTPClient


class TestLogging:
    """Test logging functionality in SDK components."""

    def test_http_client_logging(self, caplog):
        """Test HTTPClient logs requests and responses."""
        events = []
        client = HTTPClient(events=events)
        
        with caplog.at_level(logging.DEBUG):
            with patch('requests.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = 'OK'
                mock_response.headers = {}
                mock_request.return_value = mock_response
                
                client.get("https://api.example.com/test", headers={})
        
        # Check debug logs were recorded
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) >= 2
        assert "Making GET request to https://api.example.com/test" in debug_logs[0].message
        assert "GET https://api.example.com/test completed: 200" in debug_logs[1].message

    def test_http_client_timeout_logging(self, caplog):
        """Test HTTPClient logs timeout warnings."""
        events = []
        client = HTTPClient(events=events)
        
        with caplog.at_level(logging.WARNING):
            with patch('requests.request') as mock_request:
                mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
                
                with pytest.raises(requests.exceptions.Timeout):
                    client.get("https://api.example.com/slow", headers={})
        
        # Check warning log was recorded
        warning_logs = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_logs) == 1
        assert "timed out" in warning_logs[0].message
        assert "https://api.example.com/slow" in warning_logs[0].message

    def test_http_client_error_logging(self, caplog):
        """Test HTTPClient logs request errors."""
        events = []
        client = HTTPClient(events=events)
        
        with caplog.at_level(logging.ERROR):
            with patch('requests.request') as mock_request:
                mock_request.side_effect = requests.RequestException("Connection failed")
                
                with pytest.raises(requests.RequestException):
                    client.post("https://api.example.com/error", json={"test": "data"}, headers={})
        
        # Check error log was recorded
        error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_logs) == 1
        assert "failed" in error_logs[0].message
        assert "https://api.example.com/error" in error_logs[0].message

    def test_auth_validator_logging(self, caplog):
        """Test AuthValidator logs authentication events."""
        auth_config = APIKeyAuth(api_key="test-key", header_name="X-API-Key")
        validator = AuthValidator(auth_config)
        
        with caplog.at_level(logging.DEBUG):
            # Create mock request
            mock_request = Mock()
            mock_request.method = "POST"
            mock_request.url.path = "/"
            mock_request.client.host = "127.0.0.1"
            mock_request.headers = {"X-API-Key": "test-key"}
            
            # Valid authentication should succeed
            validator(mock_request)
        
        # Check debug logs were recorded
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) >= 2
        assert "Validating auth for POST / from 127.0.0.1" in debug_logs[0].message
        assert "Authentication successful for POST / from 127.0.0.1" in debug_logs[1].message

    def test_auth_validator_failure_logging(self, caplog):
        """Test AuthValidator logs authentication failures."""
        auth_config = APIKeyAuth(api_key="test-key", header_name="X-API-Key")
        validator = AuthValidator(auth_config)
        
        with caplog.at_level(logging.WARNING):
            # Create mock request with invalid key
            mock_request = Mock()
            mock_request.method = "POST"
            mock_request.url.path = "/"
            mock_request.client.host = "127.0.0.1"
            mock_request.headers = {"X-API-Key": "wrong-key"}
            
            with pytest.raises(Exception):  # HTTPException
                validator(mock_request)
        
        # Check warning log was recorded
        warning_logs = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warning_logs) == 1
        assert "Authentication failed for POST / from 127.0.0.1" in warning_logs[0].message
        assert "Invalid API key" in warning_logs[0].message

    def test_auth_validator_no_auth_logging(self, caplog):
        """Test AuthValidator logs when no auth is configured."""
        validator = AuthValidator(None)
        
        with caplog.at_level(logging.DEBUG):
            # Create mock request
            mock_request = Mock()
            mock_request.method = "GET"
            mock_request.url.path = "/health"
            mock_request.client.host = "127.0.0.1"
            
            validator(mock_request)
        
        # Check debug log was recorded
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) == 1
        assert "No auth configured for GET /health" in debug_logs[0].message