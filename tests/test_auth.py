"""Tests for authentication module."""

import base64

import pytest
from fastapi import HTTPException

from zowie_agent_sdk.auth import AuthValidator


class TestAuthValidator:
    """Test AuthValidator class."""

    def test_no_auth_config(self, mock_request):
        """Test that no authentication is required when auth_config is None."""
        validator = AuthValidator(auth_config=None)
        # Should not raise any exception
        validator(mock_request)

    def test_api_key_auth_valid(self, api_key_auth, mock_request):
        """Test valid API key authentication."""
        validator = AuthValidator(auth_config=api_key_auth)
        mock_request.headers = {"X-API-Key": "test-api-key-123"}
        
        # Should not raise any exception
        validator(mock_request)

    def test_api_key_auth_missing_header(self, api_key_auth, mock_request):
        """Test API key authentication with missing header."""
        validator = AuthValidator(auth_config=api_key_auth)
        mock_request.headers = {}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Missing X-API-Key header" in str(exc_info.value.detail)

    def test_api_key_auth_invalid_key(self, api_key_auth, mock_request):
        """Test API key authentication with invalid key."""
        validator = AuthValidator(auth_config=api_key_auth)
        mock_request.headers = {"X-API-Key": "wrong-api-key"}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value.detail)

    def test_basic_auth_valid(self, basic_auth, mock_request):
        """Test valid basic authentication."""
        validator = AuthValidator(auth_config=basic_auth)
        credentials = base64.b64encode(b"testuser:testpass123").decode("utf-8")
        mock_request.headers = {"Authorization": f"Basic {credentials}"}
        
        # Should not raise any exception
        validator(mock_request)

    def test_basic_auth_missing_header(self, basic_auth, mock_request):
        """Test basic authentication with missing header."""
        validator = AuthValidator(auth_config=basic_auth)
        mock_request.headers = {}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Missing or invalid Authorization header" in str(exc_info.value.detail)

    def test_basic_auth_invalid_format(self, basic_auth, mock_request):
        """Test basic authentication with invalid format."""
        validator = AuthValidator(auth_config=basic_auth)
        mock_request.headers = {"Authorization": "Basic invalid-base64"}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Invalid Basic auth format" in str(exc_info.value.detail)

    def test_basic_auth_wrong_credentials(self, basic_auth, mock_request):
        """Test basic authentication with wrong credentials."""
        validator = AuthValidator(auth_config=basic_auth)
        credentials = base64.b64encode(b"wronguser:wrongpass").decode("utf-8")
        mock_request.headers = {"Authorization": f"Basic {credentials}"}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Invalid credentials" in str(exc_info.value.detail)

    def test_bearer_auth_valid(self, bearer_auth, mock_request):
        """Test valid bearer token authentication."""
        validator = AuthValidator(auth_config=bearer_auth)
        mock_request.headers = {"Authorization": "Bearer test-bearer-token-xyz"}
        
        # Should not raise any exception
        validator(mock_request)

    def test_bearer_auth_missing_header(self, bearer_auth, mock_request):
        """Test bearer authentication with missing header."""
        validator = AuthValidator(auth_config=bearer_auth)
        mock_request.headers = {}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Missing or invalid Authorization header" in str(exc_info.value.detail)

    def test_bearer_auth_invalid_token(self, bearer_auth, mock_request):
        """Test bearer authentication with invalid token."""
        validator = AuthValidator(auth_config=bearer_auth)
        mock_request.headers = {"Authorization": "Bearer wrong-token"}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Invalid bearer token" in str(exc_info.value.detail)

    def test_bearer_auth_wrong_prefix(self, bearer_auth, mock_request):
        """Test bearer authentication with wrong prefix."""
        validator = AuthValidator(auth_config=bearer_auth)
        mock_request.headers = {"Authorization": "Token test-bearer-token-xyz"}
        
        with pytest.raises(HTTPException) as exc_info:
            validator(mock_request)
        
        assert exc_info.value.status_code == 401
        assert "Missing or invalid Authorization header" in str(exc_info.value.detail)