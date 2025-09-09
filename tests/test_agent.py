"""Tests for Agent class."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    AgentResponseContinue,
    AgentResponseFinish,
    Context,
    GoogleConfig,
)
from zowie_agent_sdk.types import ExternalAgentResponse


class MockTestAgent(Agent):
    """Test implementation of Agent class."""

    def handle(self, context: Context):
        """Simple test handler."""
        if context.messages and "stop" in context.messages[-1].content.lower():
            return AgentResponseFinish(
                message="Goodbye!",
                next_block="end_block"
            )
        return AgentResponseContinue(
            message="Test response"
        )


class TestAgentClass:
    """Test Agent base class."""

    def test_agent_initialization(self, google_config):
        """Test agent initialization with config."""
        agent = MockTestAgent(llm_config=google_config)
        
        assert agent.llm_config == google_config
        assert agent.http_timeout_seconds is None
        assert isinstance(agent.app, FastAPI)

    def test_agent_with_auth(self, google_config, api_key_auth):
        """Test agent initialization with authentication."""
        agent = MockTestAgent(
            llm_config=google_config,
            auth_config=api_key_auth
        )
        
        assert agent.auth_validator.auth_config == api_key_auth

    def test_agent_with_timeout(self, google_config):
        """Test agent initialization with timeout."""
        agent = MockTestAgent(
            llm_config=google_config,
            http_timeout_seconds=30.0
        )
        
        assert agent.http_timeout_seconds == 30.0

    @patch('zowie_agent_sdk.agent.LLM')
    @patch('zowie_agent_sdk.agent.HTTPClient')
    def test_handle_request_continue(self, mock_http, mock_llm, google_config, sample_request_json):
        """Test handling request that returns continue response."""
        agent = MockTestAgent(llm_config=google_config)
        client = TestClient(agent.app)
        
        # Mock LLM to avoid actual API calls
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        response = client.post("/", json=sample_request_json)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["command"]["type"] == "send_message"
        assert result["command"]["payload"]["message"] == "Test response"

    @patch('zowie_agent_sdk.agent.LLM')
    @patch('zowie_agent_sdk.agent.HTTPClient')
    def test_handle_request_finish(self, mock_http, mock_llm, google_config):
        """Test handling request that returns finish response."""
        agent = MockTestAgent(llm_config=google_config)
        client = TestClient(agent.app)
        
        # Mock LLM to avoid actual API calls
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        request_json = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Please stop",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        response = client.post("/", json=request_json)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["command"]["type"] == "go_to_next_block"
        assert result["command"]["payload"]["message"] == "Goodbye!"
        assert result["command"]["payload"]["nextBlockReferenceKey"] == "end_block"

    @patch('zowie_agent_sdk.agent.LLM')
    @patch('zowie_agent_sdk.agent.HTTPClient')
    def test_value_storage(self, mock_http, mock_llm, google_config):
        """Test that values are stored correctly during request handling."""
        
        class StorageTestAgent(Agent):
            def handle(self, context: Context):
                context.store_value("test_key", "test_value")
                context.store_value("test_number", 42)
                return AgentResponseContinue(message="Stored values")
        
        agent = StorageTestAgent(llm_config=google_config)
        client = TestClient(agent.app)
        
        # Mock LLM to avoid actual API calls
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        request_json = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [
                {
                    "author": "User",
                    "content": "Test",
                    "timestamp": "2025-01-01T10:00:00.000Z",
                }
            ],
        }
        
        response = client.post("/", json=request_json)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["valuesToSave"]["test_key"] == "test_value"
        assert result["valuesToSave"]["test_number"] == 42

    def test_auth_required(self, google_config, api_key_auth):
        """Test that authentication is enforced when configured."""
        agent = MockTestAgent(
            llm_config=google_config,
            auth_config=api_key_auth
        )
        client = TestClient(agent.app)
        
        request_json = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [],
        }
        
        # Request without auth header should fail
        response = client.post("/", json=request_json)
        assert response.status_code == 401
        
        # Request with correct auth header should succeed
        headers = {"X-API-Key": "test-api-key-123"}
        with patch('zowie_agent_sdk.agent.LLM'), patch('zowie_agent_sdk.agent.HTTPClient'):
            response = client.post("/", json=request_json, headers=headers)
            assert response.status_code == 200

    def test_abstract_method_enforcement(self, google_config):
        """Test that Agent class requires handle method implementation."""
        
        class IncompleteAgent(Agent):
            pass
        
        # Should raise TypeError when trying to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(llm_config=google_config)