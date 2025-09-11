"""Tests for Context class."""

from unittest.mock import Mock

from zowie_agent_sdk import Context
from zowie_agent_sdk.http import HTTPClient
from zowie_agent_sdk.llm import LLM


class TestContext:
    """Test Context class."""

    def test_context_initialization(self, sample_metadata, sample_messages):
        """Test context initialization with all required parameters."""
        mock_llm = Mock(spec=LLM)
        mock_http = Mock(spec=HTTPClient)
        store_value_fn = Mock()
        
        context = Context(
            metadata=sample_metadata,
            messages=sample_messages,
            context="Test context string",
            store_value=store_value_fn,
            llm=mock_llm,
            http=mock_http,
        )
        
        assert context.metadata == sample_metadata
        assert context.messages == sample_messages
        assert context.context == "Test context string"
        assert context.store_value == store_value_fn
        assert context.llm == mock_llm
        assert context.http == mock_http

    def test_context_with_none_context(self, sample_metadata, sample_messages):
        """Test context initialization with None context string."""
        mock_llm = Mock(spec=LLM)
        mock_http = Mock(spec=HTTPClient)
        store_value_fn = Mock()
        
        context = Context(
            metadata=sample_metadata,
            messages=sample_messages,
            context=None,
            store_value=store_value_fn,
            llm=mock_llm,
            http=mock_http,
        )
        
        assert context.context is None

    def test_store_value_function_called(self, sample_metadata, sample_messages):
        """Test that store_value function is callable and works correctly."""
        mock_llm = Mock(spec=LLM)
        mock_http = Mock(spec=HTTPClient)
        store_value_fn = Mock()
        
        context = Context(
            metadata=sample_metadata,
            messages=sample_messages,
            context=None,
            store_value=store_value_fn,
            llm=mock_llm,
            http=mock_http,
        )
        
        # Test calling store_value
        context.store_value("test_key", "test_value")
        store_value_fn.assert_called_once_with("test_key", "test_value")
        
        # Test with different types
        context.store_value("number", 42)
        context.store_value("dict", {"key": "value"})
        context.store_value("list", [1, 2, 3])
        
        assert store_value_fn.call_count == 4

    def test_metadata_access(self, sample_metadata, sample_messages):
        """Test accessing metadata properties."""
        mock_llm = Mock(spec=LLM)
        mock_http = Mock(spec=HTTPClient)
        
        context = Context(
            metadata=sample_metadata,
            messages=sample_messages,
            context=None,
            store_value=Mock(),
            llm=mock_llm,
            http=mock_http,
        )
        
        assert context.metadata.requestId == "test-request-123"
        assert context.metadata.chatbotId == "test-chatbot-456"
        assert context.metadata.conversationId == "test-conversation-789"
        assert context.metadata.interactionId == "test-interaction-001"

    def test_messages_access(self, sample_metadata, sample_messages):
        """Test accessing messages."""
        mock_llm = Mock(spec=LLM)
        mock_http = Mock(spec=HTTPClient)
        
        context = Context(
            metadata=sample_metadata,
            messages=sample_messages,
            context=None,
            store_value=Mock(),
            llm=mock_llm,
            http=mock_http,
        )
        
        assert len(context.messages) == 2
        assert context.messages[0].author == "User"
        assert context.messages[0].content == "Hello, I need help"
        assert context.messages[1].author == "Chatbot"
        assert context.messages[1].content == "I'm here to help you!"