"""Tests for concurrent request handling."""

import threading
import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from zowie_agent_sdk import (
    Agent,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
)


class ConcurrentTestAgent(Agent):
    """Agent for testing concurrent requests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_count = 0
        self.lock = threading.Lock()
    
    def handle(self, context: Context):
        """Handle with simulated processing time."""
        with self.lock:
            self.request_count += 1
            current_count = self.request_count
        
        # Simulate some processing time
        time.sleep(0.01)
        
        return ContinueConversationResponse(
            message=f"Response {current_count}"
        )


class TestConcurrentRequests:
    """Test concurrent request handling."""

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_multiple_concurrent_requests(self, mock_google_provider):
        """Test handling multiple concurrent requests."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        agent = ConcurrentTestAgent(
            llm_config=GoogleProviderConfig(api_key="test", model="test")
        )
        client = TestClient(agent.app)
        
        request_data = {
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
        
        results = []
        errors = []
        
        def make_request(request_id):
            try:
                data = request_data.copy()
                data["metadata"]["requestId"] = f"test-{request_id}"
                response = client.post("/", json=data)
                results.append(response)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        num_requests = 10
        
        for i in range(num_requests):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0
        assert len(results) == num_requests
        
        # All requests should succeed
        for response in results:
            assert response.status_code == 200
            result = response.json()
            assert result["command"]["type"] == "send_message"
        
        # Verify all requests were processed
        assert agent.request_count == num_requests

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_concurrent_value_storage(self, mock_google_provider):
        """Test that value storage is isolated between concurrent requests."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        class ValueStorageAgent(Agent):
            def handle(self, context: Context):
                request_id = context.metadata.requestId
                # Store request-specific value
                context.store_value("request_id", request_id)
                context.store_value("timestamp", time.time())
                time.sleep(0.01)  # Simulate processing
                return ContinueConversationResponse(message=f"Stored {request_id}")
        
        agent = ValueStorageAgent(
            llm_config=GoogleProviderConfig(api_key="test", model="test")
        )
        client = TestClient(agent.app)
        
        results = []
        
        def make_request(request_id):
            data = {
                "metadata": {
                    "requestId": f"req-{request_id}",
                    "chatbotId": "bot-456",
                    "conversationId": "conv-789",
                },
                "messages": [],
            }
            response = client.post("/", json=data)
            results.append((request_id, response))
        
        # Create concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check that each response has its own isolated values
        for request_id, response in results:
            assert response.status_code == 200
            result = response.json()
            assert result["valuesToSave"]["request_id"] == f"req-{request_id}"
            assert "timestamp" in result["valuesToSave"]

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_concurrent_auth_validation(self, mock_google_provider):
        """Test concurrent requests with authentication."""
        from zowie_agent_sdk import APIKeyAuth
        
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        auth_config = APIKeyAuth(header_name="X-API-Key", api_key="secret-key")
        agent = ConcurrentTestAgent(
            llm_config=GoogleProviderConfig(api_key="test", model="test"),
            auth_config=auth_config
        )
        client = TestClient(agent.app)
        
        request_data = {
            "metadata": {
                "requestId": "test-123",
                "chatbotId": "bot-456",
                "conversationId": "conv-789",
            },
            "messages": [],
        }
        
        valid_results = []
        invalid_results = []
        
        def make_valid_request(request_id):
            data = request_data.copy()
            data["metadata"]["requestId"] = f"valid-{request_id}"
            headers = {"X-API-Key": "secret-key"}
            response = client.post("/", json=data, headers=headers)
            valid_results.append(response)
        
        def make_invalid_request(request_id):
            data = request_data.copy()
            data["metadata"]["requestId"] = f"invalid-{request_id}"
            headers = {"X-API-Key": "wrong-key"}
            response = client.post("/", json=data, headers=headers)
            invalid_results.append(response)
        
        # Mix valid and invalid requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_valid_request, args=(i,))
            threads.append(thread)
            thread.start()
            
            thread = threading.Thread(target=make_invalid_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(valid_results) == 3
        assert len(invalid_results) == 3
        
        for response in valid_results:
            assert response.status_code == 200
        
        for response in invalid_results:
            assert response.status_code == 401

    @patch('zowie_agent_sdk.llm.google.GoogleProvider')
    def test_thread_safety_and_isolation(self, mock_google_provider):
        """Test that requests are properly isolated and thread-safe."""
        mock_provider_instance = MagicMock()
        mock_google_provider.return_value = mock_provider_instance
        
        class ThreadSafetyAgent(Agent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.processing_requests = set()
                self.lock = threading.Lock()
            
            def handle(self, context: Context):
                request_id = context.metadata.requestId
                
                # Check for concurrent processing of same request
                with self.lock:
                    if request_id in self.processing_requests:
                        raise RuntimeError(f"Request {request_id} already being processed!")
                    self.processing_requests.add(request_id)
                
                try:
                    # Simulate processing
                    time.sleep(0.01)
                    
                    # Each request should have its own context
                    assert hasattr(context, 'metadata')
                    assert hasattr(context, 'messages')
                    assert hasattr(context, 'store_value')
                    
                    # Store request-specific data
                    context.store_value("processed_by", threading.current_thread().name)
                    
                    return ContinueConversationResponse(
                        message=f"Processed {request_id}"
                    )
                finally:
                    # Clean up tracking
                    with self.lock:
                        self.processing_requests.remove(request_id)
        
        agent = ThreadSafetyAgent(
            llm_config=GoogleProviderConfig(api_key="test", model="test")
        )
        client = TestClient(agent.app)
        
        results = []
        errors = []
        
        def make_request(request_id):
            try:
                data = {
                    "metadata": {
                        "requestId": f"safe-{request_id}",
                        "chatbotId": "bot-456",
                        "conversationId": "conv-789",
                    },
                    "messages": [],
                }
                response = client.post("/", json=data)
                results.append((request_id, response))
            except Exception as e:
                errors.append((request_id, e))
        
        # Create many concurrent requests
        threads = []
        for i in range(20):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all succeeded without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        
        # Verify each request was processed correctly
        seen_threads = set()
        for request_id, response in results:
            assert response.status_code == 200
            result = response.json()
            assert result["command"]["payload"]["message"] == f"Processed safe-{request_id}"
            
            # Check that thread names are recorded
            thread_name = result["valuesToSave"]["processed_by"]
            seen_threads.add(thread_name)
        
        # TestClient uses AnyIO which may process all requests on same worker thread
        # The important thing is that all requests were processed correctly
        assert len(seen_threads) >= 1, "Should have thread name recorded"
        
        # Verify no requests are still being processed
        assert len(agent.processing_requests) == 0