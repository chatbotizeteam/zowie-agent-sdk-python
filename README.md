# Zowie Agent SDK for Python

A Python framework for building external agents that integrate with **Zowie’s Decision Engine**.
With this SDK, you can build agents that:

- Process conversations and generate natural responses
- Connect to internal systems and private APIs
- Use LLMs (Google Gemini, OpenAI GPT) for reasoning
- Transfer conversations between workflow blocks
- Get **full observability** in Zowie Supervisor (LLM calls + API calls auto-tracked)

The SDK handles all communication with the Decision Engine so you can focus on your business logic.

## Table of Contents

- [Architecture](https://www.google.com/search?q=%23architecture)
- [Prerequisites](https://www.google.com/search?q=%23prerequisites)
- [Installation](https://www.google.com/search?q=%23installation)
- [Quick Start](https://www.google.com/search?q=%23quick-start)
- [Configuration](https://www.google.com/search?q=%23configuration)
  - [LLM Provider Configuration](https://www.google.com/search?q=%23llm-provider-configuration)
  - [Authentication Configuration](https://www.google.com/search?q=%23authentication-configuration)
- [Usage Guide and API Reference](https://www.google.com/search?q=%23usage-guide-and-api-reference)
  - [Agent Class](https://www.google.com/search?q=%23agent-class)
  - [Context Class](https://www.google.com/search?q=%23context-class)
  - [Response Types](https://www.google.com/search?q=%23response-types)
  - [LLM Client](https://www.google.com/search?q=%23llm-client)
  - [HTTP Client](https://www.google.com/search?q=%23http-client)
  - [Value Storage](https://www.google.com/search?q=%23value-storage)
- [Performance and Concurrency](https://www.google.com/search?q=%23performance-and-concurrency)
- [Event Tracking and Observability](https://www.google.com/search?q=%23event-tracking-and-observability)
- [API Endpoints](https://www.google.com/search?q=%23api-endpoints)
- [Request Validation](https://www.google.com/search?q=%23request-validation)
- [Testing](https://www.google.com/search?q=%23testing)
- [Development Setup](https://www.google.com/search?q=%23development-setup)
- [Support and Contributing](https://www.google.com/search?q=%23support-and-contributing)

---

## Architecture

The SDK is built on **FastAPI**, providing an HTTP server that integrates with Zowie's Decision Engine. Your agents receive conversation requests, process them using LLMs and external APIs, then return responses to either continue the conversation or transfer control to other workflow blocks.

### System Architecture Diagram

```mermaid
flowchart TD
    subgraph Decision Engine
        DE_Core[Decision Engine Core]
    end

    subgraph Your Agent [Zowie Agent SDK]
        direction LR
        FastAPI[FastAPI Application] --> AgentClass["Agent Class: handle(context)"]
        AgentClass --> Context["Context Object <br/>(metadata, messages, llm, http, etc.)"]
    end

    subgraph External Services
        direction TB
        LLM[LLM Providers <br/>(Google, OpenAI)]
        HTTP[Internal & External APIs]
    end

    DE_Core -- Request --> FastAPI
    FastAPI -- Response --> DE_Core

    Context --> LLM
    Context --> HTTP
```

### Core Components

- **Agent Base Class**: Abstract base class defining the agent interface - implement your business logic here.
- **Context Management**: Request context with conversation history, metadata, and pre-configured clients.
- **LLM Integration**: Multi-provider support for Google Gemini and OpenAI GPT models with automatic event tracking.
- **HTTP Client**: Automatic request/response logging for private APIs and external services.
- **Authentication**: Multiple authentication methods for securing your agent endpoints.
- **Event Tracking**: All LLM calls and HTTP requests automatically logged and available in Supervisor.
- **Internal System Access**: Connect to databases, private APIs, legacy systems - HTTP requests are automatically tracked for observability.

---

## Prerequisites

- Python 3.9+
- An active Zowie AI Agent
- API keys for your chosen LLM provider (Google Gemini/OpenAI)

---

## Installation

### Using pip with virtual environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install the SDK
pip install zowie-agent-sdk
```

### Using Poetry (recommended)

```bash
poetry add zowie-agent-sdk
```

### Using uv

```bash
uv add zowie-agent-sdk
```

### Dependencies

The SDK requires Python 3.9+ and includes the following core dependencies:

- **FastAPI**: Web framework for building the HTTP API
- **Pydantic v2**: Data validation and serialization
- **Google AI SDK**: For Google Gemini model integration
- **OpenAI SDK**: For OpenAI GPT model integration
- **Requests**: HTTP client library

---

## Quick Start

### 1\. Basic Agent Implementation

Create a simple agent that responds to user messages using an LLM.

```python
# example.py
import os
from zowie_agent_sdk import (
    Agent,
    Context,
    ContinueConversationResponse,
    GoogleProviderConfig,
    AgentResponse
)

class CustomerSupportAgent(Agent):
    def handle(self, context: Context) -> AgentResponse:
        """Process incoming requests and generate responses."""
        response = context.llm.generate_content(
            messages=context.messages,
            system_instruction="You are a helpful customer support assistant. "
                             "Provide accurate and friendly assistance."
        )

        return ContinueConversationResponse(message=response)

# Configure the agent
agent = CustomerSupportAgent(
    llm_config=GoogleProviderConfig(
        api_key=os.getenv("GOOGLE_API_KEY", ""),
        model="gemini-2.5-flash"
    )
)

# The FastAPI application is accessible via agent.app
app = agent.app
```

### 2\. Running Your Agent

#### Development

Use `uvicorn` with auto-reload for a smooth development experience.

```bash
poetry run uvicorn example:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

#### Production

For production deployment, disable reload and use multiple workers to handle concurrent conversations.

```bash
poetry run uvicorn example:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

### 3\. Advanced Example: Document Verification Expert

The repository includes a complete example (`example.py`) demonstrating a **Document Verification Expert Agent** that showcases advanced SDK features:

- **Specialized Expertise**: Agent only handles document verification questions.
- **Scope Detection**: Uses structured analysis to determine if queries are within its domain.
- **Transfer Capability**: Automatically transfers out-of-scope questions to general support.
- **Internal System Integration**: Demonstrates connecting to internal APIs and private systems that cannot be exposed publicly.
- **Natural Responses**: Returns conversational answers to end users.

**Key Features Demonstrated:**

- `generate_structured_content()` for intent analysis
- `TransferToBlockResponse` for seamless handoffs
- `context.http` for internal API calls with automatic logging and Supervisor visibility
- Expert system pattern for specialized business logic

**Example interactions:**

- "What documents do I need?" → Detailed requirements
- "Why was my passport rejected?" → Specific guidance
- "Reset my password" → Transfer to general support

Run the example: `poetry run uvicorn example:app --reload`

---

## Configuration

### LLM Provider Configuration

Configure the LLM provider using one of the following configuration objects.

#### Google Gemini

```python
from zowie_agent_sdk import GoogleProviderConfig

llm_config = GoogleProviderConfig(
    api_key=os.getenv("GOOGLE_API_KEY", ""),
    model="gemini-2.5-flash"  # or "gemini-2.5-pro"
)
```

- **api_key** (`str`): Your Google AI API key.
- **model** (`str`): The model name to use.

#### OpenAI GPT

```python
from zowie_agent_sdk import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    model="gpt-5-mini"  # or "gpt-5"
)
```

- **api_key** (`str`): Your OpenAI API key.
- **model** (`str`): The model name to use.

### Authentication Configuration

Secure your agent's endpoint using an authentication strategy that matches your Zowie External Agent Block configuration.

#### API Key Authentication

```python
from zowie_agent_sdk import APIKeyAuth

auth_config = APIKeyAuth(
    header_name="X-API-Key",
    api_key=os.getenv("AGENT_API_KEY", "")
)
```

- **header_name** (`str`): Name of the HTTP header containing the API key.
- **api_key** (`str`): The expected API key value.

#### Basic Authentication

```python
from zowie_agent_sdk import BasicAuth

auth_config = BasicAuth(
    username=os.getenv("AGENT_USERNAME", "admin"),
    password=os.getenv("AGENT_PASSWORD", "")
)
```

- **username** (`str`): The expected username for HTTP Basic Auth.
- **password** (`str`): The expected password for HTTP Basic Auth.

#### Bearer Token Authentication

```python
from zowie_agent_sdk import BearerTokenAuth

auth_config = BearerTokenAuth(
    token=os.getenv("AGENT_BEARER_TOKEN", "")
)
```

- **token** (`str`): The expected bearer token value.

### Agent Configuration Parameters

You can customize the agent's behavior during initialization.

```python
agent = MyAgent(
    llm_config=llm_config,
    http_timeout_seconds=10.0,                  # Default HTTP timeout
    auth_config=auth_config,                    # Authentication (optional)
    include_persona_by_default=True,            # Include persona in LLM calls
    include_context_by_default=True,            # Include context in LLM calls
    include_http_headers_by_default=True,       # Include headers in event logs
    log_level="INFO"                            # Logging level
)
```

---

## Usage Guide and API Reference

### Agent Class

The `Agent` class is the base for all agents. Inherit from this class and implement the `handle` method.

```python
from zowie_agent_sdk import Agent, Context, AgentResponse

class MyAgent(Agent):
    def handle(self, context: Context) -> AgentResponse:
        """
        Process the incoming request and return a response.

        Args:
            context: Request context containing messages, metadata, and clients.

        Returns:
            Either ContinueConversationResponse or TransferToBlockResponse.
        """
        # Your agent logic goes here
        pass
```

### Context Class

The `Context` object provides access to all request data and pre-configured clients.

- `metadata: Metadata`: Request metadata (IDs, timestamps).
- `messages: List[Message]`: Conversation message history.
- `context: Optional[str]`: Context string from the Zowie configuration.
- `persona: Optional[Persona]`: Chatbot persona information.
- `llm: LLM`: LLM client with automatic context injection and event tracking.
- `http: HTTPClient`: HTTP client with automatic event tracking.
- `store_value: Callable[[str, Any], None]`: Function to store values in the Decision Engine.

### Response Types

Your `handle` method must return one of two response types.

#### ContinueConversationResponse

Continue the conversation in the current process block.

```python
from zowie_agent_sdk import ContinueConversationResponse

return ContinueConversationResponse(
    message="Your response message here"
)
```

#### TransferToBlockResponse

Transfer the conversation to another process block.

```python
from zowie_agent_sdk import TransferToBlockResponse

return TransferToBlockResponse(
    message="Optional message sent before transfer.",
    next_block="target-block-reference-key"
)
```

### LLM Client

The `context.llm` client provides methods for interacting with language models.

#### Text Generation

```python
response_text = context.llm.generate_content(
    messages=context.messages,
    system_instruction="Custom system prompt",
    include_persona=True,  # Override agent default
    include_context=True   # Override agent default
)
```

#### Structured Content Generation

Generate structured JSON output that conforms to a Pydantic model. This is ideal for tasks like intent detection, entity extraction, or data classification.

```python
from pydantic import BaseModel, Field
from typing import Literal

class OrderAnalysis(BaseModel):
    urgency: int = Field(ge=1, le=5, description="Urgency level from 1 to 5")
    sentiment: Literal["positive", "neutral", "negative"]

analysis = context.llm.generate_structured_content(
    messages=context.messages,
    schema=OrderAnalysis,
    system_instruction="Analyze this customer service conversation about an order."
)

print(f"Urgency: {analysis.urgency}, Sentiment: {analysis.sentiment}")
```

**Note**: Supported Pydantic features may vary by LLM provider. Refer to the [Google Gemini](https://ai.google.dev/gemini-api/docs/structured-output#schemas-in-python) and [OpenAI](https://platform.openai.com/docs/guides/structured-outputs) documentation for details.

### HTTP Client

The `context.http` client provides standard HTTP methods (`get`, `post`, `put`, etc.) with automatic event tracking for observability in Supervisor.

```python
# GET request to a document verification system
response = context.http.get(
    url="https://internal-docs.company.com/verify/passport/ABC123",
    headers={"Authorization": f"Bearer {os.getenv('DOC_SYSTEM_TOKEN', '')}"},
    timeout_seconds=5,
    include_headers=False # Don't log headers for this sensitive request
)

# POST request to a fraud detection service
fraud_check = context.http.post(
    url="https://fraud-detection.internal/analyze",
    json={"user_id": "12345", "transaction_id": "txn_abc"},
    headers={"X-API-Key": os.getenv('FRAUD_API_KEY', '')}
)

if fraud_check.status_code == 200:
    data = fraud_check.json()
    print(f"Fraud check result: {data}")
```

### Value Storage

Store key-value pairs within the conversation that can be used later in the Decision Engine.

```python
def handle(self, context: Context) -> AgentResponse:
    # Store a value for use in the current conversation
    context.store_value("user_preference", "email_notifications")

    return ContinueConversationResponse(message="Your preferences have been saved!")
```

---

## Performance and Concurrency

### Synchronous by Design

The SDK uses **synchronous handlers** for simplicity. You don't need to manage `async`/`await` patterns in your agent logic, which makes it easier to integrate with existing synchronous libraries and simplifies debugging.

```python
def handle(self, context: Context) -> AgentResponse:
    # Simple, readable synchronous code
    data = context.http.get(url, headers).json()
    response = context.llm.generate_content(messages=context.messages)
    return ContinueConversationResponse(message=response)
```

### Scaling with Workers

The underlying FastAPI server runs synchronously in a thread pool. To handle high traffic, you should scale horizontally by increasing the number of `uvicorn` workers.

- **Default Concurrency**: 40 concurrent requests (conversations) per worker.
- **Scaling**: A 4-worker setup can handle approximately 160 concurrent conversations.

<!-- end list -->

```bash
# Production with 4 workers = ~160 concurrent requests
poetry run uvicorn example:app --workers 4 --host 0.0.0.0 --port 8000
```

Consider adding more workers if your agent performs long-running I/O operations or if you expect sustained high request volumes.

---

## Event Tracking and Observability

The SDK automatically tracks all `context.http` and `context.llm` calls as events. These are sent back with every agent response and are visible in **Supervisor**, giving you complete observability into your agent's interactions with external systems.

### API Call Event Example

```json
{
  "type": "api_call",
  "payload": {
    "url": "https://fraud-detection.internal/analyze",
    "requestMethod": "POST",
    "requestHeaders": { "X-API-Key": "***" },
    "requestBody": "{\"user_id\": \"12345\"}",
    "responseStatusCode": 200,
    "responseBody": "{\"risk_score\": 0.15}",
    "durationInMillis": 245
  }
}
```

### LLM Call Event Example

```json
{
  "type": "llm_call",
  "payload": {
    "model": "gemini-2.5-flash",
    "prompt": "{\n  \"messages\": [\n    {\n      \"author\": \"User\",\n      \"content\": \"What documents do I need for verification?\",\n      \"timestamp\": \"2024-01-15T10:30:00.000Z\"\n    }\n  ],\n  \"system_instruction\": \"You are a document verification expert.\"\n}",
    "response": "For account verification, you'll need to provide a government-issued ID and proof of residence.",
    "durationInMillis": 1200
  }
}
```

---

## API Endpoints

Your agent server exposes the following HTTP endpoints.

### `POST /`

The main endpoint for processing conversation requests from the Zowie Decision Engine.

- **Request Body**: A JSON object containing `metadata`, `messages`, `context`, and optional `persona`.
- **Response Body**: A JSON object containing the `command` to execute, `valuesToSave`, and a list of `events`.

### `GET /health`

A simple health check endpoint for monitoring.

- **Response Body**: `{"status": "healthy", "agent": "YourAgentClassName"}`

---

## Request Validation

All incoming requests to the `POST /` endpoint are automatically validated against a Pydantic model. Invalid requests will receive an HTTP 422 Unprocessable Entity response with detailed validation errors. The validation is forward-compatible, meaning new fields added by Zowie in the future will be ignored and won't break your agent.

---

## Testing

The SDK is designed to be easily testable. The repository includes a comprehensive test suite with mocks for external services.

### Running Tests

```bash
# Run all fast, mocked tests (recommended for CI/CD)
poetry run pytest tests/ -k "not real" -v

# Run real E2E tests against live APIs (requires API keys)
GOOGLE_API_KEY="..." OPENAI_API_KEY="..." poetry run pytest tests/test_e2e_real_apis.py -v

# Run tests with coverage report
poetry run pytest tests/ -k "not real" --cov=src/zowie_agent_sdk --cov-report=html
```

### Writing Tests

Use standard mocking libraries to test your agent's logic without making real network calls. The SDK provides test utilities in `tests/utils.py` to help create test data and assert outcomes.

```python
from unittest.mock import patch, Mock
from tests.utils import create_test_metadata, create_test_message

# Mocking an HTTP request
@patch('requests.request')
def test_agent_with_http_call(mock_request):
    mock_response = Mock(status_code=200)
    mock_response.json.return_value = {"data": "test"}
    mock_request.return_value = mock_response

    # Your agent testing logic here...

# Mocking an LLM call
@patch('zowie_agent_sdk.llm.google.GoogleProvider.generate_content')
def test_agent_with_llm_call(mock_generate):
    mock_generate.return_value = "Test response from LLM"

    # Your agent testing logic here...
```

---

## Development Setup

For contributors working on the SDK itself:

```bash
# Clone the repository
git clone <repo-url>
cd zowie-agent-sdk-python

# Install dependencies and set up pre-commit hooks
make setup

# Run all quality checks (lint, format, typecheck, test)
make check
```

---

## Support and Contributing

For issues, questions, or contributions, please refer to the project repository.
