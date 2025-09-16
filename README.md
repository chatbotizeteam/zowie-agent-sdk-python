# Zowie Agent SDK for Python

A Python framework for building external agents that integrate with Zowie's Decision Engine. Build agents that can process conversations, connect to internal databases, call private APIs, use large language models, and transfer conversations between process blocks. The SDK handles all communication with the Decision Engine and automatically tracks HTTP requests and LLM calls for observability in Supervisor.

## Architecture

The SDK is built on **FastAPI**, providing an HTTP server with automatic request validation and structured response handling. Agents receive HTTP POST requests from Zowie's Decision Engine, process them using configurable LLM providers, database connections, private APIs, and external services, then return responses to control conversation flow. All LLM calls and HTTP requests are automatically tracked and made available in Supervisor for observability.

### System Architecture Diagram

```
┌─────────────────────────────┐          ┌──────────────────────────────────────┐
│    Decision Engine          │          │        Zowie Agent SDK               │
│                             │          │            (Your Agent)              │
├─────────────────────────────┤          ├──────────────────────────────────────┤
│                             │ Request  │                                      │
│                             │ ──────►  │  FastAPI Application                 │
│                             │          │  ┌─────────────────────────────────┐ │
│                             │          │  │          Agent Class            │ │
│                             │          │  │                                 │ │
│                             │          │  │  handle(context) -> Response    │ │
│                             │ Response │  └─────────────────────────────────┘ │
│                             │ ◄──────  │                                      │
│                             │          │  ┌─────────────────────────────────┐ │
│                             │          │  │         Context Object          │ │
│                             │          │  │                                 │ │
│                             │          │  │  • metadata: Request Metadata   │ │
│                             │          │  │  • messages: Conversation       │ │
│                             │          │  │  • persona: AI Agent Persona    │ │
│                             │          │  │  • context: Additional context  │ │
│                             │          │  │  • llm: LLM Client              │ │
│                             │          │  │  • http: HTTP Client            │ │
│                             │          │  │  • store_value: State Storage   │ │
│                             │          │  │  • events: Supervisor Events    │ │
│                             │          │  └─────────────────────────────────┘ │
└─────────────────────────────┘          └──────────────────────────────────────┘
                                                           │
                                                           │
                ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                │                                          │                                          │
                ▼                                          ▼                                          ▼
    ┌─────────────────────┐                   ┌─────────────────────┐                   ┌──────────────────────┐
    │   LLM Client        │                   │   HTTP Client       │                   │   Authentication     │
    │  • Google Gemini    │                   │  • Internal Systems │                   │  • API Key Auth      │
    │  • OpenAI           │                   │  • External APIs    │                   │  • Bearer Token      │
    │  • Event Logging    │                   │  • Event Logging    │                   │  • Basic Auth        │
    │                     │                   │                     │                   │                      │
    └─────────────────────┘                   └─────────────────────┘                   └──────────────────────┘
```

### Core Components

- **Agent Base Class**: Abstract base class defining the agent interface - implement your business logic here
- **Context Management**: Request context with conversation history, metadata, and pre-configured clients
- **LLM Integration**: Multi-provider support for Google Gemini and OpenAI GPT models with automatic event tracking
- **HTTP Client**: Automatic request/response logging for private APIs and external services
- **Authentication**: Multiple authentication methods for securing your agent endpoints
- **Event Tracking**: All LLM calls and HTTP requests automatically logged and available in Supervisor
- **Internal System Access**: Connect to databases, private APIs, legacy systems - HTTP requests are automatically tracked for observability

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

## Quick Start

### Example Use Case: Document Verification Expert Agent

The repository includes a complete example (`example.py`) demonstrating a **Document Verification Expert Agent** that showcases advanced SDK features:

- **Specialized Expertise**: Agent only handles document verification questions
- **Scope Detection**: Uses structured analysis to determine if queries are within its domain
- **Transfer Capability**: Automatically transfers out-of-scope questions to general support
- **Internal System Integration**: Demonstrates connecting to internal APIs and private systems that cannot be exposed publicly
- **Natural Responses**: Returns conversational answers to end users

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

### Basic Agent Implementation

```python
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

### Running in Development

Use uvicorn with reload for development:

```bash
poetry uvicorn example:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### Running in Production

For production deployment, disable reload and use multiple workers (if necessary):

```bash
poetry uvicorn example:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
```

## Performance and Concurrency

### Synchronous Design

The SDK uses **synchronous handlers** for simplicity - no need to manage `async`/`await` patterns in your agent logic. This makes it easier to integrate with existing libraries, debug issues, and write readable code.

```python
def handle(self, context: Context) -> AgentResponse:
    # Simple sync code - no async complexity
    data = context.http.get(url, headers)
    response = context.llm.generate_content(messages=context.messages)
    return ContinueConversationResponse(message=response)
```

### Concurrency Limits

**Default**: 40 concurrent requests per worker (FastAPI/Starlette uses AnyIO thread limiter)

For most agent workloads (conversation-based, human-paced), this default is sufficient since agents typically process one conversation at a time.

### Scaling Options

**Horizontal scaling** (recommended): Use multiple Uvicorn workers

```bash
# Single worker (development)
poetry run uvicorn example:app --reload

# Production with 4 workers = 160 concurrent requests
poetry run uvicorn example:app --workers 4 --host 0.0.0.0 --port 8000

# High-traffic deployment
poetry run uvicorn example:app --workers 8 --host 0.0.0.0 --port 8000
```

**When to scale**: Consider multiple workers if you expect sustained high request volumes or have many simultaneous conversations.

## Configuration

### LLM Provider Configuration

Configure the LLM provider using `LLMConfig`, which accepts either Google or OpenAI configurations:

#### Google Gemini

```python
from zowie_agent_sdk import GoogleProviderConfig, LLMConfig

llm_config: LLMConfig = GoogleProviderConfig(
    api_key=os.getenv("GOOGLE_API_KEY", ""),
    model="gemini-2.5-flash"  # or "gemini-1.5-pro", "gemini-1.5-flash"
)
```

**GoogleProviderConfig Parameters:**

- **api_key** (`str`): Google AI API key from Google AI Studio
- **model** (`str`): Model name (e.g., "gemini-2.5-flash", "gemini-1.5-pro")

#### OpenAI GPT

```python
from zowie_agent_sdk import OpenAIProviderConfig, LLMConfig

llm_config: LLMConfig = OpenAIProviderConfig(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    model="gpt-4o-mini"  # or "gpt-4", "gpt-4o", "gpt-3.5-turbo"
)
```

**OpenAIProviderConfig Parameters:**

- **api_key** (`str`): OpenAI API key from OpenAI platform
- **model** (`str`): Model name (e.g., "gpt-4o-mini", "gpt-4", "gpt-4o")

### Authentication Configuration

Configure authentication using `AuthConfig`, which accepts API key, basic auth, or bearer token configurations. The authentication strategy must match your Zowie External Agent Block configuration.

#### API Key Authentication

```python
from zowie_agent_sdk import APIKeyAuth, AuthConfig

auth_config: AuthConfig = APIKeyAuth(
    header_name="X-API-Key",
    api_key=os.getenv("AGENT_API_KEY", "")
)

agent = CustomerSupportAgent(
    llm_config=llm_config,
    auth_config=auth_config
)
```

**APIKeyAuth Parameters:**

- **header_name** (`str`): Name of the HTTP header containing the API key
- **api_key** (`str`): The expected API key value

#### Basic Authentication

```python
from zowie_agent_sdk import BasicAuth, AuthConfig

auth_config: AuthConfig = BasicAuth(
    username=os.getenv("AGENT_USERNAME", "admin"),
    password=os.getenv("AGENT_PASSWORD", "")
)
```

**BasicAuth Parameters:**

- **username** (`str`): Expected username for HTTP Basic Auth
- **password** (`str`): Expected password for HTTP Basic Auth

#### Bearer Token Authentication

```python
from zowie_agent_sdk import BearerTokenAuth, AuthConfig

auth_config: AuthConfig = BearerTokenAuth(
    token=os.getenv("AGENT_BEARER_TOKEN", "")
)
```

**BearerTokenAuth Parameters:**

- **token** (`str`): Expected bearer token value

#### No Authentication

```python
# Authentication is optional - use None for no authentication
agent = CustomerSupportAgent(
    llm_config=llm_config,
    auth_config=None  # No authentication required
)
```

### Agent Configuration Parameters

```python
agent = MyAgent(
    llm_config=llm_config,
    http_timeout_seconds=10.0,                      # Default HTTP timeout
    auth_config=auth_config,                        # Authentication (optional)
    include_persona_by_default=True,                # Include persona in LLM calls
    include_context_by_default=True,                # Include context in LLM calls
    include_http_headers_by_default=True,           # Include headers in event logs
    log_level="INFO"                                # Logging level
)
```

## API Reference

### Agent Class

The `Agent` class is the base class for all agents. Inherit from this class and implement the `handle` method:

```python
from abc import ABC, abstractmethod
from zowie_agent_sdk import Agent, Context, AgentResponse

class MyAgent(Agent):
    @abstractmethod
    def handle(self, context: Context) -> AgentResponse:
        """Process the incoming request and return a response.

        Args:
            context: Request context containing messages, metadata, and clients

        Returns:
            Either ContinueConversationResponse or TransferToBlockResponse
        """
        pass
```

#### Constructor Parameters

- **llm_config** (`LLMConfig`): Configuration for the LLM provider (required)
- **http_timeout_seconds** (`Optional[float]`): Default timeout for HTTP requests in seconds (default: 10.0)
- **auth_config** (`Optional[AuthConfig]`): Authentication configuration (default: None)
- **include_persona_by_default** (`bool`): Include persona in LLM prompts by default (default: True)
- **include_context_by_default** (`bool`): Include context data in LLM prompts by default (default: True)
- **include_http_headers_by_default** (`bool`): Include HTTP headers in event logs (default: True)
- **log_level** (`str`): Python logging level (default: "INFO")

### Context Class

The `Context` object provides access to all request data and pre-configured clients:

```python
class Context:
    metadata: Metadata              # Request metadata (IDs, timestamps)
    messages: List[Message]         # Conversation message history
    context: Optional[str]          # Context string from Zowie configuration
    persona: Optional[Persona]      # Chatbot persona information
    llm: LLM                       # LLM client with automatic context injection
    http: HTTPClient               # HTTP client with automatic event tracking
    store_value: Callable[[str, Any], None]  # Function to store values in Decision Engine
```

#### Metadata Fields

```python
class Metadata:
    requestId: str          # Unique request identifier
    chatbotId: str         # Chatbot identifier
    conversationId: str    # Conversation/session identifier
    interactionId: Optional[str]  # Optional interaction identifier
```

#### Message Structure

```python
class Message:
    author: Literal["User", "Chatbot"]  # Message author
    content: str                        # Message text content
    timestamp: datetime                 # Serialized as ISO 8601 timestamp
```

#### Persona Information

```python
class Persona:
    name: Optional[str]              # AI Agent name
    business_context: Optional[str]  # Business context description
    tone_of_voice: Optional[str]     # Tone and style guidelines
```

### Response Types

#### ContinueConversationResponse

Continue the conversation in the current process block:

```python
from zowie_agent_sdk import ContinueConversationResponse

return ContinueConversationResponse(
    message="Your response message here"
)
```

#### TransferToBlockResponse

Transfer the conversation to another process block:

```python
from zowie_agent_sdk import TransferToBlockResponse

return TransferToBlockResponse(
    message="Optional message sent before transfer",
    next_block="target-block-reference-key"
)
```

### LLM Client

#### Text Generation

```python
response = context.llm.generate_content(
    messages=context.messages,
    system_instruction="Custom system prompt",
    include_persona=None,    # Override default persona inclusion (None = use agent default)
    include_context=None     # Override default context inclusion (None = use agent default)
)

# Access the generated text
generated_text = response
```

**Parameters:**

- **messages** (`List[Message]`): Conversation messages to process
- **system_instruction** (`Optional[str]`): Custom system prompt (default: None)
- **include_persona** (`Optional[bool]`): Override agent's default persona inclusion setting
- **include_context** (`Optional[bool]`): Override agent's default context inclusion setting

#### Structured Content Generation

Generate structured responses using Pydantic models. Note that only a subset of Pydantic models are supported, and this varies by provider:

- **Google Gemini**: [Supported schemas documentation](https://ai.google.dev/gemini-api/docs/structured-output#schemas-in-python)
- **OpenAI**: [Structured outputs documentation](https://platform.openai.com/docs/guides/structured-outputs)

**Simple Pydantic Model:**

```python
from pydantic import BaseModel
from typing import List

class UserIntent(BaseModel):
    intent: str
    confidence: float
    entities: List[str]
    requires_escalation: bool

structured_response = context.llm.generate_structured_content(
    messages=context.messages,
    schema=UserIntent,
    system_instruction="Analyze the user's intent and extract entities",
    include_persona=None,    # Override default persona inclusion (None = use agent default)
    include_context=None     # Override default context inclusion (None = use agent default)
)

# Access structured data
print(f"Intent: {structured_response.intent}")
print(f"Confidence: {structured_response.confidence}")
```

**Parameters:**

- **messages** (`List[Message]`): Conversation messages to process
- **schema** (`Type[BaseModel]`): Pydantic model class for structured output
- **system_instruction** (`Optional[str]`): Custom system prompt (default: None)
- **include_persona** (`Optional[bool]`): Override agent's default persona inclusion setting
- **include_context** (`Optional[bool]`): Override agent's default context inclusion setting

**Pydantic Model with Field Validation:**

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class OrderAnalysis(BaseModel):
    order_status: Literal["pending", "shipped", "delivered", "cancelled"] = Field(
        description="Current status of the order"
    )
    urgency_level: int = Field(
        ge=1, le=10, description="Urgency level from 1 (low) to 10 (critical)"
    )
    action_needed: bool = Field(
        description="Whether immediate action is required"
    )
    customer_sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Customer sentiment based on message tone"
    )

analysis = context.llm.generate_structured_content(
    messages=context.messages,
    schema=OrderAnalysis,
    system_instruction="Analyze this customer service conversation about an order"
)
```

### HTTP Client

The HTTP client provides automatic event tracking for all HTTP requests to private APIs, legacy systems, and external services. All requests are logged and made available in Supervisor for observability:

```python
# GET request to document verification system
response = context.http.get(
    url="https://internal-docs.company.com/verify/passport/ABC123",
    headers={"Authorization": f"Bearer {os.getenv('DOC_SYSTEM_TOKEN', '')}"},
    timeout_seconds=5,
    include_headers=True
)

# POST request to fraud detection service
response = context.http.post(
    url="https://fraud-detection.internal/analyze",
    json={"user_id": "12345", "transaction_data": {...}},
    headers={"Content-Type": "application/json", "X-API-Key": os.getenv('FRAUD_API_KEY', '')}
)

# Other HTTP methods
response = context.http.put(url, json=data, headers=headers)
response = context.http.patch(url, json=data, headers=headers)
response = context.http.delete(url, headers=headers)

# Access response data
if response.status_code == 200:
    data = response.json()
    print(f"Response: {data}")
```

#### HTTP Client Methods

All HTTP methods support these parameters:

- **url** (`str`): Target URL (required)
- **headers** (`Dict[str, str]`): HTTP headers (required)
- **json** (`Any`): JSON payload for POST/PUT/PATCH requests
- **timeout_seconds** (`Optional[float]`): Request timeout override
- **include_headers** (`Optional[bool]`): Include headers in event logs override

### Value Storage

Store key-value pairs within the conversation context that can be accessed in the Decision Engine:

```python
def handle(self, context: Context) -> AgentResponse:
    # Store values for the current conversation context
    context.store_value("user_preference", "email_notifications")
    context.store_value("last_interaction", "2024-01-15T10:30:00Z")
    context.store_value("order_history", [{"id": "123", "status": "shipped"}])

    return ContinueConversationResponse(message="Preferences saved!")
```

## Event Tracking and Observability

The SDK automatically tracks all HTTP requests (to private APIs, legacy systems, and external services) and LLM calls as events. These events are included in agent responses and can be viewed in Supervisor for complete observability into your agent's behavior.

### Event Types

#### API Call Events

All HTTP requests made through `context.http` (private APIs, legacy systems, external services) are automatically logged and available in Supervisor:

```json
{
  "type": "api_call",
  "payload": {
    "url": "https://internal-docs.company.com/verify/passport/ABC123",
    "requestMethod": "GET",
    "requestHeaders": { "Authorization": "Bearer ***" },
    "requestBody": null,
    "responseStatusCode": 200,
    "responseHeaders": { "Content-Type": "application/json" },
    "responseBody": "{\"status\": \"verified\", \"issues\": []}",
    "durationInMillis": 245
  }
}
```

#### LLM Call Events

All LLM interactions are automatically tracked:

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

### Event Configuration

Control event detail level through agent configuration:

```python
agent = MyAgent(
    llm_config=llm_config,
    include_http_headers_by_default=False,  # Exclude headers for security
)
```

Override per request:

```python
# Exclude headers for sensitive requests
response = context.http.get(
    url="https://compliance-api.internal/sensitive-check",
    headers={"Authorization": f"Bearer {os.getenv('COMPLIANCE_TOKEN', '')}"},
    include_headers=False  # Don't log headers for this request
)
```

## API Endpoints

Your agent exposes the following HTTP endpoints:

### POST /

The main endpoint for processing requests from Zowie's Decision Engine.

**Request Format:**

```json
{
  "metadata": {
    "requestId": "unique-request-id",
    "chatbotId": "chatbot-identifier",
    "conversationId": "conversation-identifier",
    "interactionId": "optional-interaction-id"
  },
  "messages": [
    {
      "author": "User",
      "content": "What documents do I need to submit?",
      "timestamp": "2024-01-15T10:30:00.000Z"
    }
  ],
  "context": "Optional context string",
  "persona": {
    "name": "Document Expert",
    "businessContext": "Document verification and compliance",
    "toneOfVoice": "Professional and helpful"
  }
}
```

**Response Format:**

```json
{
  "command": {
    "type": "send_message",
    "payload": {
      "message": "For verification, you'll need a government-issued ID and proof of residence. Would you like specific details about acceptable document types?"
    }
  },
  "valuesToSave": {
    "interaction_type": "document_inquiry",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "events": [
    {
      "type": "llm_call",
      "payload": {
        "model": "gemini-2.5-flash",
        "prompt": "{\n  \"messages\": [...],\n  \"system_instruction\": \"...\"\n}",
        "response": "For verification, you'll need a government-issued ID and proof of residence. Would you like specific details about acceptable document types?",
        "durationInMillis": 1200
      }
    }
  ]
}
```

### GET /health

Health check endpoint for monitoring

**Response:**

```json
{
  "status": "healthy",
  "agent": "MyAgent",
  "timestamp": 1705320600000
}
```

## Request Validation

The SDK automatically validates all incoming requests using Pydantic models. Invalid requests will return HTTP 422 with detailed validation errors.

### Request Validation Model

```python
from zowie_agent_sdk import IncomingRequest

# The SDK validates requests against this model
class IncomingRequest(BaseModel):
    metadata: Metadata
    messages: List[Message]
    context: Optional[str] = None
    persona: Optional[Persona] = None
```

The validation is future-compatible - unknown fields are ignored, so new API versions won't break existing agents.

## Testing

The SDK includes comprehensive test coverage with both mocked unit tests and optional real API integration tests.

### Test Structure

- **Unit/Integration Tests (27 tests)** - Fast tests using mocks, always run in CI/CD

  - `test_simple_integration.py` - Basic agent functionality and workflows
  - `test_llm_integration.py` - LLM provider integration with mocking
  - `test_http_integration.py` - HTTP client integration with mocking

- **Real E2E Tests (5 tests)** - Optional tests that call actual external APIs
  - `test_e2e_real_apis.py` - Real API integration tests (requires API keys)

### Running Tests

#### Run All Mocked Tests (Recommended for CI/CD)

```bash
# Runs all tests with mocking - fast and reliable
poetry run pytest tests/ -k "not real" -v
```

#### Run Specific Test Categories

```bash
# Basic agent workflow tests
poetry run pytest tests/test_simple_integration.py -v

# LLM integration tests with mocking
poetry run pytest tests/test_llm_integration.py -v

# HTTP integration tests with mocking
poetry run pytest tests/test_http_integration.py -v
```

#### Run Real E2E Tests (Requires API Keys)

```bash
# Tests will be skipped if no API keys are provided
poetry run pytest tests/test_e2e_real_apis.py -v

# Run with real API keys
GOOGLE_API_KEY=your_key poetry run pytest tests/test_e2e_real_apis.py::test_real_google_gemini_integration -v
OPENAI_API_KEY=your_key poetry run pytest tests/test_e2e_real_apis.py::test_real_openai_gpt_integration -v

# Run provider comparison (requires both keys)
GOOGLE_API_KEY=your_key OPENAI_API_KEY=your_key poetry run pytest tests/test_e2e_real_apis.py::test_real_provider_comparison -v
```

#### Run Tests with Coverage

```bash
poetry run pytest tests/ -k "not real" --cov=src/zowie_agent_sdk --cov-report=html
```

### Writing Tests

#### Mocking LLM Providers

```python
from unittest.mock import patch

@patch('zowie_agent_sdk.llm.google.GoogleProvider.generate_content')
def test_llm_functionality(mock_generate):
    mock_generate.return_value = "Test response"
    # Test your agent logic here
```

#### Mocking HTTP Requests

```python
@patch('requests.request')
def test_http_functionality(mock_request):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "test"}
    mock_request.return_value = mock_response
    # Test HTTP client logic here
```

#### Using Test Utilities

The SDK provides useful utilities in `tests/utils.py`:

- `create_test_metadata()` - Create customizable Metadata objects
- `create_test_message()` - Create customizable Message objects
- `create_mock_http_response()` - Create mock HTTP responses
- `assert_valid_agent_response()` - Validate response structure
- `assert_events_recorded()` - Check events in responses

Example usage:

```python
from tests.utils import create_test_metadata, create_test_message, create_mock_http_response

def test_my_agent():
    # Create test data with custom values
    metadata = create_test_metadata(request_id="custom-123")
    message = create_test_message(content="Hello agent!", author="User")

    # Create mock HTTP response
    mock_response = create_mock_http_response(
        status_code=200,
        json_data={"result": "success"}
    )
```

The SDK also provides test agent fixtures in `tests/conftest.py`:

- `test_agent` - Simple echo test agent instance
- `llm_test_agent` - Test agent with LLM functionality
- `test_client` - FastAPI TestClient for the test agent

### Test Coverage

The test suite covers:

- Agent lifecycle and request processing
- LLM content generation (text and structured)
- HTTP client operations with event tracking
- Authentication mechanisms
- Error handling and validation
- Multi-turn conversations
- Event tracking and observability

## Support and Contributing

For issues, questions, or contributions, please refer to the project repository.
