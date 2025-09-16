# Zowie Agent SDK for Python

A Python framework for building external agents that integrate with Zowie's Decision Engine. Build agents that can process conversations, make HTTP API calls, use large language models, and transfer conversations between process blocks.

## Architecture

The SDK is built on **FastAPI**, providing an HTTP server with automatic request validation and structured response handling. Agents receive HTTP POST requests from Zowie's Decision Engine, process them using configurable LLM providers and external APIs, and return responses to control conversation flow.

```
┌─────────────────┐    HTTP POST     ┌──────────────────┐
│                 │        /         │                  │
│ Decision Engine │ ──────────────►  │   Your Agent     │
│                 │                  │   (FastAPI)      │
│                 │ ◄────────────────│                  │
└─────────────────┘    JSON Response └──────────────────┘
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
               ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
               │                 │   │                 │   │                 │
               │  LLM Providers  │   │  HTTP Client    │   │  Authentication │
               │  (OpenAI,       │   │  (External      │   │  (API Key,      │
               │   Google)       │   │   APIs)         │   │   Basic, etc.)  │
               │                 │   │                 │   │                 │
               └─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Core Components

- **Agent Base Class**: Abstract base class defining the agent interface
- **Context Management**: Request context with conversation history, metadata, and pre-configured clients
- **LLM Integration**: Multi-provider support for Google Gemini and OpenAI GPT models
- **HTTP Client**: Automatic request/response logging with configurable timeouts
- **Authentication**: Multiple authentication methods with secure credential handling
- **Event Tracking**: Comprehensive logging of API calls and LLM interactions for observability

## Installation

### Using pip with virtual environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install the SDK
pip install zowie-agent-sdk
```

### Using Poetry

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
- **Internal System Integration**: Demonstrates connecting to internal APIs that cannot be exposed publicly
- **Natural Responses**: Returns conversational answers to end users

**Key Features Demonstrated:**

- `generate_structured_content()` for intent analysis
- `TransferToBlockResponse` for seamless handoffs
- `context.http` for internal API calls with automatic logging
- Expert system pattern for specialized business logic

**Example interactions:**

- "What documents do I need?" → Detailed requirements
- "Why was my passport rejected?" → Specific guidance
- "Reset my password" → Transfer to general support

Run the example: `uvicorn example:app --reload`

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

        return ContinueConversationResponse(message=response.text)

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
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

### Running in Production

For production deployment, use multiple workers and disable reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level warning
```

## Configuration

### LLM Provider Configuration

#### Google Gemini

```python
from zowie_agent_sdk import GoogleProviderConfig

llm_config = GoogleProviderConfig(
    api_key=os.getenv("GOOGLE_API_KEY", ""),
    model="gemini-2.5-flash"
)
```

#### OpenAI GPT

```python
from zowie_agent_sdk import OpenAIProviderConfig

llm_config = OpenAIProviderConfig(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    model="gpt-5-mini"
)
```

### Authentication Configuration

The authentication strategy configured in your agent must match the authentication settings in your Zowie External Agent Block configuration. The SDK will verify the authentication headers sent from Zowie's Decision Engine.

#### API Key Authentication

```python
from zowie_agent_sdk import APIKeyAuth

auth_config = APIKeyAuth(
    header_name="X-API-Key",
    api_key=os.getenv("AGENT_API_KEY", "")
)

agent = CustomerSupportAgent(
    llm_config=llm_config,
    auth_config=auth_config
)
```

#### Basic Authentication

```python
from zowie_agent_sdk import BasicAuth

auth_config = BasicAuth(
    username=os.getenv("AGENT_USERNAME", "admin"),
    password=os.getenv("AGENT_PASSWORD", "")
)
```

#### Bearer Token Authentication

```python
from zowie_agent_sdk import BearerTokenAuth

auth_config = BearerTokenAuth(token=os.getenv("AGENT_BEARER_TOKEN", ""))
```

### Agent Configuration Parameters

```python
agent = MyAgent(
    llm_config=llm_config,
    http_timeout_seconds=30.0,                      # Default HTTP timeout
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
    store_value: Callable[[str, Any], None]  # Function to store values
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
    timestamp: str                      # ISO 8601 timestamp
```

#### Persona Information

```python
class Persona:
    name: Optional[str]              # Chatbot name
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
    include_persona=None,    # Use default setting
    include_context=None     # Use default setting
)

# Access the generated text
generated_text = response.text
```

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
    system_instruction="Analyze the user's intent and extract entities"
)

# Access structured data
print(f"Intent: {structured_response.intent}")
print(f"Confidence: {structured_response.confidence}")
```

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

The HTTP client provides automatic event tracking for all requests:

```python
# GET request
response = context.http.get(
    url="https://api.example.com/orders/123",
    headers={"Authorization": f"Bearer {os.getenv('API_TOKEN', '')}"},
    timeout_seconds=30,
    include_headers=True
)

# POST request
response = context.http.post(
    url="https://api.example.com/orders",
    json={"customer_id": "12345", "product": "widget"},
    headers={"Content-Type": "application/json"}
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

Store key-value pairs that persist beyond the current request:

```python
def handle(self, context: Context) -> AgentResponse:
    # Store values for later use
    context.store_value("user_preference", "email_notifications")
    context.store_value("last_interaction", "2024-01-15T10:30:00Z")
    context.store_value("order_history", [{"id": "123", "status": "shipped"}])

    return ContinueConversationResponse(message="Preferences saved!")
```

## Event Tracking and Observability

The SDK automatically tracks all HTTP requests and LLM calls as events. These events are included in agent responses and can be viewed in Supervisor.

### Event Types

#### API Call Events

All HTTP requests made through `context.http` are automatically logged:

```json
{
  "type": "api_call",
  "payload": {
    "url": "https://api.example.com/orders/123",
    "requestMethod": "GET",
    "requestHeaders": { "Authorization": "Bearer ***" },
    "requestBody": null,
    "responseStatusCode": 200,
    "responseHeaders": { "Content-Type": "application/json" },
    "responseBody": "{\"status\": \"shipped\"}",
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
    "prompt": "{\n  \"messages\": [\n    {\n      \"author\": \"User\",\n      \"content\": \"What's my order status?\",\n      \"timestamp\": \"2024-01-15T10:30:00.000Z\"\n    }\n  ],\n  \"system_instruction\": \"You are a helpful assistant.\"\n}",
    "response": "I'd be happy to help you check your order status! Could you please provide your order number?",
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
    log_level="WARNING"                     # Reduce log verbosity
)
```

Override per request:

```python
# Exclude headers for sensitive requests
response = context.http.get(
    url="https://api.example.com/sensitive-data",
    headers={"Authorization": f"Bearer {os.getenv('SENSITIVE_TOKEN', '')}"},
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
      "content": "Hello, I need help with my order",
      "timestamp": "2024-01-15T10:30:00.000Z"
    }
  ],
  "context": "Optional context string",
  "persona": {
    "name": "Support Bot",
    "businessContext": "Customer support for e-commerce",
    "toneOfVoice": "Friendly and professional"
  }
}
```

**Response Format:**

```json
{
  "command": {
    "type": "send_message",
    "payload": {
      "message": "Thank you for contacting us! How can I help you today?"
    }
  },
  "valuesToSave": {
    "interaction_type": "support_request",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "events": [
    {
      "type": "llm_call",
      "payload": {
        "model": "gemini-2.5-flash",
        "prompt": "{\n  \"messages\": [...],\n  \"system_instruction\": \"...\"\n}",
        "response": "Thank you for contacting us! How can I help you today?",
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

## Support and Contributing

For issues, questions, or contributions, please refer to the project repository.

## License

This project is licensed under the MIT License.
