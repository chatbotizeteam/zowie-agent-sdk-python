from src.zowie_agent_sdk import (
    AgentResponse,
    AgentResponseContinue,
    AgentResponseFinish,
    Content,
    Context,
    #    GoogleConfig,
    OpenAIConfig,
    configure_llm,
    start_agent,
)

# Configure with Google Gemini
# configure_llm(GoogleConfig(apiKey="your-google-api-key", model="gemini-2.0-flash-001"))

# OR configure with OpenAI
configure_llm(OpenAIConfig(apiKey="your-openai-api-key", model="gpt-4o-mini"))


def handler(context: Context) -> AgentResponse:
    # Access metadata
    print(f"Request ID: {context.metadata.requestId}")
    print(f"Messages: {context.messages}")

    # Check for stop condition
    texts = [m.content for m in context.messages]
    if "stop" in "".join(texts):
        return AgentResponseFinish(next_block="completed")

    # Store values (supports any JSON-compatible type)
    context.store_value("user_data", {"name": "John", "age": 30})
    context.store_value("priority", 5)

    # Make HTTP requests if needed
    # api_response = context.http.post(
    #     url="https://api.example.com/endpoint",
    #     json={"key": "value"},
    #     headers={"X-Api-Key": "your-api-key"},
    # )

    # Use LLM with unified interface - no model parameter needed!
    llm_response = context.llm.generate_content(
        contents=[Content(role="user", text="How are you?")]
    )
    print(f"LLM response: {llm_response.text}")

    # Structured response with schema
    llm_structured = context.llm.generate_structured_content(
        contents=[Content(role="user", text="How are you? Please respond with JSON.")],
        schema={
            "type": "object",
            "properties": {
                "followupQuestion": {
                    "type": "string",
                    "description": "a follow up question to ask user",
                },
                "goalAchieved": {
                    "type": "boolean",
                    "description": "whether the conversation goal is achieved",
                },
            },
            "required": ["followupQuestion", "goalAchieved"],
        },
    )
    print(f"Structured response: {llm_structured.text}")

    # Return response - now with single message field
    return AgentResponseContinue(message=llm_response.text)


agent = start_agent(handler=handler)
