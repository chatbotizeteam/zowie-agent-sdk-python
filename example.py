import os

from src.zowie_agent_sdk import (
    Agent,
    AgentResponse,
    AgentResponseContinue,
    AgentResponseFinish,
    Content,
    Context,
)
from zowie_agent_sdk.types import GoogleConfig


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            llm_config=GoogleConfig(
                api_key=os.environ.get("GOOGLE_API_KEY") or "", model="gemini-2.0-flash"
            )
        )

    def handle(self, context: Context) -> AgentResponse:
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

        # Return response
        return AgentResponseContinue(message=llm_response.text)


# Create agent
agent = MyAgent()
app = agent.app
