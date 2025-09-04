from zowie_agent_sdk import (
    start_agent,
    configure_llm,
    Context,
    AgentResponse,
    AgentResponseContinue,
    AgentResponseFinish,
    GoogleConfig,
    Content,
)


configure_llm(GoogleConfig(apiKey=""))


def handler(context: Context) -> AgentResponse:
    print(context.messages)

    texts = [m["content"] for m in context.messages]

    if "stop" in "".join(texts):
        return AgentResponseFinish(command="completed")

    context.storeValue("k1", "v1")
    context.storeValue("k2", "v2")

    api_response1 = context.http.post(
        url="https://customizations.chatbotize.com/ecommerce/users",
        json={"email": "maciej@ciolek.me"},
        headers={"X-Api-Key": "e35f4459059b45deb05890d297750828"},
    )
    print(api_response1)

    llm_response1 = context.llm.google.generate_content(
        model="gemini-2.0-flash-001",
        contents=[Content(role="user", text="How are you?")],
    )

    print(llm_response1)

    api_response2 = context.http.get(
        url="https://customizations.chatbotize.com/ecommerce/users",
        headers={"X-Api-Key": "e35f4459059b45deb05890d297750828"},
    )
    print(api_response2)

    llm_response2 = context.llm.google.generate_content_with_structured_response(
        model="gemini-2.0-flash-001",
        contents=[Content(role="user", text="How are you?")],
        response_json_schema={
            "type": "object",
            "properties": {
                "followupQuestion": {
                    "type": "string",
                    "description": "a follow up question to ask user to achieve defined goal",
                },
                "goalAchieved": {
                    "type": "boolean",
                    "description": "a follow up question to ask user to achieve defined goal",
                },
            },
            "required": ["followupQuestion", "goalAchieved"],
        },
    )

    print(llm_response2)

    return AgentResponseContinue(
        messages=[llm_response2.candidates[0].content.parts[0].text]
    )


agent = start_agent(handler=handler)
