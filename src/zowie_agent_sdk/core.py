from __future__ import annotations
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Callable, List, Dict, Optional, Annotated, Union, Literal, Any
from google import genai
import time
import requests
import json as libJson


def get_time_ms() -> int:
    return time.time_ns() // 1_000_000


llm_provider_config: Optional[LLMConfig] = None


class AgentResponseContinue(BaseModel):
    type: Literal["continue"] = "continue"
    messages: List[str]


class AgentResponseFinish(BaseModel):
    type: Literal["finish"] = "finish"
    messages: Optional[List[str]] = None
    command: str


AgentResponse = Annotated[
    Union[AgentResponseContinue, AgentResponseFinish], Field(discriminator="type")
]


class OpenAIConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    apiKey: str


class GoogleConfig(BaseModel):
    provider: Literal["google"] = "google"
    apiKey: str


LLMConfig = Annotated[
    Union[OpenAIConfig, GoogleConfig], Field(discriminator="provider")
]


class HTTPFacade:
    events: List[Event]

    def __init__(self, events: List[Event]):
        self.events = events

    def get(self, url: str, headers: Dict[str, str]) -> requests.Response:
        start = get_time_ms()
        reponse = requests.get(url=url, headers=headers)
        stop = get_time_ms()

        self.events.append(
            APICallEvent(
                payload=APICallEventPayload(
                    url=url,
                    requestMethod="GET",
                    requestHeaders=headers,
                    requestBody=None,
                    responseHeaders=reponse.headers,
                    responseStatusCode=reponse.status_code,
                    responseBody=reponse.text,
                    durationInMillis=stop - start,
                )
            )
        )
        return reponse

    def post(self, url: str, json: Any, headers: Dict[str, str]) -> requests.Response:
        start = get_time_ms()
        reponse = requests.post(url=url, json=json, headers=headers)
        stop = get_time_ms()

        self.events.append(
            APICallEvent(
                payload=APICallEventPayload(
                    url=url,
                    requestMethod="POST",
                    requestHeaders=headers,
                    requestBody=libJson.dumps(json),
                    responseHeaders=reponse.headers,
                    responseStatusCode=reponse.status_code,
                    responseBody=reponse.text,
                    durationInMillis=stop - start,
                )
            )
        )
        return reponse

    def put(self, url: str, json: Any, headers: Dict[str, str]) -> requests.Response:
        start = get_time_ms()
        reponse = requests.put(url=url, json=json, headers=headers)
        stop = get_time_ms()

        self.events.append(
            APICallEvent(
                payload=APICallEventPayload(
                    url=url,
                    requestMethod="PUT",
                    requestHeaders=headers,
                    requestBody=libJson.dumps(json),
                    responseHeaders=reponse.headers,
                    responseStatusCode=reponse.status_code,
                    responseBody=reponse.text,
                    durationInMillis=stop - start,
                )
            )
        )
        return reponse

    def delete(self, url: str, headers: Dict[str, str]) -> requests.Response:
        start = get_time_ms()
        reponse = requests.delete(url=url, headers=headers)
        stop = get_time_ms()

        self.events.append(
            APICallEvent(
                payload=APICallEventPayload(
                    url=url,
                    requestMethod="DELETE",
                    requestHeaders=headers,
                    requestBody=None,
                    responseHeaders=reponse.headers,
                    responseStatusCode=reponse.status_code,
                    responseBody=reponse.text,
                    durationInMillis=stop - start,
                )
            )
        )
        return reponse


class LLMCallEventPayload(BaseModel):
    prompt: str
    response: str
    model: str
    durationInMillis: int


class APICallEventPayload(BaseModel):
    url: str
    requestHeaders: Dict[str, str]
    requestMethod: str
    requestBody: Optional[str]
    responseHeaders: Dict[str, str]
    responseStatusCode: int
    responseBody: Optional[str]
    durationInMillis: int


class LLMCallEvent(BaseModel):
    type: Literal["llm_call"] = "llm_call"
    payload: LLMCallEventPayload


class APICallEvent(BaseModel):
    type: Literal["api_call"] = "api_call"
    payload: APICallEventPayload


Event = Annotated[Union[LLMCallEvent, APICallEvent], Field(discriminator="type")]


class Content(BaseModel):
    text: str
    role: Literal["model", "user"]


class GoogleLLMFacade:
    config: Optional[GoogleConfig]
    events: List[Event]
    client: Optional[genai.Client]

    def __init__(self, config: Optional[GoogleConfig], events: List[Event]):
        self.events = events
        self.config = config
        self.client = None

    def generate_content(
        self,
        model: str,
        contents: List[Content],
        system_instruction: Optional[str] = None,
    ) -> genai.types.GenerateContentResponse:
        if self.config is None:
            raise Exception("LLM requires config.")

        if self.client is None:
            self.client = genai.Client(api_key=self.config.apiKey)

        prepared_content = []
        for content in contents:
            prepared_content.append(
                {"role": content.role, "parts": [{"text": content.text}]}
            )

        prepared_config = genai.types.GenerateContentConfig()

        if system_instruction is not None:
            prepared_config.system_instruction = system_instruction

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=model, contents=prepared_content, config=prepared_config
        )
        stop = get_time_ms()

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=model,
                    prompt=libJson.dumps(prepared_content),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        return response

    def generate_content_with_structured_response(
        self,
        model: str,
        contents: List[Content],
        response_json_schema: Any,
        system_instruction: Optional[str] = None,
    ) -> genai.types.GenerateContentResponse:
        if self.config is None:
            raise Exception("LLM requires config.")

        if self.client is None:
            self.client = genai.Client(api_key=self.config.apiKey)

        prepared_content = []
        for content in contents:
            prepared_content.append(
                {"role": content.role, "parts": [{"text": content.text}]}
            )

        prepared_config = genai.types.GenerateContentConfig(
            response_json_schema=response_json_schema,
            response_mime_type="application/json",
        )

        if system_instruction is not None:
            prepared_config.system_instruction = system_instruction    

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=model, contents=prepared_content, config=prepared_config
        )
        stop = get_time_ms()

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=model,
                    prompt=libJson.dumps(prepared_content),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        return response


class LLMFacade:
    google: GoogleLLMFacade

    def __init__(self, config: Optional[LLMConfig], events: List[Event]):
        self.google = GoogleLLMFacade(config=None, events=events)

        match config:
            case GoogleConfig() as googleConfig:
                self.google = GoogleLLMFacade(config=googleConfig, events=events)


class Message(BaseModel):
    author: str
    content: str
    timestamp: str


class Context:
    messages: List[Message]
    storeValue: Callable[[str, str], None]
    llm: LLMFacade
    http: HTTPFacade

    def __init__(
        self,
        messages: List[Message],
        storeValue: Callable[[str, str], None],
        llm: LLMFacade,
        http: HTTPFacade,
    ) -> None:
        self.messages = messages
        self.storeValue = storeValue
        self.llm = llm
        self.http = http


class ProtocolAgentSendMessagePayload(BaseModel):
    message: str
    messages: List[str]


class ProtocolAgentGoToNextBlockPayload(BaseModel):
    message: Optional[str] = None
    messages: Optional[List[str]] = None
    nextBlockReferenceKey: str


class ProtocolAgentSendMessageCommand(BaseModel):
    type: Literal["send_message"] = "send_message"
    payload: ProtocolAgentSendMessagePayload


class ProtocolAgentGoToNextBlockCommand(BaseModel):
    type: Literal["go_to_next_block"] = "go_to_next_block"
    payload: ProtocolAgentGoToNextBlockPayload


ProtocolAgentCommand = Annotated[
    Union[
        ProtocolAgentSendMessageCommand,
        ProtocolAgentGoToNextBlockCommand,
    ],
    Field(discriminator="type"),
]


class ProtocolAgentResponse(BaseModel):
    command: ProtocolAgentCommand
    valuesToSave: Optional[Dict[str, str]] = None
    events: List[Event]


def configure_llm(config: LLMConfig) -> None:
    global llm_provider_config
    llm_provider_config = config


def start_agent(handler: Callable[[Context], AgentResponse]) -> FastAPI:
    app = FastAPI()

    @app.post("/")
    async def handle(request: Request) -> ProtocolAgentResponse:
        valueStorage: Dict[str, str] = {}
        events: List[Event] = []

        def storeValue(key: str, value: str) -> None:
            valueStorage[key] = value

        llm_facade = LLMFacade(config=llm_provider_config, events=events)
        http_facade = HTTPFacade(events=events)

        input_json = await request.json()

        context = Context(
            messages=input_json["messages"],
            storeValue=storeValue,
            llm=llm_facade,
            http=http_facade,
        )
        result = handler(context)

        match result:
            case AgentResponseContinue(messages=messages):
                response = ProtocolAgentResponse(
                    command=ProtocolAgentSendMessageCommand(
                        payload=ProtocolAgentSendMessagePayload(
                            message="\n".join(messages), messages=messages
                        )
                    ),
                    valuesToSave=valueStorage,
                    events=events,
                )

            case AgentResponseFinish(messages=messages, command=command):
                payload = ProtocolAgentGoToNextBlockPayload(
                    nextBlockReferenceKey=command,
                )
                if messages != None:
                    payload.messages = messages
                    payload.message = "\n".join(messages)

                response = ProtocolAgentResponse(
                    command=ProtocolAgentGoToNextBlockCommand(payload=payload),
                    valuesToSave=valueStorage,
                    events=events,
                )

        print(response)
        return response

    return app
