from __future__ import annotations

import json as libJson
import time
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

import openai
import requests
from fastapi import FastAPI, Request
from google import genai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field


def get_time_ms() -> int:
    return time.time_ns() // 1_000_000


llm_provider_config: Optional[LLMConfig] = None


class AgentResponseContinue(BaseModel):
    type: Literal["continue"] = "continue"
    message: str


class AgentResponseFinish(BaseModel):
    type: Literal["finish"] = "finish"
    message: Optional[str] = None
    next_block: str


AgentResponse = Annotated[
    Union[AgentResponseContinue, AgentResponseFinish], Field(discriminator="type")
]


class OpenAIConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    apiKey: str
    model: str


class GoogleConfig(BaseModel):
    provider: Literal["google"] = "google"
    apiKey: str
    model: str


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
                    responseHeaders=dict(reponse.headers),
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
                    responseHeaders=dict(reponse.headers),
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
                    responseHeaders=dict(reponse.headers),
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
                    responseHeaders=dict(reponse.headers),
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


class Persona(BaseModel):
    name: Optional[str]
    business_context: Optional[str]
    tone_of_voice: Optional[str]


class Content(BaseModel):
    text: str
    role: Literal["model", "user"]


class LLMResponse(BaseModel):
    text: str
    raw_response: Any
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    def __init__(
        self,
        config: Union[GoogleConfig, OpenAIConfig],
        events: List[Event],
        persona: Optional[Persona],
    ):
        self.model = config.model
        self.api_key = config.apiKey
        self.events = events
        self.persona = persona
        self.config = config

    @abstractmethod
    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        pass

    @abstractmethod
    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        pass

    def _build_persona_instruction(self) -> str:
        if self.persona is None:
            return ""

        instruction = "<persona>\n"
        if self.persona.name:
            instruction += f"<name>{self.persona.name}</name>\n\n"
        if self.persona.business_context:
            instruction += (
                f"<business_context>\n{self.persona.business_context}"
                f"\n</business_context>\n\n"
            )
        if self.persona.tone_of_voice:
            instruction += (
                f"<tone_of_voice>\n{self.persona.tone_of_voice}"
                f"\n</tone_of_voice>\n\n"
            )
        instruction += "</persona>"
        return instruction


class GoogleProvider(BaseLLMProvider):
    def __init__(
        self,
        config: GoogleConfig,
        events: List[Event],
        persona: Optional[Persona],
    ):
        super().__init__(config, events, persona)
        self.client: Optional[genai.Client] = None

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)

        prepared_contents = []
        for content in contents:
            prepared_contents.append(
                {"role": content.role, "parts": [{"text": content.text}]}
            )

        prepared_config = genai.types.GenerateContentConfig()

        system_instruction_temp = self._build_persona_instruction()
        if system_instruction:
            system_instruction_temp += system_instruction

        if system_instruction_temp:
            prepared_config.system_instruction = system_instruction_temp

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model, contents=prepared_contents, config=prepared_config
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": system_instruction_temp,
            "contents": prepared_contents,
        }

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(
                        prompt_data, indent=2, sort_keys=True, ensure_ascii=False
                    ),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        # Extract text from Google response
        text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="google",
            model=self.model,
        )

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)

        prepared_contents = []
        for content in contents:
            prepared_contents.append(
                {"role": content.role, "parts": [{"text": content.text}]}
            )

        prepared_config = genai.types.GenerateContentConfig(
            response_json_schema=schema,
            response_mime_type="application/json",
        )

        system_instruction_temp = self._build_persona_instruction()
        if system_instruction:
            system_instruction_temp += system_instruction

        if system_instruction_temp:
            prepared_config.system_instruction = system_instruction_temp

        start = get_time_ms()
        response = self.client.models.generate_content(
            model=self.model, contents=prepared_contents, config=prepared_config
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": system_instruction_temp,
            "response_json_schema": schema,
            "contents": prepared_contents,
        }

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(
                        prompt_data, indent=2, sort_keys=True, ensure_ascii=False
                    ),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        # Extract text from Google response
        text = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="google",
            model=self.model,
        )


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        config: OpenAIConfig,
        events: List[Event],
        persona: Optional[Persona],
    ):
        super().__init__(config, events, persona)
        self.client: Optional[openai.OpenAI] = None

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        if self.client is None:
            self.client = openai.OpenAI(api_key=self.api_key)

        messages: List[ChatCompletionMessageParam] = []

        system_instruction_temp = self._build_persona_instruction()
        if system_instruction:
            system_instruction_temp += system_instruction

        if system_instruction_temp:
            messages.append(ChatCompletionSystemMessageParam(
                role="system", content=system_instruction_temp
            ))

        for content in contents:
            if content.role == "model":
                messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant", content=content.text
                ))
            else:
                messages.append(ChatCompletionUserMessageParam(
                    role="user", content=content.text
                ))

        start = get_time_ms()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": system_instruction_temp,
            "messages": messages,
        }

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(
                        prompt_data, indent=2, sort_keys=True, ensure_ascii=False
                    ),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        # Extract text from OpenAI response
        text = ""
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="openai",
            model=self.model,
        )

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.client is None:
            self.client = openai.OpenAI(api_key=self.api_key)

        messages: List[ChatCompletionMessageParam] = []

        system_instruction_temp = self._build_persona_instruction()
        if system_instruction:
            system_instruction_temp += system_instruction

        if system_instruction_temp:
            messages.append(ChatCompletionSystemMessageParam(
                role="system", content=system_instruction_temp
            ))

        for content in contents:
            if content.role == "model":
                messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant", content=content.text
                ))
            else:
                messages.append(ChatCompletionUserMessageParam(
                    role="user", content=content.text
                ))

        start = get_time_ms()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        stop = get_time_ms()

        prompt_data = {
            "system_instruction": system_instruction_temp,
            "response_json_schema": schema,
            "messages": messages,
        }

        self.events.append(
            LLMCallEvent(
                payload=LLMCallEventPayload(
                    model=self.model,
                    prompt=libJson.dumps(
                        prompt_data, indent=2, sort_keys=True, ensure_ascii=False
                    ),
                    response=response.model_dump_json(),
                    durationInMillis=stop - start,
                )
            )
        )

        # Extract text from OpenAI response
        text = ""
        if response.choices and len(response.choices) > 0:
            text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            raw_response=response,
            provider="openai",
            model=self.model,
        )


class LLM:
    def __init__(
        self,
        config: Optional[LLMConfig],
        events: List[Event],
        persona: Optional[Persona],
    ):
        self.provider: Optional[BaseLLMProvider] = None
        
        if config is None:
            return
            
        match config:
            case GoogleConfig() as googleConfig:
                self.provider = GoogleProvider(
                    config=googleConfig, events=events, persona=persona
                )
            case OpenAIConfig() as openaiConfig:
                self.provider = OpenAIProvider(
                    config=openaiConfig, events=events, persona=persona
                )

    def generate_content(
        self, contents: List[Content], system_instruction: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_content(contents, system_instruction, **kwargs)

    def generate_structured_content(
        self,
        contents: List[Content],
        schema: Any,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.provider is None:
            raise Exception("LLM provider not configured")
        return self.provider.generate_structured_content(
            contents, schema, system_instruction, **kwargs
        )


class Metadata(BaseModel):
    requestId: str
    chatbotId: str
    conversationId: str
    interactionId: Optional[str] = None


class Message(BaseModel):
    author: str
    content: str
    timestamp: str


class Context:
    metadata: Metadata
    messages: List[Message]
    context: Optional[str]
    store_value: Callable[[str, Any], None]
    llm: LLM
    http: HTTPFacade

    def __init__(
        self,
        metadata: Metadata,
        messages: List[Message],
        context: Optional[str],
        store_value: Callable[[str, Any], None],
        llm: LLM,
        http: HTTPFacade,
    ) -> None:
        self.metadata = metadata
        self.messages = messages
        self.context = context
        self.store_value = store_value
        self.llm = llm
        self.http = http


class SendMessagePayload(BaseModel):
    message: str


class GoToNextBlockPayload(BaseModel):
    message: Optional[str] = None
    nextBlockReferenceKey: str


class SendMessageCommand(BaseModel):
    type: Literal["send_message"] = "send_message"
    payload: SendMessagePayload


class GoToNextBlockCommand(BaseModel):
    type: Literal["go_to_next_block"] = "go_to_next_block"
    payload: GoToNextBlockPayload


Command = Annotated[
    Union[
        SendMessageCommand,
        GoToNextBlockCommand,
    ],
    Field(discriminator="type"),
]


class ExternalAgentResponse(BaseModel):
    command: Command
    valuesToSave: Optional[Dict[str, Any]] = None
    events: Optional[List[Event]] = None


def configure_llm(config: LLMConfig) -> None:
    global llm_provider_config
    llm_provider_config = config


def start_agent(handler: Callable[[Context], AgentResponse]) -> FastAPI:
    app = FastAPI()

    @app.post("/")
    async def handle(request: Request) -> ExternalAgentResponse:
        valueStorage: Dict[str, Any] = {}
        events: List[Event] = []

        def storeValue(key: str, value: Any) -> None:
            valueStorage[key] = value

        input_json = await request.json()

        # Parse metadata
        metadata = Metadata(
            requestId=input_json["metadata"]["requestId"],
            chatbotId=input_json["metadata"]["chatbotId"],
            conversationId=input_json["metadata"]["conversationId"],
            interactionId=input_json["metadata"].get("interactionId"),
        )

        # Parse persona
        persona = None
        if "persona" in input_json and input_json["persona"] is not None:
            persona = Persona(name=None, business_context=None, tone_of_voice=None)
            if input_json["persona"].get("name") is not None:
                persona.name = input_json["persona"]["name"]
            if input_json["persona"].get("businessContext") is not None:
                persona.business_context = input_json["persona"]["businessContext"]
            if input_json["persona"].get("toneOfVoice") is not None:
                persona.tone_of_voice = input_json["persona"]["toneOfVoice"]

        # Create LLM and HTTP facades
        llm = LLM(config=llm_provider_config, events=events, persona=persona)
        http_facade = HTTPFacade(events=events)

        # Create context
        context = Context(
            metadata=metadata,
            messages=input_json["messages"],
            context=input_json.get("context"),
            store_value=storeValue,
            llm=llm,
            http=http_facade,
        )
        
        # Call handler
        result = handler(context)

        # Build response
        match result:
            case AgentResponseContinue(message=message):
                response = ExternalAgentResponse(
                    command=SendMessageCommand(
                        payload=SendMessagePayload(message=message)
                    ),
                    valuesToSave=valueStorage if valueStorage else None,
                    events=events if events else None,
                )

            case AgentResponseFinish(message=message, next_block=next_block):
                payload = GoToNextBlockPayload(
                    nextBlockReferenceKey=next_block,
                    message=message,
                )

                response = ExternalAgentResponse(
                    command=GoToNextBlockCommand(payload=payload),
                    valuesToSave=valueStorage if valueStorage else None,
                    events=events if events else None,
                )

        return response

    return app