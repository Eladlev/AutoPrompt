import json
import logging
import time
from typing import List, Optional, Any, Union, Iterator, AsyncIterator

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream, agenerate_from_stream
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel

GAI_API_HOST = 'http://sn-gai-api.ai.smartnews.net/v3'
GAI_API_HEADER = {'Content-Type': 'application/json', 'accept': 'application/json', 'x-api-key': ''}
GAI_API_PROJECT = 'gai_moderation'

TEXT_COMPLETION_URL = f"{GAI_API_HOST}/text/completion"
TEXT_COMPLETION_STREAMING_URL = f"{GAI_API_HOST}/text/completion_streaming"
logger = logging.getLogger(__name__)


class GAIChat(ChatOpenAI):

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        timeout = httpx.Timeout(300.0)
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=50)
        self.client = httpx.Client(timeout=timeout, limits=limits)
        self.async_client = httpx.AsyncClient(timeout=timeout, limits=limits)

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        data = self._populate_gai_payload(messages, stop, kwargs=kwargs)
        with self.client.stream(
                "POST", TEXT_COMPLETION_STREAMING_URL, headers=GAI_API_HEADER, json=data,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                chunk = json.loads(line)
                message = AIMessageChunk(content=chunk["completion"])
                chunk = ChatGenerationChunk(message=message)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        data = self._populate_gai_payload(messages, stop, kwargs=kwargs)
        start = time.time()
        response = self.client.post(
            TEXT_COMPLETION_URL, headers=GAI_API_HEADER, json=data,
        )
        response.raise_for_status()
        result = self._create_chat_result(response.json())
        cost = time.time() - start
        logger.info(f"generate in {cost:.3f}s, output: {result.llm_output}")
        return result

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        data = self._populate_gai_payload(messages, stop, kwargs=kwargs)
        async with self.async_client.stream(
                "POST", TEXT_COMPLETION_STREAMING_URL, headers=GAI_API_HEADER, json=data,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = json.loads(line)
                message = AIMessageChunk(content=chunk["completion"])
                chunk = ChatGenerationChunk(message=message)
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        data = self._populate_gai_payload(messages, stop, kwargs=kwargs)
        response = None
        for attempt in range(3):
            start = time.time()
            response = await self.async_client.post(
                TEXT_COMPLETION_URL, headers=GAI_API_HEADER, json=data,
            )
            if response.is_success:
                result = self._create_chat_result(response.json())
                cost = time.time() - start
                logger.info(f"generate in {cost:.3f}s, output: {result.llm_output}")
                return result
            else:
                logger.warning(f"Error generate: {response}")
        response.raise_for_status()

    def _populate_gai_payload(
            self, messages: List[BaseMessage], stop: Optional[List[str]], kwargs: Optional[Any] = None
    ) -> dict:
        message_dicts, params = self._create_message_dicts(messages, stop)
        data = {
            "model": self.model_name,
            "project": GAI_API_PROJECT,
            "temperature": self.temperature,
        }
        for message in message_dicts:
            if message.get("role") == "user":
                data["prompt"] = message.get("content")
            elif (message.get("role") == "assistant"
                  or message.get("role") == "system"):
                data["system_hint"] = message.get("content")
        if self.max_tokens:
            data["max_tokens"] = self.max_tokens
        if "stop" in params:
            data["stop"] = params.get("stop")[0]
        if "response_format" in params:
            data["json_response"] = True
        if 'functions' in kwargs:
            tools = []
            for f in kwargs["functions"]:
                tools.append({"type": "function", "function": f})
            data["tools"] = tools
            data["tool_choice"] = "required"
        return data

    def _create_chat_result(
            self, response: Union[dict, BaseModel]
    ) -> ChatResult:
        additional_kwargs = {}
        if 'tool_calls' in response:
            additional_kwargs["function_call"] = response.get("tool_calls")[0].get("function")
        content = response.get("completion") or ""
        message = AIMessage(content=content, additional_kwargs=additional_kwargs)
        generations = [ChatGeneration(message=message)]
        token_usage = get_token_counts_from_gai_api_response(response)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response.get("model"),
            "system_fingerprint": "",
        }
        return ChatResult(generations=generations, llm_output=llm_output)


def get_token_counts_from_gai_api_response(resp: dict) -> dict:
    tokens = {
        "prompt_tokens": resp.get("prompt_tokens", 0),
        "completion_tokens": resp.get("completion_tokens", 0)
    }
    tokens["total_tokens"] = (tokens.get("prompt_tokens")
                              + tokens.get("completion_tokens"))
    return tokens