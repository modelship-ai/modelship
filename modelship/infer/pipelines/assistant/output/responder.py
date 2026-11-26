"""`Responder`: starts a session's responses. From a `response.create`'s parameters and the session's settings it
builds the response, the LLM call that answers it, and, for a spoken response, the voice that says it."""

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from modelship.infer.pipelines.assistant.calls import Answer, PipelineCalls
from modelship.infer.pipelines.assistant.conversation import Conversation, as_input
from modelship.infer.pipelines.assistant.items import audio_in_item, missing_item
from modelship.infer.pipelines.assistant.output.playback import Playback
from modelship.infer.pipelines.assistant.output.response import ActiveResponse, Send
from modelship.infer.pipelines.assistant.output.speech import Spoken
from modelship.infer.pipelines.assistant.output.tts_voice import TtsVoice
from modelship.infer.pipelines.assistant.session_config import no_tts
from modelship.logging import get_logger
from modelship.openai.protocol.realtime import (
    Item,
    ItemReference,
    RealtimeError,
    RealtimeResponse,
    RealtimeSession,
    ResponseAudio,
    ResponseAudioOutput,
    ResponseParams,
)
from modelship.openai.protocol.responses.schemas import ResponsesRequest
from modelship.utils import random_uuid
from modelship.utils.config_schema import AssistantPipelineConfig

logger = get_logger("realtime")


@dataclass
class _LlmCall:
    """The LLM call a response answers with: once *pending*, the transcription of the turns committed before the
    response, is done, it sends *model* the request *fields* with *items* as its input, so the turns' transcripts
    are in it."""

    calls: PipelineCalls
    model: str
    session_id: str
    items: list[Item]
    fields: dict[str, Any]
    pending: asyncio.Task | None

    async def start(self) -> Answer | RealtimeError:
        if self.pending is not None:
            await asyncio.wait([self.pending])
        request = ResponsesRequest(model=self.model, input=as_input(self.items), **self.fields)
        answer = self.calls.respond(self.model, request, self.session_id)
        if answer is None:
            return RealtimeError(message=f"Pipeline model {self.model!r} isn't ready.", type="server_error")
        return answer


class Responder:
    """Starts the responses of the session whose settings are *state*, on the pipeline *config*, with *calls*.
    *conversation* is the session's, *send* sends its events, *playback* counts its spoken audio and *spoken* keeps
    what each spoken message said."""

    def __init__(
        self,
        config: AssistantPipelineConfig,
        calls: PipelineCalls,
        state: RealtimeSession,
        conversation: Conversation,
        playback: Playback,
        spoken: dict[str, Spoken],
        send: Send,
    ):
        self._config = config
        self._calls = calls
        self._state = state
        self._conversation = conversation
        self._playback = playback
        self._spoken = spoken
        self._send = send

    def start(self, params: ResponseParams, pending: asyncio.Task | None) -> ActiveResponse | RealtimeError:
        """The response *params* ask for, answering once *pending* is done; or why it can't start."""
        state = self._state
        modalities = params.output_modalities or state.output_modalities or ["text"]
        if "audio" in modalities and (error := no_tts(self._config, "response.output_modalities")) is not None:
            return error
        if ignored := sorted(params.model_extra or {}):
            logger.debug("Realtime session %s: ignoring response fields not used yet: %s", state.id, ignored)
        items = self._conversation.items if params.input is None else self._input(params.input)
        if isinstance(items, RealtimeError):
            return items
        llm = self._config.assistant_config.llm
        assert llm is not None
        max_output_tokens = params.max_output_tokens or state.max_output_tokens or "inf"
        call = _LlmCall(self._calls, llm, state.id, items, self._request_fields(params, max_output_tokens), pending)
        in_conversation = params.conversation != "none"
        response = RealtimeResponse(
            id=f"resp_{random_uuid()[:20]}",
            conversation_id=self._conversation.id if in_conversation else None,
            output_modalities=modalities,
            max_output_tokens=max_output_tokens,
            metadata=params.metadata,
        )
        voice = self._voice(params, response) if "audio" in modalities else None
        conversation = self._conversation if in_conversation else None
        return ActiveResponse(response, llm, call.start, conversation, self._send, voice)

    def _request_fields(self, params: ResponseParams, max_output_tokens: int | str) -> dict[str, Any]:
        """The LLM request's fields but its model and input: the response's own settings, else the session's."""
        state = self._state
        reasoning = params.reasoning or state.reasoning
        tools = params.tools if params.tools is not None else state.tools
        tool_choice = params.tool_choice if params.tool_choice is not None else state.tool_choice
        parallel = params.parallel_tool_calls
        return {
            "instructions": params.instructions if params.instructions is not None else state.instructions,
            "max_output_tokens": None if max_output_tokens == "inf" else max_output_tokens,
            "reasoning": {"effort": reasoning.effort} if reasoning is not None and reasoning.effort else None,
            "tools": [tool.model_dump(exclude_none=True) for tool in tools] if tools else None,
            "tool_choice": tool_choice.model_dump() if isinstance(tool_choice, BaseModel) else tool_choice,
            "parallel_tool_calls": parallel if parallel is not None else state.parallel_tool_calls,
            "stream": True,
            "store": False,
        }

    def _voice(self, params: ResponseParams, response: RealtimeResponse) -> TtsVoice:
        """The voice a spoken *response* says its answer in: the response's own format and voice, else the
        session's; it sets them on *response*."""
        tts = self._config.assistant_config.tts
        output = self._state.audio.output
        assert tts is not None and output is not None
        given = params.audio.output if params.audio is not None else None
        audio_format = given.format if given is not None and given.format is not None else output.format
        voice = given.voice if given is not None and given.voice is not None else output.voice
        response.audio = ResponseAudio(output=ResponseAudioOutput(format=audio_format, voice=voice))
        return TtsVoice(
            self._calls, tts, self._state.id, audio_format, voice, output.speed, self._playback, self._spoken
        )

    def _input(self, given: list[Item | ItemReference]) -> list[Item] | RealtimeError:
        """A response's own input items, each reference replaced by the conversation item it names."""
        items = []
        for i, item in enumerate(given):
            if isinstance(item, ItemReference):
                found = self._conversation.get(item.id)
                if found is None:
                    return missing_item(item.id, f"response.input[{i}].id")
                item = found
            elif (error := audio_in_item(item, f"response.input[{i}]")) is not None:
                return error
            items.append(item)
        return items
