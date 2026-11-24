"""One realtime response: the pipeline's LLM answers through the gateway's Responses path, and the client gets its
answer as Realtime events, in text or spoken by the pipeline's tts model with the text as its transcript. Reasoning
reaches neither the client nor the conversation."""

import asyncio
import base64
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from modelship.logging import get_logger
from modelship.metrics import RequestTracker
from modelship.openai.protocol import ErrorResponse
from modelship.openai.protocol.realtime import (
    AssistantMessage,
    FunctionCall,
    OutputAudio,
    OutputText,
    RealtimeError,
    RealtimeResponse,
    ResponseStatus,
    StatusDetails,
    error_event,
    response_event,
    server_event,
)
from modelship.openai.realtime.calls import Answer
from modelship.openai.realtime.conversation import Conversation
from modelship.openai.realtime.speech import SendAudio, Speech

logger = get_logger("realtime")

Send = Callable[[dict], Awaitable[None]]

_TERMINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})
_PART_EVENTS = frozenset(
    {
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
    }
)


def _usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Responses usage in the Realtime shape: all text."""
    if usage is None:
        return None
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "total_tokens": usage["total_tokens"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "input_token_details": {
            "text_tokens": usage["input_tokens"],
            "audio_tokens": 0,
            "cached_tokens": cached,
            "cached_tokens_details": {"text_tokens": cached, "audio_tokens": 0},
        },
        "output_token_details": {"text_tokens": usage["output_tokens"], "audio_tokens": 0},
    }


Start = Callable[[], Awaitable[Answer | RealtimeError]]
# a new spoken answer for the message item named, sending its audio through the given function
NewSpeech = Callable[[str, SendAudio], Speech]
CancelReason = Literal["turn_detected", "client_cancelled"]


class _SpeechError(Exception):
    def __init__(self, error: RealtimeError):
        self.error = error


@dataclass
class _AudioPart:
    """The spoken part of a message, while it is spoken."""

    part: OutputAudio
    where: dict[str, Any]
    speech: Speech
    transcript_done: bool = False


class ActiveResponse:
    """A response in progress: the answer *start* gets from *model*, or the error it can't, sent through *send*. Its
    output joins *conversation*, unless that is None. With *speech*, each message is spoken as it is written."""

    def __init__(
        self,
        response: RealtimeResponse,
        model: str,
        start: Start,
        conversation: Conversation | None,
        send: Send,
        speech: NewSpeech | None = None,
    ):
        self.response = response
        self._model = model
        self._start = start
        self._conversation = conversation
        self._send = send
        self._new_speech = speech
        self._audio: _AudioPart | None = None
        # its message items by id; reasoning and tool calls have none
        self._items: dict[str, AssistantMessage | FunctionCall] = {}
        self._terminal: dict[str, Any] | None = None
        self._started = False
        self._cancelled: CancelReason | None = None
        self._finishing = False
        self.task = asyncio.ensure_future(self._run())

    def done(self) -> bool:
        return self.task.done()

    @property
    def cancelled(self) -> bool:
        return self._cancelled is not None

    @property
    def in_conversation(self) -> bool:
        """Whether its output joins the conversation."""
        return self._conversation is not None

    def has_item(self, item_id: str) -> bool:
        return item_id in self._items

    def cancel(self, reason: CancelReason = "client_cancelled") -> None:
        """Ends the response early, with a `response.done` of status cancelled for *reason*."""
        if self._cancelled or self._finishing:
            return
        self._cancelled = reason
        # a task cancelled before its first step never runs `_run`, so it would send nothing
        if self._started:
            self.task.cancel()

    async def abandon(self) -> None:
        """Stops the response and sends nothing more: the client has left."""
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.task

    async def _run(self) -> None:
        self._started = True
        tracker = RequestTracker(self._model, "realtime_response")
        try:
            await self._send(response_event("response.created", self.response))
            if self._cancelled:
                raise asyncio.CancelledError
            failure = await self._generate()
        except asyncio.CancelledError:
            tracker.finish("cancelled")
            if not self._cancelled:
                if self._audio is not None:
                    await self._audio.speech.stop()
                raise
            self._finishing = True
            await self._end("cancelled", StatusDetails(type="cancelled", reason=self._cancelled))
            return
        except _SpeechError as failed:
            failure = (failed.error, "tts_error")
        except Exception:
            logger.exception("Realtime response %s failed", self.response.id)
            failure = (_server_error(), "unhandled")
        self._finishing = True
        terminal = self._terminal
        if failure is None and terminal is None:
            failure = (_server_error(), "stream_error")
        if failure is None and terminal is not None and terminal.get("status") == "failed":
            error = terminal.get("error") or {}
            code = error.get("code") or "server_error"
            kind = "server_error" if code == "server_error" else "invalid_request_error"
            failure = (RealtimeError(message=error.get("message") or "", type=kind, code=code), "stream_error")
        if failure is not None:
            error, error_type = failure
            tracker.finish("error", error_type)
            await self._send(error_event(error))
            await self._end("failed", StatusDetails(type="failed", error={"type": error.type, "code": error.code}))
            return
        assert terminal is not None
        self.response.usage = _usage(terminal.get("usage"))
        tracker.finish("ok")
        if terminal.get("status") == "incomplete":
            reason = (terminal.get("incomplete_details") or {}).get("reason")
            known = reason if reason in ("max_output_tokens", "content_filter") else None
            await self._end("incomplete", StatusDetails(type="incomplete", reason=known))
            return
        await self._end("completed", None)

    async def _generate(self) -> tuple[RealtimeError, str] | None:
        """Streams the LLM's answer to the client. The error it couldn't start with or a loader answered with
        instead, or None."""
        started = await self._start()
        if isinstance(started, RealtimeError):
            return started, "model_not_ready"
        failure = None
        async with contextlib.aclosing(started) as answer:
            async for event in answer:
                if isinstance(event, ErrorResponse):
                    error = event.error
                    failure = (
                        RealtimeError(message=error.message, type=error.type, code=error.code),
                        "inference_error",
                    )
                    continue
                await self._on_event(event)
        return failure

    async def _on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind in _TERMINAL:
            self._terminal = event["response"]
            return
        if kind == "response.output_item.added":
            added = event["item"]
            if added.get("type") == "message":
                await self._add_item(
                    AssistantMessage(id=added["id"], type="message", role="assistant", status="in_progress", content=[])
                )
            elif added.get("type") == "function_call":
                call = FunctionCall(
                    id=added["id"],
                    type="function_call",
                    status="in_progress",
                    call_id=added["call_id"],
                    name=added["name"],
                    arguments="",
                )
                await self._add_item(call)
            return
        item_id = event["item"]["id"] if kind == "response.output_item.done" else event.get("item_id")
        item = self._items.get(item_id or "")
        if item is None:
            return
        where = {"response_id": self.response.id, "item_id": item.id, "output_index": self._index(item)}
        if isinstance(item, FunctionCall):
            await self._on_call_event(kind, event, item, where)
            return
        if kind in _PART_EVENTS:
            where["content_index"] = event["content_index"]
        if kind == "response.output_item.done":
            item.status = event["item"]["status"]
            await self._item_done(item)
        elif self._new_speech is not None:
            await self._on_audio_event(kind, event, item, where)
        elif kind == "response.content_part.added":
            item.content.append(OutputText(type="output_text", text=""))
            await self._send(server_event(kind, **where, part={"type": "text", "text": ""}))
        elif kind == "response.output_text.delta":
            part = item.content[-1]
            assert isinstance(part, OutputText)
            part.text += event["delta"]
            await self._send(server_event(kind, **where, delta=event["delta"]))
        elif kind == "response.output_text.done":
            await self._send(server_event(kind, **where, text=event["text"]))
        elif kind == "response.content_part.done":
            await self._send(server_event(kind, **where, part={"type": "text", "text": event["part"]["text"]}))

    async def _on_audio_event(
        self, kind: str | None, event: dict[str, Any], item: AssistantMessage, where: dict
    ) -> None:
        """A message part's events, for a spoken answer: the text goes out as its transcript and to the tts model."""
        assert self._new_speech is not None
        if kind == "response.content_part.added":
            part = OutputAudio(type="output_audio")
            item.content.append(part)
            await self._send(server_event(kind, **where, part={"type": "audio", "transcript": ""}))

            at = dict(where)

            async def send_audio(audio: bytes) -> bool:
                if self._cancelled:
                    # the client stopped listening
                    return False
                delta = base64.b64encode(audio).decode()
                await self._send(server_event("response.output_audio.delta", **at, delta=delta))
                return True

            self._audio = _AudioPart(part, at, self._new_speech(item.id, send_audio))
            return
        audio = self._audio
        if audio is None:
            return
        if kind == "response.output_text.delta":
            audio.part.transcript += event["delta"]
            await self._send(server_event("response.output_audio_transcript.delta", **where, delta=event["delta"]))
            audio.speech.write(event["delta"])
            if audio.speech.failure is not None:
                raise _SpeechError(audio.speech.failure)
        elif kind == "response.output_text.done":
            audio.transcript_done = True
            transcript = event["text"]
            await self._send(server_event("response.output_audio_transcript.done", **where, transcript=transcript))
        elif kind == "response.content_part.done":
            failure = await audio.speech.finish()
            if failure is not None:
                raise _SpeechError(failure)
            await self._close_audio(audio)

    async def _close_audio(self, audio: _AudioPart) -> None:
        """Ends the spoken part *audio*, spoken or not."""
        self._audio = None
        where, transcript = audio.where, audio.part.transcript
        if not audio.transcript_done:
            await self._send(server_event("response.output_audio_transcript.done", **where, transcript=transcript))
        await self._send(server_event("response.output_audio.done", **where))
        part = {"type": "audio", "transcript": transcript}
        await self._send(server_event("response.content_part.done", **where, part=part))

    async def _on_call_event(self, kind: str | None, event: dict[str, Any], call: FunctionCall, where: dict) -> None:
        if kind == "response.function_call_arguments.delta":
            call.arguments += event["delta"]
            await self._send(server_event(kind, **where, call_id=call.call_id, delta=event["delta"]))
        elif kind == "response.function_call_arguments.done":
            call.arguments = event["arguments"]
            await self._send(server_event(kind, **where, call_id=call.call_id, arguments=call.arguments))
        elif kind == "response.output_item.done":
            done = event["item"]
            # a loader may send the function's name after its first chunk
            call.name, call.arguments, call.status = done["name"], done["arguments"], done["status"]
            await self._item_done(call)

    def _index(self, item: AssistantMessage | FunctionCall) -> int:
        return next(i for i, other in enumerate(self.response.output) if other is item)

    async def _add_item(self, item: AssistantMessage | FunctionCall) -> None:
        item_id = item.id
        self._items[item_id] = item
        self.response.output.append(item)
        await self._send(
            server_event(
                "response.output_item.added", response_id=self.response.id, output_index=self._index(item), item=item
            )
        )
        if self._conversation is not None:
            before = self._conversation.insert(item)
            await self._send(server_event("conversation.item.added", previous_item_id=before, item=item))

    async def _item_done(self, item: AssistantMessage | FunctionCall) -> None:
        await self._send(
            server_event(
                "response.output_item.done", response_id=self.response.id, output_index=self._index(item), item=item
            )
        )
        if self._conversation is not None and self._conversation.get(item.id) is not None:
            before = self._conversation.before(item.id)
            await self._send(server_event("conversation.item.done", previous_item_id=before, item=item))

    async def _end(self, status: ResponseStatus, details: StatusDetails | None) -> None:
        if self._audio is not None:
            await self._audio.speech.stop()
            await self._close_audio(self._audio)
        for item in self.response.output:
            if item.status == "in_progress":
                item.status = "incomplete"
                await self._item_done(item)
        self.response.status = status
        self.response.status_details = details
        await self._send(response_event("response.done", self.response))


def _server_error() -> RealtimeError:
    return RealtimeError(message="Internal error during generation.", type="server_error", code="server_error")
