"""One realtime response: the pipeline's LLM answers through its loader's Responses path, and the client gets its
answer as Realtime events, in text or spoken by the pipeline's tts model with the text as its transcript. Reasoning
reaches neither the client nor the conversation.

`ActiveResponse` turns the loader's Responses events into Realtime events: a table maps each event to its handler,
and each message part goes out through the response's output, `TextOutput` or `AudioOutput`."""

import asyncio
import base64
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from modelship.infer.pipelines.assistant.calls import Answer
from modelship.infer.pipelines.assistant.conversation import Conversation
from modelship.infer.pipelines.assistant.output.speech import Speech
from modelship.infer.pipelines.assistant.output.tts_voice import TtsVoice
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

logger = get_logger("realtime")

Send = Callable[[dict], Awaitable[None]]
Start = Callable[[], Awaitable[Answer | RealtimeError]]
CancelReason = Literal["turn_detected", "client_cancelled"]
# where an event is in the response: its response, item, output index and, for a part's events, content index
Where = dict[str, Any]

_TERMINAL = frozenset({"response.completed", "response.incomplete", "response.failed"})


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


def _server_error() -> RealtimeError:
    return RealtimeError(message="Internal error during generation.", type="server_error", code="server_error")


@dataclass(frozen=True)
class _Failure:
    """Why a response failed: the error the client gets, and the kind the request metrics count it as."""

    error: RealtimeError
    kind: str


class _SpeechError(Exception):
    def __init__(self, error: RealtimeError):
        self.error = error


class Output(Protocol):
    """How a message's content parts go out to the client."""

    async def part_added(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None: ...

    async def text_delta(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None: ...

    async def text_done(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None: ...

    async def part_done(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None: ...

    async def abandon(self) -> None:
        """Stops at once: the client has left."""
        ...

    async def end(self) -> None:
        """Ends the part in progress, as the response ends."""
        ...


class TextOutput:
    """The parts go out as text."""

    def __init__(self, send: Send):
        self._send = send

    async def part_added(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        item.content.append(OutputText(type="output_text", text=""))
        await self._send(server_event(event["type"], **where, part={"type": "text", "text": ""}))

    async def text_delta(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        part = item.content[-1]
        assert isinstance(part, OutputText)
        part.text += event["delta"]
        await self._send(server_event(event["type"], **where, delta=event["delta"]))

    async def text_done(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        await self._send(server_event(event["type"], **where, text=event["text"]))

    async def part_done(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        part = {"type": "text", "text": event["part"]["text"]}
        await self._send(server_event(event["type"], **where, part=part))

    async def abandon(self) -> None:
        pass

    async def end(self) -> None:
        pass


@dataclass
class _AudioPart:
    """The spoken part of a message, while it is spoken."""

    part: OutputAudio
    where: Where
    speech: Speech
    transcript_done: bool = False


class AudioOutput:
    """The parts are spoken by *voice*, their text going out as the audio's transcript. *cancelled* says whether the
    response was cancelled, after which no more audio goes out."""

    def __init__(self, send: Send, voice: TtsVoice, cancelled: Callable[[], bool]):
        self._send = send
        self._voice = voice
        self._cancelled = cancelled
        # the part being spoken, if any
        self._audio: _AudioPart | None = None

    async def part_added(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        part = OutputAudio(type="output_audio")
        item.content.append(part)
        await self._send(server_event(event["type"], **where, part={"type": "audio", "transcript": ""}))
        at = dict(where)

        async def send_audio(audio: bytes) -> bool:
            if self._cancelled():
                # the client stopped listening
                return False
            delta = base64.b64encode(audio).decode()
            await self._send(server_event("response.output_audio.delta", **at, delta=delta))
            return True

        self._audio = _AudioPart(part, at, self._voice.speech(item.id, send_audio, self._cancelled))

    async def text_delta(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        audio = self._audio
        if audio is None:
            return
        audio.part.transcript += event["delta"]
        await self._send(server_event("response.output_audio_transcript.delta", **where, delta=event["delta"]))
        audio.speech.write(event["delta"])
        if audio.speech.failure is not None:
            raise _SpeechError(audio.speech.failure)

    async def text_done(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        audio = self._audio
        if audio is None:
            return
        audio.transcript_done = True
        transcript = event["text"]
        await self._send(server_event("response.output_audio_transcript.done", **where, transcript=transcript))

    async def part_done(self, item: AssistantMessage, where: Where, event: dict[str, Any]) -> None:
        audio = self._audio
        if audio is None:
            return
        failure = await audio.speech.finish()
        if failure is not None:
            raise _SpeechError(failure)
        await self._close(audio)

    async def abandon(self) -> None:
        if self._audio is not None:
            await self._audio.speech.stop()

    async def end(self) -> None:
        if self._audio is not None:
            await self._audio.speech.stop()
            await self._close(self._audio)

    async def _close(self, audio: _AudioPart) -> None:
        """Ends the spoken part *audio*, spoken or not."""
        self._audio = None
        where, transcript = audio.where, audio.part.transcript
        if not audio.transcript_done:
            await self._send(server_event("response.output_audio_transcript.done", **where, transcript=transcript))
        await self._send(server_event("response.output_audio.done", **where))
        part = {"type": "audio", "transcript": transcript}
        await self._send(server_event("response.content_part.done", **where, part=part))


# handles a message's part event: the message, where the event is, and the event
PartHandler = Callable[[AssistantMessage, Where, dict[str, Any]], Awaitable[None]]


class ActiveResponse:
    """A response in progress: the answer *start* gets from *model*, or the error it can't, sent through *send*. Its
    output joins *conversation*, unless that is None. With *voice*, each message is spoken as it is written."""

    def __init__(
        self,
        response: RealtimeResponse,
        model: str,
        start: Start,
        conversation: Conversation | None,
        send: Send,
        voice: TtsVoice | None = None,
    ):
        self.response = response
        self._model = model
        self._start = start
        self._conversation = conversation
        self._send = send
        self._output: Output = TextOutput(send) if voice is None else AudioOutput(send, voice, lambda: self.cancelled)
        self._part_handlers: dict[str, PartHandler] = {
            "response.content_part.added": self._output.part_added,
            "response.output_text.delta": self._output.text_delta,
            "response.output_text.done": self._output.text_done,
            "response.content_part.done": self._output.part_done,
        }
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
                await self._output.abandon()
                raise
            self._finishing = True
            await self._end("cancelled", StatusDetails(type="cancelled", reason=self._cancelled))
            return
        except _SpeechError as failed:
            failure = _Failure(failed.error, "tts_error")
        except Exception:
            logger.exception("Realtime response %s failed", self.response.id)
            failure = _Failure(_server_error(), "unhandled")
        self._finishing = True
        failure = failure or self._terminal_failure()
        if failure is not None:
            tracker.finish("error", failure.kind)
            await self._send(error_event(failure.error))
            error = {"type": failure.error.type, "code": failure.error.code}
            await self._end("failed", StatusDetails(type="failed", error=error))
            return
        terminal = self._terminal
        assert terminal is not None
        self.response.usage = _usage(terminal.get("usage"))
        tracker.finish("ok")
        if terminal.get("status") == "incomplete":
            reason = (terminal.get("incomplete_details") or {}).get("reason")
            known = reason if reason in ("max_output_tokens", "content_filter") else None
            await self._end("incomplete", StatusDetails(type="incomplete", reason=known))
            return
        await self._end("completed", None)

    def _terminal_failure(self) -> _Failure | None:
        """The failure the loader's last event says, or that it sent none."""
        terminal = self._terminal
        if terminal is None:
            return _Failure(_server_error(), "stream_error")
        if terminal.get("status") != "failed":
            return None
        error = terminal.get("error") or {}
        code = error.get("code") or "server_error"
        kind = "server_error" if code == "server_error" else "invalid_request_error"
        return _Failure(RealtimeError(message=error.get("message") or "", type=kind, code=code), "stream_error")

    async def _generate(self) -> _Failure | None:
        """Streams the LLM's answer to the client. The error it couldn't start with or a loader answered with
        instead, or None."""
        started = await self._start()
        if isinstance(started, RealtimeError):
            return _Failure(started, "model_not_ready")
        failure = None
        async with contextlib.aclosing(started) as answer:
            async for event in answer:
                if isinstance(event, ErrorResponse):
                    error = event.error
                    failure = _Failure(
                        RealtimeError(message=error.message, type=error.type, code=error.code), "inference_error"
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
            await self._item_added(event["item"])
            return
        item_id = event["item"]["id"] if kind == "response.output_item.done" else event.get("item_id")
        item = self._items.get(item_id or "")
        if item is None:
            return
        where: Where = {"response_id": self.response.id, "item_id": item.id, "output_index": self._index(item)}
        if isinstance(item, FunctionCall):
            await self._on_call_event(kind, event, item, where)
            return
        if kind == "response.output_item.done":
            item.status = event["item"]["status"]
            await self._item_done(item)
            return
        handler = self._part_handlers.get(kind or "")
        if handler is not None:
            where["content_index"] = event["content_index"]
            await handler(item, where, event)

    async def _item_added(self, added: dict[str, Any]) -> None:
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

    async def _on_call_event(self, kind: str | None, event: dict[str, Any], call: FunctionCall, where: Where) -> None:
        if kind == "response.function_call_arguments.delta":
            call.arguments += event["delta"]
            await self._send(server_event(kind, **where, call_id=call.call_id, delta=event["delta"]))
        elif kind == "response.function_call_arguments.done":
            call.name, call.arguments = event["name"], event["arguments"]
            await self._send(
                server_event(kind, **where, call_id=call.call_id, name=call.name, arguments=call.arguments)
            )
        elif kind == "response.output_item.done":
            done = event["item"]
            # a loader may send the function's name after its first chunk
            call.name, call.arguments, call.status = done["name"], done["arguments"], done["status"]
            await self._item_done(call)

    def _index(self, item: AssistantMessage | FunctionCall) -> int:
        return next(i for i, other in enumerate(self.response.output) if other is item)

    async def _add_item(self, item: AssistantMessage | FunctionCall) -> None:
        self._items[item.id] = item
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
        stored = self._conversation.get(item.id) if self._conversation is not None else None
        if self._conversation is not None and stored is not None:
            # a truncated message is stored as a cut copy
            stored.status = item.status
            before = self._conversation.before(item.id)
            await self._send(server_event("conversation.item.done", previous_item_id=before, item=stored))

    async def _end(self, status: ResponseStatus, details: StatusDetails | None) -> None:
        await self._output.end()
        for item in self.response.output:
            if item.status == "in_progress":
                item.status = "incomplete"
                await self._item_done(item)
        self.response.status = status
        self.response.status_details = details
        await self._send(response_event("response.done", self.response))
