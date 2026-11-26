"""The gateway's `/v1/realtime` relay: refusing what it can't open, handing the client's frames to the session in
order and together while a hand-over is in flight, stopping reading once enough wait, and closing the client's
WebSocket the way the session or its replica ended."""

import asyncio
import json

import pytest

from modelship.infer.pipelines import relay
from modelship.infer.pipelines.relay import RealtimeClosed, RealtimeOpened, RealtimeOpening
from modelship.openai.realtime_relay import serve_relay

VOICE = {"name": "voice", "type": "assistant", "assistant_config": {"stt": "w"}}
READY = {"voice": {"config": VOICE, "app": "gw.voice-p0", "ready": True}}


class _WebSocket:
    """A client's WebSocket: the frames it sends wait in *inbound*, a None in it being the client leaving."""

    def __init__(self, model: str | None = "voice") -> None:
        self.query_params = {"model": model} if model is not None else {}
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.closed: int | None = None
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict:
        frame = await self.inbound.get()
        if frame is None:
            return {"type": "websocket.disconnect"}
        if isinstance(frame, bytes):
            return {"type": "websocket.receive", "bytes": frame}
        return {"type": "websocket.receive", "text": frame}

    async def send_text(self, text: str) -> None:
        if self.closed is not None:
            raise RuntimeError("closed")
        self.sent.append(text)

    async def close(self, code: int = 1000) -> None:
        if self.closed is None:
            self.closed = code
        await self.inbound.put(None)


class _Link:
    """A pipeline replica that streams *frames* after RealtimeOpened, then holds the stream open until the session
    ends; each hand-over waits for *gate* when it's set."""

    def __init__(self, *frames, held: bool = True) -> None:
        self.frames = list(frames)
        self.held = held
        self.openings: list[RealtimeOpening] = []
        self.batches: list[list] = []
        self.left: list[str] = []
        self.stopped = False
        self.gate: asyncio.Event | None = None
        self.ended = asyncio.Event()
        self.answer = True

    async def _stream(self):
        yield RealtimeOpened()
        for frame in self.frames:
            if isinstance(frame, Exception):
                raise frame
            yield frame
        if self.held:
            await self.ended.wait()

    def open(self, opening: RealtimeOpening):
        self.openings.append(opening)
        return self._stream()

    async def send(self, key: str, frames: list) -> bool:
        if self.gate is not None:
            await self.gate.wait()
        self.batches.append(frames)
        return self.answer

    async def leave(self, key: str) -> None:
        self.left.append(key)
        self.ended.set()

    async def stop(self) -> None:
        self.stopped = True


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _refusal(websocket: _WebSocket, pipelines: dict, link=None) -> dict:
    await serve_relay(websocket, pipelines, lambda name: link, {}, "")  # type: ignore[arg-type]
    assert websocket.closed == 1008
    [sent] = websocket.sent
    return json.loads(sent)["error"]


class TestRefusals:
    @pytest.mark.asyncio
    async def test_without_a_model_it_is_refused(self):
        error = await _refusal(_WebSocket(None), READY)
        assert (error["code"], error["param"]) == ("missing_required_parameter", "model")

    @pytest.mark.asyncio
    async def test_an_unknown_pipeline_is_refused(self):
        error = await _refusal(_WebSocket("nope"), READY)
        assert (error["code"], error["message"]) == ("model_not_found", "No pipeline 'nope' on this gateway.")

    @pytest.mark.asyncio
    async def test_a_pipeline_not_ready_is_refused(self):
        pipelines = {"voice": {**READY["voice"], "ready": False}}
        error = await _refusal(_WebSocket(), pipelines, _Link())
        assert (error["type"], error["code"]) == ("server_error", None)

    @pytest.mark.asyncio
    async def test_a_pipeline_whose_app_cant_be_reached_is_refused(self):
        error = await _refusal(_WebSocket(), READY, None)
        assert (error["type"], error["code"]) == ("server_error", None)


async def _relay(link: _Link, websocket: _WebSocket | None = None) -> tuple[asyncio.Task, _WebSocket]:
    websocket = websocket or _WebSocket()
    task = asyncio.ensure_future(
        serve_relay(websocket, READY, lambda name: link, {"x-device": "kitchen"}, "caller")  # type: ignore[arg-type]
    )
    await _settle()
    return task, websocket


class TestRelaying:
    @pytest.mark.asyncio
    async def test_a_session_opens_with_the_clients_headers_and_identity_under_its_own_key(self):
        link = _Link()
        task, websocket = await _relay(link)
        [opening] = link.openings
        assert (opening.headers, opening.identity) == ({"x-device": "kitchen"}, "caller")
        assert opening.key
        await websocket.inbound.put(None)
        await task
        assert link.left == [opening.key]

    @pytest.mark.asyncio
    async def test_the_sessions_frames_reach_the_client(self):
        link = _Link('{"type": "session.created"}', '{"type": "conversation.created"}')
        task, websocket = await _relay(link)
        assert websocket.sent == ['{"type": "session.created"}', '{"type": "conversation.created"}']
        await websocket.inbound.put(None)
        await task

    @pytest.mark.asyncio
    async def test_frames_arriving_during_a_hand_over_go_together_in_order(self):
        link = _Link()
        link.gate = asyncio.Event()
        task, websocket = await _relay(link)
        for n in range(5):
            await websocket.inbound.put(f"e{n}")
            await _settle()
        link.gate.set()
        await _settle()
        assert link.batches == [["e0"], ["e1", "e2", "e3", "e4"]]
        await websocket.inbound.put(None)
        await task

    @pytest.mark.asyncio
    async def test_binary_frames_go_through_as_bytes(self):
        link = _Link()
        task, websocket = await _relay(link)
        await websocket.inbound.put(b"\x00\x01")
        await _settle()
        assert link.batches == [[b"\x00\x01"]]
        await websocket.inbound.put(None)
        await task

    @pytest.mark.asyncio
    async def test_reading_stops_while_enough_waits_to_be_handed_over(self, monkeypatch):
        monkeypatch.setattr(relay, "MAX_QUEUED_BYTES", 250)
        link = _Link()
        link.gate = asyncio.Event()
        task, websocket = await _relay(link)
        await websocket.inbound.put("x" * 100)
        await _settle()
        for _ in range(10):
            await websocket.inbound.put("y" * 100)
        await _settle()
        # one frame in the stuck hand-over, three waiting, the rest unread
        assert websocket.inbound.qsize() == 7
        link.gate.set()
        await _settle()
        assert sum(len(batch) for batch in link.batches) == 11
        await websocket.inbound.put(None)
        await task

    @pytest.mark.asyncio
    async def test_a_client_that_leaves_has_its_last_frames_handed_over_first(self):
        link = _Link()
        link.gate = asyncio.Event()
        task, websocket = await _relay(link)
        await websocket.inbound.put("first")
        await _settle()
        await websocket.inbound.put("last")
        await websocket.inbound.put(None)
        await _settle()
        assert link.left == []
        link.gate.set()
        await task
        assert link.batches == [["first"], ["last"]]
        assert len(link.left) == 1
        assert link.stopped

    @pytest.mark.asyncio
    async def test_the_session_closing_closes_the_websocket_with_its_code(self):
        link = _Link('{"type": "error"}', RealtimeClosed(1008), held=False)
        task, websocket = await _relay(link)
        await task
        assert websocket.sent == ['{"type": "error"}']
        assert websocket.closed == 1008
        assert link.stopped

    @pytest.mark.asyncio
    async def test_a_replica_that_fails_closes_the_websocket_as_an_error(self):
        link = _Link(ConnectionError("replica died"), held=False)
        task, websocket = await _relay(link)
        await task
        assert websocket.closed == 1011

    @pytest.mark.asyncio
    async def test_a_replica_that_lost_the_session_closes_the_websocket_as_an_error(self):
        link = _Link()
        link.answer = False
        task, websocket = await _relay(link)
        await websocket.inbound.put("e0")
        await task
        assert websocket.closed == 1011
