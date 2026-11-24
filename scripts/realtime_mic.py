#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice>=0.5", "websockets>=13"]
# ///
"""Talk to a modelship pipeline from the terminal: streams the microphone to /v1/realtime, lets server VAD find
each turn, prints what was heard and answered, and plays the spoken answer. Talking over the answer stops it, and
the session hears how much of it was played.

    uv run scripts/realtime_mic.py --pipeline assistant

It needs PortAudio (`apt install libportaudio2`, `brew install portaudio`). It cancels no echo: use headphones, or a
pipeline with `duplex: half`, which doesn't listen while it speaks."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import threading

import sounddevice as sd
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


class Player:
    """Plays 16-bit mono PCM as it arrives, counting how much of each item's audio it played."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self._pending = bytearray()
        # the items whose audio waits to play, in order, with the bytes of each left
        self._queued: list[list] = []
        self._played: dict[str, int] = {}
        # items cut off, whose audio still arriving is dropped
        self._cut: set[str] = set()
        self._lock = threading.Lock()
        self._stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", callback=self._fill)
        self._stream.start()

    def play(self, item_id: str, pcm: bytes) -> None:
        with self._lock:
            if item_id in self._cut:
                return
            self._pending += pcm
            if self._queued and self._queued[-1][0] == item_id:
                self._queued[-1][1] += len(pcm)
            else:
                self._queued.append([item_id, len(pcm)])

    def stop(self) -> list[tuple[str, int]]:
        """Stops playing: each item cut off, with the ms of it played."""
        with self._lock:
            cut = [(item_id, self._played.get(item_id, 0) * 1000 // (self.rate * 2)) for item_id, _ in self._queued]
            self._cut.update(item_id for item_id, _ in cut)
            self._pending.clear()
            self._queued.clear()
        return cut

    def _fill(self, out, frames, _time, _status) -> None:
        wanted = frames * 2
        with self._lock:
            chunk = bytes(self._pending[:wanted])
            del self._pending[:wanted]
            left = len(chunk)
            while left and self._queued:
                item = self._queued[0]
                used = min(left, item[1])
                item[1] -= used
                left -= used
                self._played[item[0]] = self._played.get(item[0], 0) + used
                if not item[1]:
                    self._queued.pop(0)
        out[: len(chunk)] = chunk
        out[len(chunk) :] = bytes(wanted - len(chunk))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


async def _send_microphone(ws, rate: int) -> None:
    loop = asyncio.get_running_loop()
    chunks: asyncio.Queue[bytes] = asyncio.Queue()

    def captured(data, _frames, _time, _status) -> None:
        loop.call_soon_threadsafe(chunks.put_nowait, bytes(data))

    with sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", blocksize=rate // 10, callback=captured):
        while True:
            pcm = await chunks.get()
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}))


async def _handle(ws, event: dict, player: Player) -> None:
    kind = event["type"]
    if kind == "response.output_audio.delta":
        player.play(event["item_id"], base64.b64decode(event["delta"]))
    elif kind == "input_audio_buffer.speech_started":
        print("\n[listening]", flush=True)
        # the session cuts its answer off: tell it how much was heard
        for item_id, played_ms in player.stop():
            truncate = {"item_id": item_id, "content_index": 0, "audio_end_ms": played_ms}
            await ws.send(json.dumps({"type": "conversation.item.truncate", **truncate}))
    elif kind == "conversation.item.truncated":
        print(f"[heard {event['audio_end_ms']} ms of it]", flush=True)
    elif kind == "input_audio_buffer.speech_stopped":
        print("[heard you]", flush=True)
    elif kind == "conversation.item.input_audio_transcription.completed":
        print(f"you: {event['transcript']}", flush=True)
    elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
        print(event["delta"], end="", flush=True)
    elif kind == "response.done":
        print(f"\n[{event['response']['status']}]", flush=True)
    elif kind == "error":
        print(f"\n[error] {event['error'].get('message')}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pipeline", required=True, help="the pipeline to talk to")
    parser.add_argument("--url", default="ws://localhost:8000/modelship/v1/realtime", help="the gateway's /v1/realtime")
    parser.add_argument("--rate", type=int, default=24000, help="the sample rate to record and play at")
    parser.add_argument("--voice", help="the tts model's voice; unset, its default")
    parser.add_argument("--instructions", help="the system prompt, over the pipeline's")
    args = parser.parse_args()

    audio_format = {"type": "audio/pcm", "rate": args.rate}
    session: dict = {"audio": {"input": {"format": audio_format}, "output": {"format": audio_format}}}
    if args.voice:
        session["audio"]["output"]["voice"] = args.voice
    if args.instructions:
        session["instructions"] = args.instructions

    player = Player(args.rate)
    try:
        async with connect(f"{args.url}?model={args.pipeline}", max_size=None) as ws:
            await ws.send(json.dumps({"type": "session.update", "session": session}))
            print("Speak; Ctrl+C quits.", flush=True)
            microphone = asyncio.ensure_future(_send_microphone(ws, args.rate))
            try:
                async for message in ws:
                    await _handle(ws, json.loads(message), player)
                print("\n[session closed]", flush=True)
            except ConnectionClosed as closed:
                print(f"\n[session closed: {closed}]", flush=True)
            finally:
                microphone.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionClosed):
                    await microphone
    finally:
        player.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
