#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice>=0.5", "websockets>=13"]
# ///
"""Talk to a modelship pipeline from the terminal: streams the microphone to /v1/realtime, lets server VAD find
each turn, prints what was heard and answered, and plays the spoken answer.

    uv run scripts/realtime_mic.py --pipeline assistant

It needs PortAudio (`apt install libportaudio2`, `brew install portaudio`). Use headphones: the session doesn't
tell its own voice from yours yet, so a speaker would answer itself."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import threading

import sounddevice as sd
from websockets.asyncio.client import connect


class Player:
    """Plays 16-bit mono PCM as it arrives."""

    def __init__(self, rate: int) -> None:
        self._pending = bytearray()
        self._lock = threading.Lock()
        self._stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", callback=self._fill)
        self._stream.start()

    def play(self, pcm: bytes) -> None:
        with self._lock:
            self._pending += pcm

    def _fill(self, out, frames, _time, _status) -> None:
        wanted = frames * 2
        with self._lock:
            chunk = bytes(self._pending[:wanted])
            del self._pending[:wanted]
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


def _show(event: dict, player: Player) -> None:
    kind = event["type"]
    if kind == "response.output_audio.delta":
        player.play(base64.b64decode(event["delta"]))
    elif kind == "input_audio_buffer.speech_started":
        print("\n[listening]", flush=True)
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
                    _show(json.loads(message), player)
            finally:
                microphone.cancel()
    finally:
        player.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
