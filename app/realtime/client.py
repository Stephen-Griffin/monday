from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any
from urllib.parse import quote_plus

import websockets

from app.audio.io import DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ
from app.events import AudioFrameEvent, EventBus, StatusEvent, TranscriptEvent

WS_URL_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"


class RealtimeClient:
    """WebSocket client for OpenAI Realtime with background reconnect support."""

    def __init__(self, event_bus: EventBus, model: str, api_key: str) -> None:
        self._event_bus = event_bus
        self._model = model
        self._api_key = api_key
        self._logger = logging.getLogger(__name__)

        self._state_lock = threading.RLock()
        self._desired_connected = False
        self._connected = False
        self._listening = False
        self._pending_finalize = False
        self._sent_audio_since_listen = False

        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mic_queue: asyncio.Queue[bytes] | None = None
        self._ws: Any | None = None

        self._event_bus.subscribe(AudioFrameEvent, self._on_audio_frame_event)

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connected

    def connect(self) -> None:
        if not self._api_key:
            self._event_bus.publish(
                StatusEvent(
                    component="realtime",
                    status="blocked",
                    detail="OPENAI_API_KEY is missing.",
                )
            )
            return

        with self._state_lock:
            self._desired_connected = True
            self._stop_requested.clear()
            should_start = self._thread is None or not self._thread.is_alive()

        if should_start:
            self._thread = threading.Thread(
                target=self._thread_main,
                name="realtime-client-loop",
                daemon=True,
            )
            self._thread.start()
        self._event_bus.publish(StatusEvent(component="realtime", status="connecting"))

    def disconnect(self) -> None:
        with self._state_lock:
            self._desired_connected = False
            self._listening = False
            self._pending_finalize = False
            self._sent_audio_since_listen = False
            loop = self._loop
            thread = self._thread

        self._stop_requested.set()

        if loop is not None:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(self._close_websocket()))

        if thread is not None:
            thread.join(timeout=4.0)

        self._set_connected(False)
        self._event_bus.publish(StatusEvent(component="realtime", status="disconnected"))

    def start_listening(self) -> None:
        with self._state_lock:
            self._listening = True
            self._pending_finalize = False
            self._sent_audio_since_listen = False
        self._event_bus.publish(StatusEvent(component="realtime", status="listening"))

    def stop_listening(self) -> None:
        with self._state_lock:
            was_listening = self._listening
            self._listening = False
            self._pending_finalize = was_listening
        self._event_bus.publish(StatusEvent(component="realtime", status="idle"))

    def send_audio_frame(self, frame: bytes, sample_rate_hz: int, channels: int) -> None:
        event = AudioFrameEvent(
            frame=frame,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            source="mic",
        )
        self._on_audio_frame_event(event)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with self._state_lock:
            self._loop = loop
            self._mic_queue = asyncio.Queue(maxsize=256)

        try:
            loop.run_until_complete(self._run_forever())
        finally:
            self._set_connected(False)
            with self._state_lock:
                self._loop = None
                self._mic_queue = None
                self._ws = None
                self._thread = None
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _run_forever(self) -> None:
        backoff_seconds = 0.5
        while not self._stop_requested.is_set():
            with self._state_lock:
                should_connect = self._desired_connected
            if not should_connect:
                await asyncio.sleep(0.1)
                continue

            try:
                await self._run_connection()
                backoff_seconds = 0.5
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop_requested.is_set():
                    break
                self._logger.warning("Realtime connection failed: %s", exc)
                self._event_bus.publish(
                    StatusEvent(
                        component="realtime",
                        status="reconnecting",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 8.0)

    async def _run_connection(self) -> None:
        ws_url = WS_URL_TEMPLATE.format(model=quote_plus(self._model))
        ws = await self._open_websocket(ws_url)
        with self._state_lock:
            self._ws = ws

        self._set_connected(True)
        await self._send_session_update(ws)

        receive_task = asyncio.create_task(self._receive_loop(ws), name="realtime-recv")
        send_task = asyncio.create_task(self._send_loop(ws), name="realtime-send")

        done, pending = await asyncio.wait(
            {receive_task, send_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass

        first_error: Exception | None = None
        for task in done:
            try:
                task.result()
            except Exception as exc:  # pragma: no cover - defensive branch
                first_error = exc
                break

        await self._close_websocket()
        self._set_connected(False)
        if first_error is not None:
            raise first_error

    async def _open_websocket(self, ws_url: str) -> Any:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        kwargs = dict(
            ping_interval=20,
            ping_timeout=20,
            close_timeout=2,
            max_size=8 * 1024 * 1024,
        )
        try:
            return await websockets.connect(ws_url, additional_headers=headers, **kwargs)
        except TypeError:
            return await websockets.connect(ws_url, extra_headers=headers, **kwargs)

    async def _close_websocket(self) -> None:
        with self._state_lock:
            ws = self._ws
            self._ws = None

        if ws is None:
            return

        try:
            await ws.close()
        except Exception:
            self._logger.debug("Ignoring websocket close failure.", exc_info=True)

    async def _send_session_update(self, ws: Any) -> None:
        event = {
            "type": "session.update",
            "session": {
                "output_modalities": ["text", "audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": DEFAULT_SAMPLE_RATE_HZ},
                        "turn_detection": {"type": "semantic_vad"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm"},
                        "voice": "alloy",
                    },
                },
            },
        }
        await ws.send(json.dumps(event))

    async def _send_loop(self, ws: Any) -> None:
        while not self._stop_requested.is_set():
            with self._state_lock:
                desired = self._desired_connected
                listening = self._listening
            if not desired:
                return

            if not listening:
                self._drain_mic_queue()
                await self._finalize_turn_if_needed(ws)
                await asyncio.sleep(0.05)
                continue

            chunk = await self._read_next_mic_chunk(timeout_seconds=0.1)
            if chunk is None:
                await self._finalize_turn_if_needed(ws)
                continue

            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            with self._state_lock:
                self._sent_audio_since_listen = True

    async def _receive_loop(self, ws: Any) -> None:
        while not self._stop_requested.is_set():
            message = await ws.recv()
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            event = json.loads(message)
            self._handle_server_event(event)

    async def _finalize_turn_if_needed(self, ws: Any) -> None:
        with self._state_lock:
            pending_finalize = self._pending_finalize
            had_audio = self._sent_audio_since_listen
            if pending_finalize:
                self._pending_finalize = False
                self._sent_audio_since_listen = False

        if not pending_finalize or not had_audio:
            return

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

    async def _read_next_mic_chunk(self, timeout_seconds: float) -> bytes | None:
        queue_obj = self._mic_queue
        if queue_obj is None:
            await asyncio.sleep(timeout_seconds)
            return None
        try:
            return await asyncio.wait_for(queue_obj.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None

    def _drain_mic_queue(self) -> None:
        queue_obj = self._mic_queue
        if queue_obj is None:
            return
        while True:
            try:
                queue_obj.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if not event_type:
            return

        # TODO(Agent 2): Verify if/when OpenAI drops one of these aliases and remove dead branches.
        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            audio_b64 = event.get("delta")
            if isinstance(audio_b64, str):
                self._publish_assistant_audio(audio_b64)
            return

        # TODO(Agent 2): Confirm final transcript event names against latest Realtime docs after GA changes.
        if event_type in {
            "response.output_audio_transcript.delta",
            "response.audio_transcript.delta",
            "response.output_text.delta",
        }:
            delta = self._extract_text(event, "delta")
            if delta:
                self._event_bus.publish(TranscriptEvent(source="assistant", text=delta, is_final=False))
            return

        if event_type in {
            "response.output_audio_transcript.done",
            "response.audio_transcript.done",
            "response.output_text.done",
        }:
            transcript = self._extract_text(event, "transcript", "text", "delta")
            if transcript:
                self._event_bus.publish(TranscriptEvent(source="assistant", text=transcript, is_final=True))
            return

        if event_type in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcript.delta",
        }:
            delta = self._extract_text(event, "delta")
            if delta:
                self._event_bus.publish(TranscriptEvent(source="user", text=delta, is_final=False))
            return

        if event_type in {
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcript.completed",
            "conversation.item.input_audio_transcription.done",
        }:
            transcript = self._extract_text(event, "transcript", "text")
            if transcript:
                self._event_bus.publish(TranscriptEvent(source="user", text=transcript, is_final=True))
            return

        if event_type == "error":
            error = event.get("error", {})
            detail = None
            if isinstance(error, dict):
                detail = error.get("message") or json.dumps(error)
            self._event_bus.publish(
                StatusEvent(component="realtime", status="error", detail=detail or "Unknown server error.")
            )
            return

        if event_type == "input_audio_buffer.speech_started":
            self._event_bus.publish(StatusEvent(component="realtime", status="speech_started"))
            return

        if event_type == "input_audio_buffer.speech_stopped":
            self._event_bus.publish(StatusEvent(component="realtime", status="speech_stopped"))
            return

    def _publish_assistant_audio(self, audio_b64: str) -> None:
        try:
            pcm_bytes = base64.b64decode(audio_b64)
        except Exception:
            self._logger.warning("Skipping invalid assistant audio delta.")
            return
        if not pcm_bytes:
            return
        self._event_bus.publish(
            AudioFrameEvent(
                frame=pcm_bytes,
                sample_rate_hz=DEFAULT_SAMPLE_RATE_HZ,
                channels=DEFAULT_CHANNELS,
                source="assistant",
            )
        )

    def _on_audio_frame_event(self, event: AudioFrameEvent) -> None:
        if event.source != "mic":
            return
        if event.channels != DEFAULT_CHANNELS or event.sample_rate_hz != DEFAULT_SAMPLE_RATE_HZ:
            self._logger.warning(
                "Dropping mic frame with unsupported format: %d Hz / %d channels",
                event.sample_rate_hz,
                event.channels,
            )
            return

        with self._state_lock:
            listening = self._listening
            loop = self._loop
        if not listening or loop is None:
            return

        loop.call_soon_threadsafe(self._enqueue_mic_chunk, event.frame)

    def _enqueue_mic_chunk(self, chunk: bytes) -> None:
        queue_obj = self._mic_queue
        if queue_obj is None:
            return
        try:
            queue_obj.put_nowait(chunk)
            return
        except asyncio.QueueFull:
            pass

        try:
            queue_obj.get_nowait()
        except asyncio.QueueEmpty:
            pass

        try:
            queue_obj.put_nowait(chunk)
        except asyncio.QueueFull:
            # Queue remained full due to concurrent producer; drop frame.
            pass

    def _set_connected(self, connected: bool) -> None:
        with self._state_lock:
            if self._connected == connected:
                return
            self._connected = connected
        status = "connected" if connected else "disconnected"
        self._event_bus.publish(StatusEvent(component="realtime", status=status))

    @staticmethod
    def _extract_text(event: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = event.get(key)
            if isinstance(value, str):
                return value
        return ""
