from __future__ import annotations

import logging
import threading

from app.audio.io import AudioIO, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ
from app.events import AudioFrameEvent, EventBus, StatusEvent


class AudioEngine:
    """Captures mic audio and plays assistant audio through local speakers."""

    def __init__(
        self,
        event_bus: EventBus,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        self._event_bus = event_bus
        self._audio_io = AudioIO(sample_rate_hz=sample_rate_hz, channels=channels)
        self._listening = False
        self._capture_thread: threading.Thread | None = None
        self._stop_capture = threading.Event()
        self._unsubscribe_audio = self._event_bus.subscribe(AudioFrameEvent, self._on_audio_frame_event)
        self._logger = logging.getLogger(__name__)

    @property
    def listening(self) -> bool:
        return self._listening

    def start_listening(self) -> None:
        if self._listening:
            return
        try:
            self._audio_io.start_output()
            self._audio_io.start_input()
        except Exception as exc:
            self._logger.exception("Failed to start audio streams: %s", exc)
            self._audio_io.stop_input()
            self._audio_io.stop_output()
            self._event_bus.publish(
                StatusEvent(component="audio", status="error", detail=f"Unable to start audio: {exc}")
            )
            return
        self._listening = True
        self._stop_capture.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="audio-capture-loop",
            daemon=True,
        )
        self._capture_thread.start()
        self._logger.info("Audio listening started (sample_rate=%dHz).", self._audio_io.sample_rate_hz)
        self._event_bus.publish(StatusEvent(component="audio", status="listening"))

    def stop_listening(self) -> None:
        if not self._listening:
            return
        self._stop_capture.set()
        capture_thread = self._capture_thread
        self._capture_thread = None
        if capture_thread is not None:
            capture_thread.join(timeout=1.5)

        self._audio_io.stop_input()
        self._audio_io.stop_output()
        self._listening = False
        self._logger.info("Audio listening stopped.")
        self._event_bus.publish(StatusEvent(component="audio", status="idle"))

    def shutdown(self) -> None:
        self.stop_listening()
        self._unsubscribe_audio()

    def _capture_loop(self) -> None:
        while not self._stop_capture.is_set():
            frame = self._audio_io.read_input_chunk(timeout=0.1)
            if frame is None:
                continue
            self._event_bus.publish(
                AudioFrameEvent(
                    frame=frame,
                    sample_rate_hz=self._audio_io.sample_rate_hz,
                    channels=self._audio_io.channels,
                    source="mic",
                )
            )

    def _on_audio_frame_event(self, event: AudioFrameEvent) -> None:
        if event.source != "assistant":
            return
        self._audio_io.enqueue_output_chunk(event.frame)
