from __future__ import annotations

import logging
from queue import Empty, Full, Queue
import threading

import numpy as np
import sounddevice as sd

DEFAULT_SAMPLE_RATE_HZ = 24_000
DEFAULT_CHANNELS = 1
DEFAULT_FRAMES_PER_CHUNK = 960  # 40 ms at 24 kHz


class AudioIO:
    """Low-level CoreAudio I/O with bounded queues for non-blocking handoff."""

    def __init__(
        self,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
        channels: int = DEFAULT_CHANNELS,
        frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK,
        input_queue_size: int = 256,
        output_queue_size: int = 256,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.frames_per_chunk = frames_per_chunk

        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._input_queue: Queue[bytes] = Queue(maxsize=input_queue_size)
        self._output_queue: Queue[bytes] = Queue(maxsize=output_queue_size)
        self._playback_buffer = bytearray()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    def start_input(self) -> None:
        with self._lock:
            if self._input_stream is not None:
                return
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate_hz,
                channels=self.channels,
                blocksize=self.frames_per_chunk,
                dtype="int16",
                callback=self._on_input,
            )
            self._input_stream.start()

    def start_output(self) -> None:
        with self._lock:
            if self._output_stream is not None:
                return
            self._output_stream = sd.OutputStream(
                samplerate=self.sample_rate_hz,
                channels=self.channels,
                blocksize=self.frames_per_chunk,
                dtype="int16",
                callback=self._on_output,
            )
            self._output_stream.start()

    def stop_input(self) -> None:
        with self._lock:
            stream = self._input_stream
            self._input_stream = None
        if stream is None:
            return
        stream.stop()
        stream.close()
        self.clear_input_queue()

    def stop_output(self) -> None:
        with self._lock:
            stream = self._output_stream
            self._output_stream = None
        if stream is None:
            return
        stream.stop()
        stream.close()
        self.clear_output_queue()

    def read_input_chunk(self, timeout: float = 0.0) -> bytes | None:
        try:
            return self._input_queue.get(timeout=timeout)
        except Empty:
            return None

    def enqueue_output_chunk(self, pcm_chunk: bytes) -> None:
        if not pcm_chunk:
            return
        self._put_non_blocking(self._output_queue, pcm_chunk)

    def clear_input_queue(self) -> None:
        self._drain_queue(self._input_queue)

    def clear_output_queue(self) -> None:
        self._drain_queue(self._output_queue)
        self._playback_buffer.clear()

    def _on_input(self, indata, _frames: int, _time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            self._logger.debug("Input stream status: %s", status)
        pcm = indata.copy().reshape(-1).tobytes()
        self._put_non_blocking(self._input_queue, pcm)

    def _on_output(self, outdata, frames: int, _time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            self._logger.debug("Output stream status: %s", status)

        required_bytes = frames * self.channels * 2
        while len(self._playback_buffer) < required_bytes:
            try:
                next_chunk = self._output_queue.get_nowait()
            except Empty:
                break
            self._playback_buffer.extend(next_chunk)

        if len(self._playback_buffer) >= required_bytes:
            payload = self._playback_buffer[:required_bytes]
            del self._playback_buffer[:required_bytes]
        else:
            payload = bytes(self._playback_buffer)
            self._playback_buffer.clear()
            payload += b"\x00" * (required_bytes - len(payload))

        outdata[:] = np.frombuffer(payload, dtype=np.int16).reshape(frames, self.channels)

    @staticmethod
    def _drain_queue(queue_obj: Queue[bytes]) -> None:
        while True:
            try:
                queue_obj.get_nowait()
            except Empty:
                return

    @staticmethod
    def _put_non_blocking(queue_obj: Queue[bytes], value: bytes) -> None:
        try:
            queue_obj.put_nowait(value)
            return
        except Full:
            pass

        try:
            queue_obj.get_nowait()
        except Empty:
            pass

        try:
            queue_obj.put_nowait(value)
        except Full:
            # Another producer won the race; dropping this frame is acceptable.
            pass
