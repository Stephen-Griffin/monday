from __future__ import annotations

import logging

from app.events import EventBus, StatusEvent


class RealtimeClient:
    """
    Placeholder Realtime API client for module integration.

    TODO(Agent 2): Replace internals with a websocket-based client that streams
    mic audio to OpenAI Realtime and publishes transcript/tool events back to bus.
    """

    def __init__(self, event_bus: EventBus, model: str, api_key: str) -> None:
        self._event_bus = event_bus
        self._model = model
        self._api_key = api_key
        self._connected = False
        self._logger = logging.getLogger(__name__)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return
        self._logger.info("Realtime stub connected (model=%s).", self._model)
        if not self._api_key:
            self._logger.warning("OPENAI_API_KEY is not configured; realtime calls will fail until set.")
        self._connected = True
        self._event_bus.publish(StatusEvent(component="realtime", status="connected"))

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        self._logger.info("Realtime stub disconnected.")
        self._event_bus.publish(StatusEvent(component="realtime", status="disconnected"))

    def send_audio_frame(self, frame: bytes, sample_rate_hz: int, channels: int) -> None:
        if not self._connected:
            self._logger.debug("Dropping audio frame because realtime is disconnected.")
            return
        self._logger.debug(
            "Realtime stub received frame bytes=%d sample_rate=%d channels=%d",
            len(frame),
            sample_rate_hz,
            channels,
        )

