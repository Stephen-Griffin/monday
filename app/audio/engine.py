from __future__ import annotations

import logging

from app.events import EventBus, StatusEvent


class AudioEngine:
    """
    Placeholder audio capture/playback engine.

    TODO(Agent 2): Implement sounddevice input/output streams and publish
    `AudioFrameEvent` events into the event bus.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._listening = False
        self._logger = logging.getLogger(__name__)

    @property
    def listening(self) -> bool:
        return self._listening

    def start_listening(self) -> None:
        if self._listening:
            return
        self._listening = True
        self._logger.info("Audio stub listening started.")
        self._event_bus.publish(StatusEvent(component="audio", status="listening"))

    def stop_listening(self) -> None:
        if not self._listening:
            return
        self._listening = False
        self._logger.info("Audio stub listening stopped.")
        self._event_bus.publish(StatusEvent(component="audio", status="idle"))

