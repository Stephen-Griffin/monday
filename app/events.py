from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Callable, DefaultDict, TypeVar
from uuid import uuid4

T = TypeVar("T")


@dataclass(frozen=True)
class AudioFrameEvent:
    frame: bytes
    sample_rate_hz: int
    channels: int
    source: str = "mic"
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class TranscriptEvent:
    source: str
    text: str
    is_final: bool
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class ToolProposalEvent:
    proposal_id: str
    tool: str
    args: dict[str, Any]
    reason: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class StatusEvent:
    component: str
    status: str
    detail: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class ActionDecisionEvent:
    action_id: str
    decision: str
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class NotesUpdateEvent:
    content: str
    mode: str = "append"
    source: str = "jarvis"
    event_id: str = field(default_factory=lambda: str(uuid4()))


AppEvent = (
    AudioFrameEvent
    | TranscriptEvent
    | ToolProposalEvent
    | StatusEvent
    | ActionDecisionEvent
    | NotesUpdateEvent
)
AppEventHandler = Callable[[AppEvent], None]


class EventBus:
    """Simple in-process pub/sub for cross-module communication."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._typed_handlers: DefaultDict[type[Any], list[Callable[[Any], None]]] = defaultdict(list)
        self._global_handlers: list[AppEventHandler] = []

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> Callable[[], None]:
        with self._lock:
            self._typed_handlers[event_type].append(handler)

        def _unsubscribe() -> None:
            with self._lock:
                handlers = self._typed_handlers.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)

        return _unsubscribe

    def subscribe_all(self, handler: AppEventHandler) -> Callable[[], None]:
        with self._lock:
            self._global_handlers.append(handler)

        def _unsubscribe() -> None:
            with self._lock:
                if handler in self._global_handlers:
                    self._global_handlers.remove(handler)

        return _unsubscribe

    def publish(self, event: AppEvent) -> None:
        with self._lock:
            typed_handlers = list(self._typed_handlers.get(type(event), []))
            global_handlers = list(self._global_handlers)

        for handler in typed_handlers + global_handlers:
            try:
                handler(event)
            except Exception:
                self._logger.exception("Event handler failed for %s", type(event).__name__)
