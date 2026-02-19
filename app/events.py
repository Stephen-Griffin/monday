from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Callable, DefaultDict, Literal, TypeVar
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
class ProposalCreatedEvent:
    action_id: str
    tool: str
    args: dict[str, Any]
    reason: str
    event_type: Literal["proposal_created"] = "proposal_created"
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class DecisionRecordedEvent:
    action_id: str
    decision: Literal["approved", "rejected", "cancelled"]
    reason: str | None = None
    event_type: Literal["decision_recorded"] = "decision_recorded"
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class ExecutionResultEvent:
    action_id: str
    tool: str | None
    executed: bool
    detail: str
    event_type: Literal["execution_result"] = "execution_result"
    event_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class NotesUpdateEvent:
    action_id: str
    mode: Literal["append", "replace"]
    text: str
    event_type: Literal["notes_update"] = "notes_update"
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
    decision: Literal["approved", "rejected", "cancelled"]
    reason: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid4()))
      
      
AppEvent = (
    AudioFrameEvent
    | TranscriptEvent
    | ToolProposalEvent
    | ProposalCreatedEvent
    | DecisionRecordedEvent
    | ExecutionResultEvent
    | NotesUpdateEvent
    | StatusEvent
    | ActionDecisionEvent
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
