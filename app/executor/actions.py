from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Callable
import webbrowser

from app.events import EventBus, NotesUpdateEvent
from app.permissions.state import PermissionState, PermissionStateError, PermissionStateMachine


@dataclass
class ActionResult:
    executed: bool
    detail: str


class ActionExecutor:
    def __init__(
        self,
        permission_state: PermissionStateMachine,
        event_bus: EventBus,
        dry_run: bool = False,
        actions_log_path: Path | str = "logs/actions.jsonl",
        open_url_fn: Callable[[str], bool] | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._permission_state = permission_state
        self._event_bus = event_bus
        self._dry_run = dry_run
        self._actions_log_path = Path(actions_log_path)
        self._open_url_fn = open_url_fn or webbrowser.open
        self._now_factory = now_factory or (lambda: datetime.now(tz=UTC))

    def execute(self, action_id: str) -> ActionResult:
        proposal = self._permission_state.get_proposal(action_id)
        state = self._permission_state.state_for(action_id)
        if state != PermissionState.APPROVED:
            detail = f"execution blocked: action_id={action_id} state={state.value if state else 'unknown'}"
            self._append_log(
                event_type="execution_blocked",
                action_id=action_id,
                tool=proposal.tool,
                executed=False,
                detail=detail,
            )
            raise PermissionError(detail)

        if self._dry_run or not self._permission_state.can_execute(action_id):
            detail = "dry-run prevented execution"
            try:
                self._permission_state.mark_execution_result(
                    action_id=action_id,
                    executed=False,
                    detail=detail,
                )
            except PermissionStateError:
                pass
            self._append_log(
                event_type="execution_result",
                action_id=action_id,
                tool=proposal.tool,
                executed=False,
                detail=detail,
            )
            return ActionResult(executed=False, detail=detail)

        try:
            if proposal.tool == "open_url":
                return self._execute_open_url(action_id=action_id, url=proposal.args["url"])
            if proposal.tool == "write_notes":
                return self._execute_write_notes(
                    action_id=action_id,
                    text=proposal.args["text"],
                    mode=proposal.args["mode"],
                )
        except Exception as exc:
            detail = f"execution failed: {exc}"
            try:
                self._permission_state.mark_execution_result(
                    action_id=action_id,
                    executed=False,
                    detail=detail,
                )
            except PermissionStateError:
                pass
            self._append_log(
                event_type="execution_result",
                action_id=action_id,
                tool=proposal.tool,
                executed=False,
                detail=detail,
            )
            return ActionResult(executed=False, detail=detail)

        detail = f"unsupported tool: {proposal.tool}"
        try:
            self._permission_state.mark_execution_result(action_id=action_id, executed=False, detail=detail)
        except PermissionStateError:
            pass
        self._append_log(
            event_type="execution_result",
            action_id=action_id,
            tool=proposal.tool,
            executed=False,
            detail=detail,
        )
        return ActionResult(executed=False, detail=detail)

    def _execute_open_url(self, action_id: str, url: str) -> ActionResult:
        opened = bool(self._open_url_fn(url))
        detail = f"open_url returned {opened}"
        self._permission_state.mark_execution_result(action_id=action_id, executed=opened, detail=detail)
        self._append_log(
            event_type="execution_result",
            action_id=action_id,
            tool="open_url",
            executed=opened,
            detail=detail,
            args={"url": url},
        )
        return ActionResult(executed=opened, detail=detail)

    def _execute_write_notes(self, action_id: str, text: str, mode: str) -> ActionResult:
        self._event_bus.publish(NotesUpdateEvent(action_id=action_id, text=text, mode=mode))
        detail = f"notes_update emitted (mode={mode})"
        self._permission_state.mark_execution_result(action_id=action_id, executed=True, detail=detail)
        self._append_log(
            event_type="execution_result",
            action_id=action_id,
            tool="write_notes",
            executed=True,
            detail=detail,
            args={"text": text, "mode": mode},
        )
        return ActionResult(executed=True, detail=detail)

    def _append_log(self, event_type: str, action_id: str, **extra: Any) -> None:
        self._actions_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": self._now_factory().isoformat(),
            "event_type": event_type,
            "action_id": action_id,
            **extra,
        }
        with self._actions_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
