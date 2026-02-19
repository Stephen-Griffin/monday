from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.events import DecisionRecordedEvent, EventBus, ExecutionResultEvent, ProposalCreatedEvent
from app.tools.contracts import ToolProposal


class PermissionState(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    state: PermissionState
    proposal: ToolProposal | None = None
    decision_reason: str | None = None
    execution_detail: str | None = None


class PermissionStateError(ValueError):
    pass


class PermissionStateMachine:
    def __init__(
        self,
        event_bus: EventBus,
        dry_run: bool = False,
        actions_log_path: Path | str = "logs/actions.jsonl",
        id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._dry_run = dry_run
        self._actions_log_path = Path(actions_log_path)
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._now_factory = now_factory or (lambda: datetime.now(tz=UTC))
        self._records: dict[str, ActionRecord] = {}

    def create_proposal(self, proposal: ToolProposal) -> ActionRecord:
        action_id = self._id_factory()
        record = ActionRecord(action_id=action_id, state=PermissionState.PROPOSED, proposal=proposal)
        self._records[action_id] = record
        self._event_bus.publish(
            ProposalCreatedEvent(
                action_id=action_id,
                tool=proposal.tool,
                args=proposal.args,
                reason=proposal.reason,
            )
        )
        self._append_log(
            event_type="proposal_created",
            action_id=action_id,
            state=record.state.value,
            proposal=asdict(proposal),
        )
        return record

    def reject_invalid_proposal(self, reason: str, raw_payload: Any) -> ActionRecord:
        action_id = self._id_factory()
        record = ActionRecord(
            action_id=action_id,
            state=PermissionState.REJECTED,
            decision_reason=reason,
        )
        self._records[action_id] = record
        self._event_bus.publish(
            DecisionRecordedEvent(
                action_id=action_id,
                decision="rejected",
                reason=reason,
            )
        )
        self._append_log(
            event_type="proposal_rejected_invalid",
            action_id=action_id,
            state=record.state.value,
            reason=reason,
            raw_payload=raw_payload,
        )
        return record

    def record_decision(self, action_id: str, approved: bool, reason: str | None = None) -> ActionRecord:
        record = self._require_record(action_id)
        if record.state != PermissionState.PROPOSED:
            raise PermissionStateError(
                f"cannot record decision for action_id={action_id} from state={record.state.value}"
            )

        next_state = PermissionState.APPROVED if approved else PermissionState.REJECTED
        next_record = replace(record, state=next_state, decision_reason=reason)
        self._records[action_id] = next_record
        self._event_bus.publish(
            DecisionRecordedEvent(
                action_id=action_id,
                decision=next_state.value,
                reason=reason,
            )
        )
        self._append_log(
            event_type="decision_recorded",
            action_id=action_id,
            state=next_state.value,
            reason=reason,
        )
        return next_record

    def mark_execution_result(self, action_id: str, executed: bool, detail: str) -> ActionRecord:
        record = self._require_record(action_id)
        if record.state != PermissionState.APPROVED:
            raise PermissionStateError(
                f"cannot record execution for action_id={action_id} from state={record.state.value}"
            )

        next_state = PermissionState.EXECUTED if executed else PermissionState.CANCELLED
        next_record = replace(record, state=next_state, execution_detail=detail)
        self._records[action_id] = next_record

        self._event_bus.publish(
            ExecutionResultEvent(
                action_id=action_id,
                tool=record.proposal.tool if record.proposal else None,
                executed=executed,
                detail=detail,
            )
        )
        self._append_log(
            event_type="execution_result",
            action_id=action_id,
            state=next_state.value,
            executed=executed,
            detail=detail,
        )
        return next_record

    def cancel(self, action_id: str, reason: str) -> ActionRecord:
        record = self._require_record(action_id)
        if record.state != PermissionState.APPROVED:
            raise PermissionStateError(
                f"cannot cancel action_id={action_id} from state={record.state.value}"
            )
        next_record = replace(record, state=PermissionState.CANCELLED, decision_reason=reason)
        self._records[action_id] = next_record
        self._event_bus.publish(
            DecisionRecordedEvent(action_id=action_id, decision="cancelled", reason=reason)
        )
        self._append_log(
            event_type="decision_recorded",
            action_id=action_id,
            state=PermissionState.CANCELLED.value,
            reason=reason,
        )
        return next_record

    def can_execute(self, action_id: str) -> bool:
        if self._dry_run:
            return False
        record = self._records.get(action_id)
        return record is not None and record.state == PermissionState.APPROVED

    def state_for(self, action_id: str) -> PermissionState | None:
        record = self._records.get(action_id)
        return record.state if record else None

    def get_record(self, action_id: str) -> ActionRecord:
        return self._require_record(action_id)

    def get_proposal(self, action_id: str) -> ToolProposal:
        record = self._require_record(action_id)
        if record.proposal is None:
            raise PermissionStateError(f"action_id={action_id} does not have a valid proposal")
        return record.proposal

    def _require_record(self, action_id: str) -> ActionRecord:
        record = self._records.get(action_id)
        if record is None:
            raise PermissionStateError(f"unknown action_id={action_id}")
        return record

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
