from __future__ import annotations

from enum import Enum


class DecisionState(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class PermissionGate:
    """
    Tracks user decisions for action proposals.

    Executor calls must check `can_execute` before performing side effects.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run
        self._states: dict[str, DecisionState] = {}

    def register_proposal(self, proposal_id: str) -> None:
        self._states[proposal_id] = DecisionState.PROPOSED

    def approve(self, proposal_id: str) -> None:
        self._states[proposal_id] = DecisionState.APPROVED

    def reject(self, proposal_id: str) -> None:
        self._states[proposal_id] = DecisionState.REJECTED

    def mark_executed(self, proposal_id: str) -> None:
        self._states[proposal_id] = DecisionState.EXECUTED

    def state_for(self, proposal_id: str) -> DecisionState | None:
        return self._states.get(proposal_id)

    def can_execute(self, proposal_id: str) -> bool:
        if self._dry_run:
            return False
        return self._states.get(proposal_id) == DecisionState.APPROVED

