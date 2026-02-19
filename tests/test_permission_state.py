import tempfile
import unittest

from app.events import DecisionRecordedEvent, EventBus, ExecutionResultEvent, ProposalCreatedEvent
from app.permissions.state import PermissionState, PermissionStateError, PermissionStateMachine
from app.tools.contracts import ToolProposal


class PermissionStateMachineTests(unittest.TestCase):
    def test_proposed_to_approved_to_executed(self) -> None:
        bus = EventBus()
        observed = []
        bus.subscribe_all(observed.append)

        with tempfile.TemporaryDirectory() as tmp_dir:
            machine = PermissionStateMachine(
                event_bus=bus,
                actions_log_path=f"{tmp_dir}/actions.jsonl",
                id_factory=_id_factory(["action-1"]),
            )
            proposal = ToolProposal(
                tool="open_url",
                args={"url": "https://youtube.com/watch?v=123"},
                reason="User asked to open a video.",
            )

            record = machine.create_proposal(proposal)
            self.assertEqual(record.action_id, "action-1")
            self.assertEqual(machine.state_for("action-1"), PermissionState.PROPOSED)
            self.assertIsInstance(observed[0], ProposalCreatedEvent)

            machine.record_decision("action-1", approved=True)
            self.assertEqual(machine.state_for("action-1"), PermissionState.APPROVED)
            self.assertTrue(machine.can_execute("action-1"))
            self.assertIsInstance(observed[1], DecisionRecordedEvent)

            machine.mark_execution_result("action-1", executed=True, detail="ok")
            self.assertEqual(machine.state_for("action-1"), PermissionState.EXECUTED)
            self.assertIsInstance(observed[2], ExecutionResultEvent)

    def test_rejected_proposal_cannot_execute(self) -> None:
        bus = EventBus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            machine = PermissionStateMachine(
                event_bus=bus,
                actions_log_path=f"{tmp_dir}/actions.jsonl",
                id_factory=_id_factory(["action-2"]),
            )
            proposal = ToolProposal(
                tool="write_notes",
                args={"text": "hello", "mode": "append"},
                reason="User asked to save notes.",
            )
            machine.create_proposal(proposal)
            machine.record_decision("action-2", approved=False, reason="user rejected")
            self.assertEqual(machine.state_for("action-2"), PermissionState.REJECTED)

            with self.assertRaises(PermissionStateError):
                machine.mark_execution_result("action-2", executed=True, detail="should fail")

    def test_invalid_proposal_emits_rejection_event(self) -> None:
        bus = EventBus()
        observed: list[DecisionRecordedEvent] = []
        bus.subscribe(DecisionRecordedEvent, observed.append)

        with tempfile.TemporaryDirectory() as tmp_dir:
            machine = PermissionStateMachine(
                event_bus=bus,
                actions_log_path=f"{tmp_dir}/actions.jsonl",
                id_factory=_id_factory(["action-3"]),
            )
            record = machine.reject_invalid_proposal(
                reason="invalid payload: missing reason",
                raw_payload={"tool": "open_url", "args": {"url": "https://youtube.com"}},
            )

            self.assertEqual(record.action_id, "action-3")
            self.assertEqual(record.state, PermissionState.REJECTED)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0].action_id, "action-3")
            self.assertEqual(observed[0].decision, "rejected")


def _id_factory(ids: list[str]):
    iterator = iter(ids)
    return lambda: next(iterator)


if __name__ == "__main__":
    unittest.main()
