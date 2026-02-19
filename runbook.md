# runbook.md

## Manual safety test checklist (MVP)

### Setup
1. Launch app with `--dry-run` first.
2. Ensure `.env` has `ALLOWLIST_DOMAINS=youtube.com,youtu.be`.
3. Confirm `logs/actions.jsonl` exists.

### Tool validation tests
1. Valid `open_url` proposal:
```json
{"tool":"open_url","args":{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"},"reason":"User asked for the video"}
```
Expected:
- Proposal accepted.
- UI receives `proposal_created`.
- No browser launch until explicit approval.

2. Missing required field (`reason` absent):
```json
{"tool":"open_url","args":{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}
```
Expected:
- Proposal rejected as invalid.
- UI receives `decision_recorded` (`decision="rejected"`).
- `logs/actions.jsonl` has `event_type="proposal_rejected_invalid"`.

3. Unknown tool:
```json
{"tool":"run_shell","args":{"cmd":"open /Applications"},"reason":"Do it now"}
```
Expected:
- Rejected as invalid.
- No side effects.

4. Disallowed scheme (`file:`):
```json
{"tool":"open_url","args":{"url":"file:///etc/passwd"},"reason":"Open local file"}
```
Expected:
- Rejected as invalid.
- No side effects.

5. Disallowed scheme (`javascript:`):
```json
{"tool":"open_url","args":{"url":"javascript:alert(1)"},"reason":"Open link"}
```
Expected:
- Rejected as invalid.
- No side effects.

6. Disallowed domain:
```json
{"tool":"open_url","args":{"url":"https://example.com"},"reason":"User asked"}
```
Expected:
- Rejected because domain not in allowlist.
- No side effects.

7. Oversized `write_notes` text:
- Send `write_notes` with text longer than max validator limit.
Expected:
- Rejected as invalid.
- No notes update event.

### Permission gating tests
1. Proposal approved path:
- Submit valid proposal.
- Click Approve.
- Trigger execution.
Expected:
- State transition: Proposed -> Approved -> Executed (or Cancelled on failure).
- `execution_result` event emitted and logged.

2. Proposal rejected path:
- Submit valid proposal.
- Click Reject.
- Attempt execution anyway.
Expected:
- Execution blocked by state machine.
- No browser open / no notes update.

3. Dry-run behavior:
- Run with `--dry-run`.
- Approve valid proposal and execute.
Expected:
- No external side effects.
- Execution outcome logged as not executed/cancelled.

### Audit log checks
1. Verify each action line in `logs/actions.jsonl` includes:
- `timestamp`
- `event_type`
- `action_id`
2. Verify invalid proposals include raw payload + reason.
3. Verify approved executions include result details.
