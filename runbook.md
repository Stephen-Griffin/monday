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
## Manual test checklist (Realtime audio MVP)

1. Create/activate Python 3.11+ virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create `.env`:
```bash
cp .env.example .env
```
4. Set at least:
```bash
OPENAI_API_KEY=your_key_here
MODEL=gpt-realtime-mini
```

## Smoke test (CLI)

Run:
```bash
python -m app.realtime.smoke_test --record-seconds 3 --response-wait-seconds 8
```

Expected:
- status lines show websocket connect/disconnect lifecycle
- mic records for 3 seconds
- transcript events print (partial/final), if returned by the model
- command exits cleanly without import/runtime errors

## End-to-end app test (UI)

1. Start app:
```bash
python -m app.main --dry-run
```
2. Click `Connect Realtime`.
3. Click `Start Listening`, speak a short prompt, click `Stop Listening`.
4. Confirm:
- status line changes across `connected`, `listening`, `idle`
- transcript pane receives model transcript deltas/finals
- assistant audio is audible from laptop speakers

## Troubleshooting

- If connection is blocked, check `OPENAI_API_KEY` in `.env`.
- If no audio is heard, verify macOS microphone/speaker permissions and selected default devices.
- If transcripts are empty, rerun smoke test and inspect printed status/error events.
