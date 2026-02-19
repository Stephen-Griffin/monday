# UI Manual Runbook (Agent 3)

## Prerequisites
1. Install dependencies and create `.env` as described in `NOTES.md`.
2. Launch app:
```bash
python -m app.main --dry-run
```

## Quick UI checklist
1. Verify status indicators render at startup:
   - `Connected: OFF`
   - `Listening: OFF`
   - `Speaking: OFF`
   - `Streaming to cloud: OFF`
2. Click `Connect Realtime`.
   - Expect button text to switch to `Disconnect Realtime`.
   - Expect connected indicator to turn `ON`.
3. Click `Start Listening`.
   - Expect listening and streaming indicators to turn `ON`.
   - Click again and confirm both return `OFF`.
4. Hold and release `Space` (push-to-talk).
   - On hold: listening/streaming should switch `ON`.
   - On release: listening/streaming should switch `OFF`.
5. Publish a `TranscriptEvent` from REPL/test harness.
   - Expect transcript line format: `[HH:MM:SS] Speaker: text`.
   - For assistant speaker, speaking indicator should briefly switch `ON`.
6. Publish a `ToolProposalEvent`.
   - Expect a new action card with tool, reason, pretty JSON args.
   - Click `Approve` or `Reject`; button pair should disable and decision text should update.
7. Publish a `NotesUpdateEvent` with `mode="append"`, then `mode="replace"`.
   - Expect notes editor to update accordingly.
   - Expect a brief banner indicating Jarvis updated notes.
8. Confirm no action side effects happen from UI approval itself.
   - UI must emit decisions only; it must not open URLs or write outside the local notes pane.

## Event payload assumptions used by UI
- `ActionDecisionEvent`:
  - `action_id`: proposal/action identifier from `ToolProposalEvent.proposal_id`
  - `decision`: `"approved"` or `"rejected"`
- `NotesUpdateEvent`:
  - `content`: text payload for notes update
  - `mode`: `"append"` (default) or `"replace"`
  - `source`: optional display label for notes update banner
