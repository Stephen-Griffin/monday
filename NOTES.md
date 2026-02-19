# Agent 1 Notes (System Architect / Integrator)

## What this branch provides
- Runnable project skeleton under `app/` with module boundaries from `agents.md`.
- Minimal PySide6 window launched by `app/main.py` with:
  - transcript pane
  - actions queue pane
  - notes editor pane
  - basic connect/listen controls and status line
- Config loading from `.env` (`OPENAI_API_KEY`, `MODEL`, `ALLOWLIST_DOMAINS`).
- Logging bootstrap with `logs/` directory creation and `logs/app.log`, `logs/actions.jsonl`.
- Shared event bus + event message types for cross-module integration.
- Realtime/audio/vision modules are stubs with explicit TODOs.

## Run
1. Create and activate a Python 3.11+ virtualenv.
2. Install deps:
```bash
pip install -r requirements.txt
```
3. Create env file:
```bash
cp .env.example .env
```
4. Start app:
```bash
python -m app.main --dry-run
```

## Test
```bash
python -m unittest tests/test_event_bus.py
```

## Shared interfaces for other agents
- `app/config.py`
  - `AppConfig`
  - `load_config(env_path: Path | None = None) -> AppConfig`
- `app/events.py`
  - `AudioFrameEvent`
  - `TranscriptEvent`
  - `ToolProposalEvent`
  - `StatusEvent`
  - `EventBus.subscribe(...)`, `EventBus.subscribe_all(...)`, `EventBus.publish(...)`
- `app/realtime/client.py`
  - `RealtimeClient.connect()`, `disconnect()`, `send_audio_frame(...)`
- `app/audio/engine.py`
  - `AudioEngine.start_listening()`, `stop_listening()`

## Agent 4 Notes (Tools + Safety)

### New event types emitted to UI via `EventBus`
- `ProposalCreatedEvent` (`event_type="proposal_created"`)
  - Fields: `action_id`, `tool`, `args`, `reason`
  - Emitted by `PermissionStateMachine.create_proposal(...)` after schema validation succeeds.
- `DecisionRecordedEvent` (`event_type="decision_recorded"`)
  - Fields: `action_id`, `decision` (`approved` | `rejected` | `cancelled`), `reason`
  - Emitted by:
    - `PermissionStateMachine.record_decision(...)`
    - `PermissionStateMachine.cancel(...)`
    - `PermissionStateMachine.reject_invalid_proposal(...)` (invalid model payloads)
- `ExecutionResultEvent` (`event_type="execution_result"`)
  - Fields: `action_id`, `tool`, `executed`, `detail`
  - Emitted by `PermissionStateMachine.mark_execution_result(...)`.
- `NotesUpdateEvent` (`event_type="notes_update"`)
  - Fields: `action_id`, `mode` (`append` | `replace`), `text`
  - Emitted by `ActionExecutor` only when `write_notes` is approved and executed.

### UI approval flow contract
1. Realtime/model payload arrives.
2. Call `parse_tool_proposal(raw_payload, allowlist_domains=...)`.
3. If validation fails:
   - Call `PermissionStateMachine.reject_invalid_proposal(reason=..., raw_payload=...)`.
   - This emits `decision_recorded` with `decision="rejected"` and logs the rejection.
4. If validation succeeds:
   - Call `PermissionStateMachine.create_proposal(proposal)`.
   - UI receives `proposal_created` and renders Approve/Reject controls keyed by `action_id`.
5. When user clicks:
   - Approve: `PermissionStateMachine.record_decision(action_id, approved=True, reason=<optional>)`
   - Reject: `PermissionStateMachine.record_decision(action_id, approved=False, reason=<optional>)`
6. After approval, trigger execution:
   - `ActionExecutor.execute(action_id)`
   - Executor refuses to run if state is not approved.

### Logging behavior
- All proposal/decision/execution transitions are written to `logs/actions.jsonl`.
- Invalid payload rejections are logged with `event_type="proposal_rejected_invalid"`.
- Executor also logs execution outcomes and blocked execution attempts.
