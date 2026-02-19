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

## Agent 3 Notes (UI/UX)

### UI event contract
- UI consumes:
  - `TranscriptEvent`: appends transcript lines with local timestamp and speaker label.
  - `ToolProposalEvent`: renders action proposal cards with tool, reason, and pretty JSON args.
  - `StatusEvent`: updates status line and runtime indicators (`connected`, `listening`, `speaking`, `streaming`).
  - `NotesUpdateEvent`: applies notes updates in the local notes pane (`mode=append|replace`) and shows a brief banner.
- UI emits:
  - `ActionDecisionEvent(action_id: str, decision: str)` where `decision` is `"approved"` or `"rejected"`.

### Notes for integration
- Action proposal card Approve/Reject buttons only publish decision events. UI does not execute actions.
- Streaming indicator is derived from listening state (`ON` while listening, else `OFF`).
- Speaking indicator is inferred from assistant transcript activity and decays back to `OFF` after a short timer.
