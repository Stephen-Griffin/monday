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

