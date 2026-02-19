# agents.md

## Goal
Build an MVP "monday-lite" on macOS (MacBook Pro M1 2020) using:
- Built-in mic + speakers
- Built-in 720p camera (optional in MVP, but scaffold it)
- OpenAI **Realtime API** using **mini** model(s)
- A permissioned action layer that can:
  - write text to a local "Notes" pane (inside the app)
  - open a URL (e.g., YouTube video) **only after user approval**

The system should feel like: wake word → talk → assistant responds by voice → sometimes suggests a screen action → user approves → app performs it.

## Non-goals (MVP)
- No always-on continuous video streaming
- No OS-wide keystroke/mouse control
- No background daemons, launch agents, or Arduino hardware
- No long-term memory store beyond current session (can add later)

---

## Roles (multiple agents)
### Agent 1: System Architect / Integrator
- Owns overall structure, module boundaries, config, app wiring
- Ensures "permission gates" exist and are enforced
- Produces a working runnable skeleton quickly

### Agent 2: Realtime + Audio Engineer
- Implements mic capture, speaker playback
- Implements Realtime connection + streaming audio in/out
- Ensures push-to-talk + VAD mode work
- Handles reconnection/backoff

### Agent 3: UI / UX Engineer
- Builds simple local UI:
  - transcript pane
  - "actions" pane with Approve/Reject
  - a text editor pane ("Monday Notes")
- Connects action approvals to executor

### Agent 4: Tools + Safety Engineer
- Defines tool schema and validates JSON outputs
- Implements allowlists (URLs, app actions)
- Implements audit logs for actions and model outputs

---

## Tech choices (MVP)
- Language: **Python 3.11+**
- UI: **PySide6 (Qt)** (simple, native-feeling enough)
- Audio I/O: `sounddevice` + `numpy` (CoreAudio backend)
- Camera: OpenCV (`opencv-python`) or `AVFoundation` wrapper (OpenCV is fine for MVP)
- Networking: `websockets` (Realtime is WS-based), `asyncio`
- Config: `.env` via `python-dotenv`
- Logging: stdlib `logging` + JSONL action log file

> If PySide6 feels heavy, a fallback is a local web UI (FastAPI + WebSocket + React/Vite),
> but PySide6 is usually fastest for a single-machine MVP.

---

## Repository layout (suggested)
- `app/`
  - `main.py` (entry)
  - `ui/` (Qt UI)
  - `realtime/` (WS client)
  - `audio/` (capture/playback)
  - `vision/` (camera capture; optional in MVP)
  - `tools/` (action schema + validator)
  - `executor/` (open URL, write text)
  - `permissions/` (approval workflow)
  - `state/` (conversation state)
- `tests/`
- `agents.md`, `plan.md`, `schema.md`, `runbook.md`

---

## Definition of done (MVP)
1. Press "Start Listening" (or hold space for push-to-talk) after wake word is detected locally.
2. User speaks; audio is streamed to Realtime.
3. Assistant responds in natural audio + transcript.
4. Assistant may propose an action:
   - `open_url` (YouTube or any allowed URL)
   - `write_notes` (append/replace in Monday Notes pane)
5. UI shows action proposal with Approve/Reject.
6. Only on Approve does the executor perform it.
7. Everything is logged:
   - transcripts (optional)
   - proposed actions
   - approvals
   - executed actions

---

## Coding rules
- Always validate model tool outputs against schema (reject if invalid).
- Never execute actions without explicit user approval.
- Use allowlists for URLs (MVP: allow `youtube.com`, `youtu.be`).
- Keep audio streaming robust; degrade gracefully (text-only fallback).
- Provide a `--dry-run` mode where actions are never executed (but still shown).

---

## Testing expectations
- Unit tests for:
  - schema validation
  - allowlist logic
  - permission state machine
- Integration "smoke test":
  - connect to Realtime and exchange a short audio turn (can be mocked in CI)
- Manual test checklist in `runbook.md`.

