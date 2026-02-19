# plan.md

## Overview
We are building a laptop MVP that mirrors the eventual always-on garage assistant architecture:
- Local wake word gate (no cloud until activated)
- Realtime audio conversation loop (mini model)
- Permissioned tool execution for screen actions
- Simple UI for visibility + approvals

## Milestones
### M0: Repo + skeleton (1 session)
- Create Python project + venv
- Add dependencies
- Basic Qt window with:
  - transcript area
  - action queue area with Approve/Reject
  - notes editor area
  - status bar (Disconnected/Connected/Listening)

### M1: Realtime audio loop (core)
- WebSocket connect to Realtime
- Stream mic PCM → model
- Receive audio → play to speakers
- Receive transcript events → display in UI
- Add push-to-talk button (space bar hold)
- Add fallback: if audio playback fails, show text transcript only

### M2: Wake word gate (local)
- Implement local "Hey Monday" detection:
  - MVP approach: use an offline wake word library (Porcupine/OpenWakeWord)
  - Alternative: push-to-talk only for MVP if wake word is too much friction
- Only start streaming audio to OpenAI after wake word triggers (or user holds PTT)

### M3: Tooling + permission layer
- Define JSON tool schema: `open_url`, `write_notes`
- Configure the session so the model proposes tool calls
- Validate tool calls; show proposals in UI
- Approval required to execute
- Executor:
  - `open_url` uses `webbrowser.open(url)`
  - `write_notes` edits notes pane (append/replace)

### M4: Optional camera scaffolding (no streaming by default)
- Add camera capture module
- Add "Send Snapshot" button that sends a single frame
- Do NOT stream video continuously in MVP

### M5: Hardening
- Reconnect logic
- Rate limiting / debounce action proposals
- Persistent logs and export

---

## Detailed tasks
### 1) Project setup
- `pyproject.toml` (or `requirements.txt`)
- `.env.example` includes:
  - `OPENAI_API_KEY=...`
  - `MODEL=gpt-realtime-mini`
  - `ALLOWLIST_DOMAINS=youtube.com,youtu.be`
- Add `Makefile` or `taskfile`:
  - `make run`
  - `make test`
  - `make lint`

### 2) UI
- Qt main window
- Components:
  - Transcript widget (read-only)
  - Notes editor (editable)
  - Action list (cards with approve/reject)
  - Buttons: Connect/Disconnect, PTT, Send Snapshot (optional)
- UX:
  - When assistant speaks, show "Speaking…" indicator
  - Show the exact action payload before approval

### 3) Realtime session
- Use asyncio websocket client
- Maintain a session object:
  - connected flag
  - current conversation id
  - audio in queue
  - audio out queue
- Events handled:
  - audio output chunks
  - transcription updates
  - tool call proposals (structured output)

### 4) Permissions + Executor
- State machine:
  - Proposed → Approved/Rejected → Executed/Cancelled
- Logging:
  - `logs/actions.jsonl`
  - include timestamp, payload, decision, result

---

## Acceptance criteria
- You can talk and get voice responses reliably.
- You can say “open that YouTube video” and the model proposes `open_url`.
- The app asks permission; only after clicking Approve does it open the URL.
- You can say “write a summary of what we decided” and it proposes `write_notes`.
- Notes update only after approval.
- No silent execution. Ever.

---

## Stretch goals (after MVP)
- Add a small local memory store (last N facts) with “forget” controls
- Add “screen readback” (send screenshot on demand)
- Add multi-camera selection (future garage build)
- Add hotword + continuous conversation mode with safe timeout

