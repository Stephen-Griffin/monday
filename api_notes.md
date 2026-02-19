# api_notes.md

## OpenAI usage notes (MVP)

### Model
Use the Realtime **mini** model for both voice + tool proposals:
- `MODEL=gpt-realtime-mini` (name may vary; keep configurable)

### Sessions
- Connect with WebSocket
- Send audio input chunks
- Receive audio output chunks + transcript events
- Receive tool proposals (structured JSON) to show in UI

### Cost control
- Wake word / PTT locally so we only stream when engaged
- Do not stream video continuously
- Keep system prompt short
- Consider "snapshot-only" camera mode

### Local privacy stance
- Never stream audio until wake word/ptt activates
- Provide a visible indicator "Streaming to cloud: ON/OFF"
- Keep logs opt-in (or redact by default)

