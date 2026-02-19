# runbook.md

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
