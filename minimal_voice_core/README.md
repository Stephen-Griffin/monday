# Monday Minimal Voice Core

This export includes only:
- speaking/listening with Gemini Native Audio
- text input in the same session
- camera frame support
- browser opening via JSON function definition (`open_web_browser`)

## Key behavior
- Text commands like `open a youtube video on serpentine belts` are routed locally to a JSON function call and executed directly.
- Those text "open" commands are not sent to the model.
- Voice "open" commands are detected from live transcription and executed through the same JSON function call path.

## Files
- `monday_voice_core.py`: core logic
- `run.py`: startup script
- `requirements.txt`
- `.env.example`

## Setup
```bash
cd minimal_voice_core
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add GEMINI_API_KEY to .env
```

## Run
```bash
python run.py
```

Options:
- `--model gemini-2.5-flash-native-audio-preview`
- `--camera-index 0`
- `--no-camera`

In-app text commands:
- `/camera on`
- `/camera off`
- `/quit`
