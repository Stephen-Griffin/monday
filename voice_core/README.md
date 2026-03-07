# Monday Voice Core

`Monday Voice Core` is a terminal-based Gemini Live client that combines:

- live microphone input
- spoken audio responses
- typed text input in the same session
- optional camera snapshots for multimodal turns
- local browser launching for browser-intent requests and YouTube-style searches

The installable CLI is driven by `run.py`, and the core behavior lives in
`monday_voice_core.py`.

## What The App Actually Does

### Core Session Behavior

- Connects to the Gemini Live API using `google-genai` with `api_version="v1beta"`.
- Streams microphone audio to the model at `16 kHz`.
- Plays model audio responses back locally at `24 kHz`.
- Prints live input and output transcriptions in the terminal while the conversation runs.
- Uses the `Aoede` prebuilt voice for spoken responses.
- Applies a system instruction that makes the assistant respond as "Monday" with a witty, practical, Cortana-like tone.

### Camera Behavior

- Camera support is enabled by default.
- The app continuously captures a JPEG frame from the selected camera index about once per second.
- The latest frame is attached automatically:
  - when speech starts after silence
  - before a typed text turn is sent
- You can disable camera capture at startup with `--no-camera` or during runtime with `/camera off`.

### Browser Opening Behavior

The app can open pages locally without sending the request to the model first.

- Text input with browser-intent phrases is parsed locally before it is sent to the model.
- Explicit URLs like `https://example.com` or `www.example.com` open directly.
- Requests mentioning YouTube are converted into a YouTube search URL.
- Other open-style requests become a Google search URL.
- Voice transcriptions are also checked for the same open-style commands and can trigger the browser locally.
- Voice-triggered browser opens are rate-limited and deduplicated to reduce repeated launches.

The model is also given an `open_web_browser` tool declaration, so model-issued tool calls for browser opening are executed locally in the same way.

## Project Files

| File | Purpose |
| --- | --- |
| `../pyproject.toml` | Project metadata, dependencies, console script, and tool configuration |
| `run.py` | CLI entrypoint, argument parsing, and startup/shutdown flow |
| `monday_voice_core.py` | Realtime audio, camera, text loop, browser command parsing, and Gemini Live session management |
| `requirements.txt` | Flat dependency list for the current prototype and local development tools |
| `.env.example` | Environment variable template |

## Requirements

Before running the app, make sure you have:

- Python `3.10+` supported, `3.11+` recommended
- a valid `GEMINI_API_KEY`
- a working microphone
- speakers or headphones
- a webcam if you want camera support
- OS permission granted for microphone access, and camera access if enabled

`monday_voice_core.py` contains a compatibility shim for Python versions earlier than
`3.11`, and the root `pyproject.toml` installs the required backport packages
automatically. In practice, `Python 3.11+` is still the cleanest setup.

## Step-By-Step Setup

### 1. Change Into The Project Directory

```bash
cd /path/to/monday
```

### 2. Create A Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate The Virtual Environment

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install -e .
```

If you prefer a direct requirements-file install for local development tools, you can
also use:

```bash
python -m pip install -r voice_core/requirements.txt
```

### 5. Create Your Environment File

```bash
cp voice_core/.env.example .env
```

### 6. Add Your Gemini API Key

Open `.env` and set:

```dotenv
GEMINI_API_KEY=your_real_api_key
```

Optional template:

```dotenv
GEMINI_API_KEY=your_real_api_key
# Optional override:
# GEMINI_MODEL=gemini-2.5-flash-native-audio-preview
```

### 7. Grant Device Permissions

When your OS prompts for permissions:

- allow terminal access to the microphone
- allow terminal access to the camera if you plan to use it

### 8. Start The App

```bash
monday-voice
```

Once the session connects, you can speak naturally or type into the terminal prompt.

## Run Commands

### Default Startup

```bash
monday-voice
```

### Start Without Camera

```bash
monday-voice --no-camera
```

### Use A Different Camera Index

```bash
monday-voice --camera-index 0
```

### Override The Model Explicitly

```bash
monday-voice --model models/gemini-2.5-flash-native-audio-preview-12-2025
```

## Tests

The repository now includes a `pytest` suite for the Python backend in
`voice_core/tests/`. Those tests cover the CLI entrypoints and the non-device
helper behavior in `monday_voice_core.py` without launching the live app,
opening the microphone, or requiring camera access.

### Run The Test Suite

Run the tests from inside `voice_core`:

```bash
cd voice_core
python -m pytest tests
```

### Optional Local Validation

If you want to match the lightweight checks used during development:

```bash
cd voice_core
python -m compileall . tests
ruff format --check . tests
python -m pytest tests
```

## CLI Options

`monday-voice` exposes these arguments:

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `models/gemini-2.5-flash-native-audio-preview-12-2025` | Gemini Live model to connect to |
| `--camera-index` | `1` | Camera device index passed to OpenCV |
| `--no-camera` | off | Disables camera capture completely |

## In-App Commands

### Working Commands

| Command | Effect |
| --- | --- |
| `/camera on` | Re-enables camera capture during the session |
| `/camera off` | Disables camera capture during the session |
| `/quit` | Stops the app and exits |

### Natural-Language Browser Commands

These are typed as normal text, not slash commands:

```text
open youtube video on serpentine belts
visit openai.com
go to github.com
launch a search for python asyncio taskgroup
```

These requests are handled locally by the app's browser parser.


## Runtime Flow

At startup, the app:

1. loads environment variables from `.env`
2. reads the Gemini API key
3. creates a Gemini client
4. opens a live session with audio responses enabled
5. starts concurrent tasks for:
   - sending realtime microphone data
   - listening to the microphone
   - playing model audio
   - receiving model responses
   - polling the camera
   - handling terminal text input

The session runs until you enter `/quit` or terminate the process.

## Notes

- `GEMINI_MODEL` is supported by `MondayVoiceCore`, but `run.py` always passes its own default `--model` value unless you explicitly override it on the command line.
- `playwright` and `websockets` are still listed in the project dependencies, but they are not imported by the current entrypoint code path in this trimmed voice-core version.
- `python run.py` still works if you `cd voice_core` first, but the package-installed CLI is now the primary path.
- Browser opening uses Python's built-in `webbrowser` module, so the exact browser launched depends on your local system defaults.
- GitHub Actions now runs the `pytest` suite on every push and pull request alongside the existing Ruff formatter check.
- `requirements.txt` now includes both the current runtime dependencies and the local test/dev tools used in this repo.

## Quick Start

If you want the shortest path:

```bash
cd /path/to/monday
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp voice_core/.env.example .env
# edit .env and set GEMINI_API_KEY
monday-voice
```
