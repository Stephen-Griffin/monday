# AGENTS.md

## Purpose

This file is the default operating guide for work inside `voice_core/`.

The code in this directory is the active terminal-based Monday voice prototype.
Keep changes practical, testable, and aligned with the current implementation rather
than inventing extra layers.

## Current Surface

- `run.py` is the CLI entrypoint and argument parser.
- `monday_voice_core.py` contains nearly all runtime behavior.
- The app handles live microphone input, audio playback, typed text input, optional
  camera frames, and local browser opening.
- `tests/` contains a committed `pytest` suite for non-device behavior.
- `__init__.py` and `__main__.py` are thin package wrappers.

## Repository Map

- `README.md`: setup, runtime behavior, CLI flags, and validation commands.
- `run.py`: CLI parsing and startup/shutdown flow.
- `monday_voice_core.py`: Gemini Live session logic, device handling, browser command
  parsing, and text loop behavior.
- `tests/test_run.py`: CLI and package entrypoint coverage.
- `tests/test_monday_voice_core.py`: helper and side-effect boundary coverage.
- `tests/conftest.py`: dependency stubs for `google.genai`, `cv2`, `pyaudio`, and
  `dotenv`.
- `requirements.txt`: flat dependency list for local installs and dev tools.
- `.env.example`: environment variable template.
- `../pyproject.toml`: package metadata, console script wiring, and Ruff/Pytest/Mypy
  config shared with this directory.

## Current Tech Stack

- Python `3.10+`, with `3.11+` preferred
- `google-genai`
- `python-dotenv`
- `opencv-python`
- `pyaudio`
- `pytest`
- `ruff`

The root `pyproject.toml` also still lists `playwright` and `websockets`. Do not
document them as active runtime features unless the code in this directory actually
uses them.

## Working Rules

- Make the smallest change that fully solves the request.
- If the task affects runtime behavior, inspect both `run.py` and
  `monday_voice_core.py` before changing anything.
- Keep CLI flags, README instructions, and package metadata in sync.
- Keep environment variable names consistent with the implementation:
  `GEMINI_API_KEY` and optional `GEMINI_MODEL`.
- If you change dependencies, update both `requirements.txt` and the relevant setup
  docs, and check whether `../pyproject.toml` also needs to change.
- If you change user-visible behavior, update `README.md` in the same change unless
  the user explicitly limits scope.
- Prefer adding testable helpers over burying more logic in device-heavy code paths.

## Files And Paths To Avoid

Do not modify these unless the user explicitly asks:

- `.env`
- `.venv/`
- `__pycache__/`
- `tests/__pycache__/`
- `.DS_Store`

These are local or generated artifacts.

## Environment Expectations

- The documented setup copies `voice_core/.env.example` to `.env` at the repo root.
- A valid `GEMINI_API_KEY` is required for live runs.
- The interactive app may require OS permission for microphone and camera access.
- Browser-opening behavior uses Python's built-in `webbrowser` module and can launch
  a real browser on the host machine.

## Side Effects And Caution Areas

Treat `run.py` and `monday_voice_core.py` as side-effectful:

- Starting the app can access the microphone.
- Camera capture is enabled by default unless `--no-camera` is used.
- The app can open browser tabs or windows from typed input, voice transcription, or
  model tool calls.
- Live execution depends on external API credentials and local devices.

Prefer static checks and tests over manual runtime validation unless the user
explicitly asks for a live run.

## Safe Validation

Use the least invasive validation that matches the change:

- For documentation-only changes, verify paths, filenames, env var names, and
  commands manually.
- For Python edits, prefer:
  - `cd voice_core && python -m compileall . tests`
- For behavior that should stay covered by the committed suite, run:
  - `cd voice_core && python -m pytest tests`
- If `ruff` is available, run:
  - `cd voice_core && ruff format --check . tests`
- Only run the live app when the user explicitly wants manual validation:
  - `cd voice_core && python run.py`
  - or `monday-voice` from the repo root after install

## Testing Guidance

- The tests in `tests/` are designed to avoid real device and network access.
- `tests/conftest.py` stubs heavy dependencies so unit tests can import the package
  without a microphone, camera, or Gemini client.
- When adding behavior in helper methods, extend the pytest suite instead of relying
  on manual testing alone.
- If a change cannot be covered safely without real hardware or credentials, say so
  explicitly in your final notes.

## Editing Guidance

- Keep `run.py` thin. Put substantive runtime logic in `monday_voice_core.py` or in a
  narrowly scoped helper if the file needs relief.
- Preserve the current CLI flags unless the task requires changing them.
- Keep the text command list and README command tables aligned.
- When touching browser-opening behavior, review both local text handling and
  model-tool handling.
- When touching camera behavior, keep `--camera-index`, `--no-camera`, and runtime
  `/camera on` or `/camera off` behavior consistent.
- If packaging or console entrypoints change, update `../pyproject.toml`,
  `README.md`, and tests together.

## What Not To Do

- Do not run the interactive app as a default smoke test.
- Do not commit secrets or local environment files.
- Do not add speculative framework code around the current prototype unless the task
  truly requires it.
- Do not run destructive git commands unless the user explicitly requests them.
