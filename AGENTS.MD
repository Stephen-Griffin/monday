# AGENTS.MD

## Purpose

This repository currently has one active product surface: the terminal-based voice prototype in `voice_core/`.

Use this file as the default operating guide for any agent working in this repo. Keep changes narrow, practical, and aligned with the current codebase rather than inventing structure that does not exist yet.

## Repository Map

- `README.md`: top-level project overview for the broader Monday vision. Keep this file high level.
- `voice_core/README.md`: setup and usage guide for the current working prototype. Put runtime details here, not in the root README.
- `voice_core/run.py`: CLI entrypoint and argument parsing.
- `voice_core/monday_voice_core.py`: core application logic for Gemini Live, microphone streaming, audio playback, camera capture, text input, and local browser actions.
- `voice_core/requirements.txt`: Python dependency list for the current prototype.

## Current Tech Stack

- Python is the only active implementation in this repository today.
- The app depends on `google-genai`, `python-dotenv`, `opencv-python`, and `pyaudio`.
- The current CI workflow only checks `ruff format --check`.
- There are no committed automated tests at the time this file was written.

## Working Rules

- Make the smallest change that fully solves the request.
- Prefer updating existing files over creating new structure unless the task clearly requires it.
- Keep documentation accurate to the current repo state. Do not describe planned architecture as if it already exists.
- Preserve the split between product overview docs at the root and implementation details inside `voice_core/`.
- Use ASCII unless a file already requires another character set.
- Do not commit secrets, tokens, or local machine state.

## Files And Paths To Avoid

Do not modify these unless the user explicitly asks:

- `voice_core/.env`
- `voice_core/.venv/`
- `voice_core/__pycache__/`
- `.DS_Store`

These are local or generated artifacts and should not be part of normal changes.

## Environment Expectations

- Prefer Python `3.11+`.
- The interactive app requires a valid `GEMINI_API_KEY` in `voice_core/.env`.
- The app may require local OS permission for microphone and camera access.
- Browser-opening behavior uses Python's `webbrowser` module and can launch real browser windows on the host machine.

## Safe Validation

Use the least invasive validation that matches the change:

- For documentation-only changes, verify paths, commands, and filenames manually.
- For Python edits, prefer a non-interactive syntax check first:
  - `cd voice_core && python -m compileall .`
- If `ruff` is available, run:
  - `ruff format --check voice_core`
- Only run the full app when the user explicitly asks for a live/manual test:
  - `cd voice_core && python run.py`

## Side Effects And Caution Areas

Treat `voice_core/run.py` and `voice_core/monday_voice_core.py` as side-effectful:

- Starting the app can access the microphone.
- Camera capture is enabled by default unless `--no-camera` is used.
- The app can open browser tabs or windows from typed input, voice transcription, or model tool calls.
- Running the full app without user intent is not a safe default for automated work.

When validating changes, prefer static checks unless the task specifically requires live device or browser behavior.

## Editing Guidance

- If the task affects runtime behavior, check both `voice_core/run.py` and `voice_core/monday_voice_core.py` before making assumptions.
- Keep CLI flags and README instructions in sync when changing startup behavior.
- Keep environment-variable names consistent with the implementation (`GEMINI_API_KEY`, optional `GEMINI_MODEL`).
- If you change dependencies, update `voice_core/requirements.txt` and the relevant setup docs together.
- If you change behavior that users will notice, update `voice_core/README.md` in the same change unless the user explicitly limits the scope.

## Pull Request Expectations

When preparing a PR:

- Keep the branch focused on one task.
- Summarize what changed and why.
- Call out any manual verification performed.
- Explicitly state when live runtime validation was not done because the app requires local devices, API credentials, or browser access.
- Keep commits atomic.
- Documentation should be updated with the changes that are made, in the same commit.
- Changes with different purposes should live within different commits (atomic).
- Commit message convention should be as follows: use the format "CHORE: xyz", "FEATURE: xyz", or "BUG: xyz".
    - where:
        - CHORE is a change that affects developers but not end users (such as adding a linter, or updating variable names).
        - FEATURE is a change that affects the end user (such as new functionality).
        - BUG is a fixed bug, plain and simple.

## What Not To Do

- Do not run destructive git commands unless the user explicitly requests them.
- Do not rewrite broad sections of documentation for style alone.
- Do not add speculative frameworks, folders, or tooling just because they may be useful later.
- Do not run the interactive voice app as a default smoke test.

