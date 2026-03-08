# AGENTS.MD

## Purpose

This file is the root operating guide for cross-cutting work in this repository.

Use it when the task touches the repo root, spans multiple product surfaces, or
changes shared project metadata. For implementation details inside a specific
surface, follow the closer guide first:

- `frontend/AGENTS.md`
- `voice_core/AGENTS.md`

## Scope At The Root

The root of this repo is for coordination, not product-specific runtime logic.
Root work includes:

- `README.md`
- `pyproject.toml`
- `.github/`
- repo-wide conventions, docs, and workflow wiring

Do not use this file as the primary guide for editing `frontend/` or `voice_core/`
internals. Those directories now have their own instructions.

## Repository Map

- `README.md`: high-level product overview for the Monday project.
- `pyproject.toml`: Python package metadata, shared tool configuration, and the
  `monday-voice` console script definition.
- `.github/workflows/eslint.yml`: frontend ESLint CI.
- `.github/workflows/frontend-prettier.yml`: frontend Prettier CI.
- `.github/workflows/pytest.yml`: Python test CI for `voice_core/tests`.
- `.github/workflows/ruff.yml`: repo Ruff formatter check.
- `frontend/`: Next.js frontend surface with its own local guide.
- `voice_core/`: terminal Python voice prototype with its own local guide.

## Current Repo State

- The repo now has two active product surfaces: `frontend/` and `voice_core/`.
- The root README is intentionally high level and should stay that way.
- Python packaging and tool config are centralized in `pyproject.toml`.
- CI currently runs frontend ESLint, frontend Prettier check, Python pytest, and
  Ruff format checks.
- There is no root-level app runner, monorepo task runner, or shared component
  package today.

## Working Rules

- Make the smallest change that fully solves the request.
- Keep documentation accurate to the current repo state. Do not describe planned
  structure as if it already exists.
- Route implementation-specific guidance to the nearest directory guide instead of
  duplicating it here.
- When a change spans root and a product surface, update both the root file and the
  relevant surface docs in the same change.
- Preserve the split between high-level docs at the root and runtime/setup details in
  subdirectory READMEs.
- Use ASCII unless a file already requires another character set.
- Do not commit secrets, tokens, or local machine state.

## Files And Paths To Avoid

Do not modify these unless the task explicitly requires it:

- `.venv/`
- `.pytest_cache/`
- `.ruff_cache/`
- `frontend/.next/`
- `frontend/node_modules/`
- `voice_core/.venv/`
- `voice_core/__pycache__/`
- `voice_core/tests/__pycache__/`
- `.env`
- `.DS_Store`

These are generated, local, or environment-specific artifacts.

## Safe Validation

Use only the checks that match the files you changed:

- For root documentation changes, verify paths, filenames, and commands manually.
- For `pyproject.toml` or Python workflow changes, use the relevant `voice_core`
  checks from `voice_core/AGENTS.md`.
- For frontend workflow or root docs that reference frontend commands, use the
  relevant checks from `frontend/AGENTS.md`.
- For CI workflow edits, inspect the YAML carefully and make sure referenced paths,
  working directories, and commands still exist.
- Do not launch interactive apps as a default validation step.

## Coordination Guidance

- If the task only affects `frontend/`, work there and follow `frontend/AGENTS.md`.
- If the task only affects `voice_core/`, work there and follow
  `voice_core/AGENTS.md`.
- If the task affects both surfaces, keep changes clearly separated and update docs
  for each affected area.
- If you change shared metadata in `pyproject.toml`, confirm the change still matches
  the current Python package layout in `voice_core/`.
- If you change CI, keep workflow names, directories, and dependency-install steps
  aligned with the actual repo layout.

## Pull Request Expectations

When preparing a PR:

- Keep the branch focused on one task.
- Summarize what changed and why.
- Call out what you verified locally.
- State clearly when live runtime validation was not done because it would require
  devices, credentials, or a browser session.
- Keep commits atomic.
- Update documentation in the same commit as the behavior or workflow change it
  describes.
- Use commit messages in the form `CHORE: xyz`, `FEATURE: xyz`, or `BUG: xyz`.

## What Not To Do

- Do not run destructive git commands unless the user explicitly requests them.
- Do not rewrite broad sections of documentation for style alone.
- Do not add speculative root tooling, folders, or monorepo layers just because they
  might be useful later.
- Do not let root docs become a duplicate of the subdirectory guides.
