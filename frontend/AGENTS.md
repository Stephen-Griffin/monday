# AGENTS.md

## Purpose

This file is the default operating guide for work inside `frontend/`.

The frontend is a small Next.js App Router project. Keep changes narrow, match the
existing structure, and avoid inventing extra architecture unless the task clearly
calls for it.

## Current Surface

- The app is currently a minimal single-page prototype in `app/`.
- `app/page.tsx` contains the main UI.
- `app/layout.tsx` defines global metadata and font setup.
- `app/globals.css` holds the shared styling entrypoint.
- `public/` contains static assets, mostly starter SVGs.
- There is no app-specific test suite committed in this directory today.

## Repository Map

- `README.md`: local frontend setup notes and commands.
- `package.json`: npm scripts and dependency versions.
- `app/layout.tsx`: root layout, metadata, and `next/font` wiring.
- `app/page.tsx`: current page-level UI.
- `app/globals.css`: Tailwind import and global CSS variables.
- `next.config.ts`: Next.js config.
- `tsconfig.json`: strict TypeScript config with the `@/*` path alias.
- `eslint.config.mjs`: canonical ESLint config used by `npm run lint`.
- `postcss.config.mjs`: Tailwind/PostCSS integration.

## Current Tech Stack

- Next.js `16.1.6`
- React `19.2.3`
- TypeScript in strict mode
- Tailwind CSS `4` through `@tailwindcss/postcss`
- ESLint `9`
- Prettier `3`

## Working Rules

- Make the smallest change that fully solves the request.
- Prefer extending the existing App Router files before adding new folders or
  abstractions.
- Keep `package.json`, config files, and docs in sync when scripts or tooling change.
- Treat `eslint.config.mjs` as the active ESLint config.
- Do not edit `eslint.config 2.mjs` unless the task is explicitly cleaning up that
  duplicate file.
- Keep root-level product framing in `../README.md`; keep frontend runtime details in
  `frontend/README.md`.
- Use ASCII unless a file already requires another character set.

## Files And Paths To Avoid

Do not modify these unless the user explicitly asks:

- `node_modules/`
- `.next/`
- `next-env.d.ts`
- `package-lock.json` for style-only changes
- `.env*`
- `.DS_Store`

These are generated, machine-local, or lockfile artifacts that should only change
when the task genuinely requires it.

## Safe Validation

Use the least invasive validation that matches the change:

- For documentation-only changes, verify paths, script names, and filenames manually.
- For source or config edits, run:
  - `cd frontend && npm run lint`
  - `cd frontend && npm run format:check`
- If the change could affect bundling or routing, optionally run:
  - `cd frontend && npm run build`
- Only run the dev server when the user explicitly wants a manual browser check:
  - `cd frontend && npm run dev`

## Editing Guidance

- Preserve the App Router structure unless the task requires a broader reorganization.
- Keep `app/layout.tsx` metadata aligned with any branding or content changes you make.
- Keep `app/globals.css` and component markup aligned when changing typography,
  spacing, or color tokens.
- If you add reusable UI, prefer a small local component structure over a speculative
  design system.
- If you replace starter assets in `public/`, remove or update unused defaults in the
  same change.
- If startup, lint, or formatting commands change, update `frontend/README.md` in the
  same change.

## What Not To Do

- Do not add a test framework, storybook, or state library unless the task requires it.
- Do not rewrite the whole UI just because the current page is scaffolded.
- Do not treat generated files as source of truth.
- Do not run destructive git commands unless the user explicitly requests them.
