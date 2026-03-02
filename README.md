# Monday

**Monday** is a practical, multimodal assistant project built around a simple idea:

make it easier to work on real things in the real world.

This repository is the starting point for that vision. The goal is to grow Monday into a helpful sidekick for physical projects, especially the kind of work that happens in a garage, workshop, driveway, or home office.

Think:

- "Can you look at this part and tell me what I am holding?"
- "Read this label out loud and explain what it means."
- "Open the right reference page while I keep working."
- "Talk me through the next step while my hands are busy."

Today, the main working prototype lives in `voice_core/`. This root README is the high-level overview for people landing on the repository for the first time.

---

## The Big Idea

Monday is meant to combine a few natural ways of interacting so it feels less like "using software" and more like working with a capable assistant:

| Capability | What it means in plain English |
| --- | --- |
| Camera | Monday can use visual input to understand what is in front of you. |
| Text | You can type instructions, questions, or links when that is faster than speaking. |
| Audio | You can talk naturally and hear responses back, which is useful when your hands are occupied. |
| Browser | Monday can open pages, searches, and references so information is right there when you need it. |
| Front End | Over time, this will grow beyond a terminal into a cleaner, more approachable interface. |

The long-term goal is not just "AI chat," but a genuinely useful helper for hands-on tasks.

---

## Why This Project Exists

A lot of helpful work does not happen inside an IDE.

Sometimes you are:

- diagnosing a part
- assembling something
- comparing tools
- following a repair guide
- trying to keep your place in a multi-step task

In those moments, keyboard-first software is often awkward. Monday is being shaped around a more natural workflow:

1. Look at the thing.
2. Ask a question.
3. Get a useful answer.
4. Pull up the right reference.
5. Keep moving.

That makes it a strong fit for garage projects, repairs, prototyping, and other physical work where quick context matters.

---

## What Exists Right Now

The current implementation in this repository is the **voice core prototype**.

It already demonstrates the foundation of the larger idea:

- live microphone input
- spoken responses
- typed text input in the same session
- optional camera snapshots for visual context
- local browser launching for links and searches

Right now, that prototype runs in the terminal and lives in [`voice_core/`](./voice_core).

If you want the detailed setup instructions, runtime behavior, and command list, start here:

- [`voice_core/README.md`](./voice_core/README.md)

---

## Where Monday Is Headed

The broader project direction is to turn this into a more polished, practical assistant that can help with real-world tasks from end to end.

Some of the natural next layers are:

- a friendlier front end for everyday use
- better task flow for step-by-step project help
- stronger visual understanding for tools, parts, labels, and workspaces
- smoother browser and reference handling
- a more "always useful" assistant experience for physical builds and repairs

The core principle is simple: Monday should reduce friction while you are doing actual work.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `voice_core/` | Current working prototype for voice, text, camera context, and browser actions |
| `README.md` | This high-level repository overview |

As the project expands, more components can sit alongside `voice_core` without changing the purpose of this root README: explain the project clearly to first-time visitors.

---

## Getting Started

If you want to try the current prototype:

1. Open the [`voice_core/README.md`](./voice_core/README.md).
2. Follow the setup steps there.
3. Run the terminal app and test the voice core.

If you are just here to understand the project, this is the short version:

**Monday is an attempt to build a practical assistant that can see, listen, speak, read, and help while you work on real things.**

