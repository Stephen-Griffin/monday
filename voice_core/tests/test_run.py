import asyncio
import importlib
import runpy
import sys

import pytest

import voice_core.run as run_module


# Verifies the CLI parser returns the documented defaults when no flags are provided.
def test_parse_args_uses_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["monday-voice"])

    args = run_module.parse_args()

    assert args.model == "models/gemini-2.5-flash-native-audio-preview-12-2025"
    assert args.camera_index == 1
    assert args.no_camera is False


# Verifies the CLI parser accepts explicit model, camera index, and camera-disable flags.
def test_parse_args_accepts_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monday-voice",
            "--model",
            "models/custom",
            "--camera-index",
            "3",
            "--no-camera",
        ],
    )

    args = run_module.parse_args()

    assert args.model == "models/custom"
    assert args.camera_index == 3
    assert args.no_camera is True


# Verifies the async entrypoint builds MondayVoiceCore from parsed args, awaits run, and closes afterward.
def test_main_runs_app_and_closes(monkeypatch):
    events = []

    class FakeApp:
        def __init__(self, *, model, camera_index, enable_camera):
            events.append(("init", model, camera_index, enable_camera))

        async def run(self):
            events.append(("run",))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(
        run_module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "model": "models/test",
                "camera_index": 4,
                "no_camera": True,
            },
        )(),
    )
    monkeypatch.setattr(run_module, "MondayVoiceCore", FakeApp)

    asyncio.run(run_module.main())

    assert events == [
        ("init", "models/test", 4, False),
        ("run",),
        ("close",),
    ]


# Verifies the async entrypoint still closes the app when the run loop raises an exception.
def test_main_closes_app_on_error(monkeypatch):
    events = []

    class FakeApp:
        def __init__(self, *, model, camera_index, enable_camera):
            events.append(("init", model, camera_index, enable_camera))

        async def run(self):
            events.append(("run",))
            raise RuntimeError("boom")

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(
        run_module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "model": "models/test",
                "camera_index": 0,
                "no_camera": False,
            },
        )(),
    )
    monkeypatch.setattr(run_module, "MondayVoiceCore", FakeApp)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(run_module.main())

    assert events == [
        ("init", "models/test", 0, True),
        ("run",),
        ("close",),
    ]


# Verifies the synchronous CLI wrapper delegates to asyncio.run with the module's async main function.
def test_cli_uses_asyncio_run(monkeypatch):
    calls = []

    async def fake_main():
        return None

    def fake_asyncio_run(coro):
        calls.append(coro.cr_code.co_name)
        coro.close()

    monkeypatch.setattr(run_module, "main", fake_main)
    monkeypatch.setattr(run_module.asyncio, "run", fake_asyncio_run)

    run_module.cli()

    assert calls == ["fake_main"]


# Verifies importing the package re-exports MondayVoiceCore from the core module.
def test_package_re_exports_monday_voice_core():
    package = importlib.import_module("voice_core")
    core_module = importlib.import_module("voice_core.monday_voice_core")

    assert package.MondayVoiceCore is core_module.MondayVoiceCore


# Verifies running the package as __main__ triggers the CLI entrypoint.
def test_package_main_invokes_cli(monkeypatch):
    calls = []

    monkeypatch.setattr(run_module, "cli", lambda: calls.append("cli"))

    runpy.run_module("voice_core", run_name="__main__")

    assert calls == ["cli"]
