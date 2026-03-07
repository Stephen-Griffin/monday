import asyncio
import base64
import queue
import types

import pytest

from voice_core.monday_voice_core import MondayVoiceCore


def make_core() -> MondayVoiceCore:
    core = MondayVoiceCore.__new__(MondayVoiceCore)
    core.audio_in_queue = None
    core.camera_enabled = True
    core.session = None
    core._latest_image_payload = None
    core._monday_line_active = False
    core._current_output_transcription = ""
    core._last_output_transcription = ""
    core._last_voice_open_signature = ""
    core._last_voice_open_time = 0.0
    core._suppress_audio_until = 0.0
    core._camera = None
    core.camera_index = 0
    return core


# Verifies constructor state is populated from the explicit arguments and environment-derived defaults.
def test_init_uses_constructor_and_environment_defaults(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *, http_options, api_key):
            captured["client"] = {
                "http_options": http_options,
                "api_key": api_key,
            }

    class FakePyAudio:
        def __init__(self):
            captured["audio_created"] = True

    class Factory:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_types = types.SimpleNamespace(
        LiveConnectConfig=Factory,
        SpeechConfig=Factory,
        VoiceConfig=Factory,
        PrebuiltVoiceConfig=Factory,
    )

    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "models/from-env")
    monkeypatch.setattr("voice_core.monday_voice_core.load_dotenv", lambda: None)
    monkeypatch.setattr(
        "voice_core.monday_voice_core.genai", types.SimpleNamespace(Client=FakeClient)
    )
    monkeypatch.setattr("voice_core.monday_voice_core.types", fake_types)
    monkeypatch.setattr(
        "voice_core.monday_voice_core.pyaudio",
        types.SimpleNamespace(PyAudio=FakePyAudio, paInt16=16),
    )

    core = MondayVoiceCore(camera_index=7, enable_camera=False)

    assert core.api_key == "env-key"
    assert core.model == "models/from-env"
    assert core.camera_index == 7
    assert core.camera_enabled is False
    assert captured["client"]["api_key"] == "env-key"
    assert captured["audio_created"] is True


# Verifies the constructor raises a clear error when no Gemini API key is available.
def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("voice_core.monday_voice_core.load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="Missing GEMINI_API_KEY"):
        MondayVoiceCore(api_key=None)


# Verifies close sets the stop event and terminates the audio subsystem exactly once.
def test_close_sets_stop_event_and_terminates_audio():
    calls = []
    core = make_core()
    core.stop_event = asyncio.Event()
    core.audio = types.SimpleNamespace(terminate=lambda: calls.append("terminate"))

    core.close()

    assert core.stop_event.is_set()
    assert calls == ["terminate"]


# Verifies URL extraction finds explicit URLs and ignores strings without one.
def test_extract_url_handles_matching_and_missing_values():
    assert MondayVoiceCore._extract_url("visit https://example.com/docs now") == (
        "https://example.com/docs"
    )
    assert MondayVoiceCore._extract_url("nothing to parse here") is None


# Verifies URL normalization adds an https scheme and falls back to Google for invalid inputs.
def test_normalize_url_adds_scheme_and_handles_invalid_input():
    assert (
        MondayVoiceCore._normalize_url("example.com/path") == "https://example.com/path"
    )
    assert (
        MondayVoiceCore._normalize_url("http://example.com/test")
        == "http://example.com/test"
    )
    assert MondayVoiceCore._normalize_url("   not a url   ") == "https://not a url"
    assert (
        MondayVoiceCore._normalize_url("https:///missing-host")
        == "https://www.google.com"
    )


# Verifies search URL building routes YouTube requests to YouTube and all others to Google.
def test_build_search_url_uses_expected_provider():
    core = make_core()

    assert (
        core._build_search_url("oil filter", site="youtube")
        == "https://www.youtube.com/results?search_query=oil+filter"
    )
    assert (
        core._build_search_url("oil filter")
        == "https://www.google.com/search?q=oil+filter"
    )


# Verifies browser-intent parsing handles explicit URLs, YouTube searches, generic searches, and rejected short commands.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "open example.com",
            {"name": "open_web_browser", "arguments": {"url": "https://example.com"}},
        ),
        (
            "play youtube video on torque wrench review",
            {
                "name": "open_web_browser",
                "arguments": {
                    "url": "https://www.youtube.com/results?search_query=torque+wrench+review",
                    "query": "torque wrench review",
                    "site": "youtube",
                },
            },
        ),
        (
            "launch torque specs for honda civic",
            {
                "name": "open_web_browser",
                "arguments": {
                    "url": "https://www.google.com/search?q=torque+specs+for+honda+civic",
                    "query": "torque specs for honda civic",
                },
            },
        ),
        ("open abc", None),
        ("tell me a joke", None),
    ],
)
def test_parse_open_browser_command_returns_expected_payload(text, expected):
    core = make_core()

    assert core.parse_open_browser_command(text) == expected


# Verifies function execution rejects unsupported tools and normalizes browser URLs before opening them.
def test_execute_function_json_validates_name_and_opens_browser(monkeypatch):
    opened = []
    core = make_core()

    monkeypatch.setattr(
        "voice_core.monday_voice_core.webbrowser.open",
        lambda url, new: opened.append((url, new)),
    )

    assert core.execute_function_json({"name": "not_supported"}) == {
        "ok": False,
        "error": "Unsupported function: not_supported",
    }
    assert core.execute_function_json(
        {"name": "open_web_browser", "arguments": {"url": "example.com"}}
    ) == {"ok": True, "url": "https://example.com"}
    assert opened == [("https://example.com", 2)]


# Verifies function execution can derive a URL from a query or fall back to Google when no arguments are provided.
def test_execute_function_json_builds_url_from_query_or_default(monkeypatch):
    opened = []
    core = make_core()

    monkeypatch.setattr(
        "voice_core.monday_voice_core.webbrowser.open",
        lambda url, new: opened.append((url, new)),
    )

    assert core.execute_function_json(
        {
            "name": "open_web_browser",
            "arguments": {"query": "socket set", "site": "youtube"},
        }
    ) == {
        "ok": True,
        "url": "https://www.youtube.com/results?search_query=socket+set",
    }
    assert core.execute_function_json(
        {"name": "open_web_browser", "arguments": {}}
    ) == {"ok": True, "url": "https://www.google.com"}
    assert opened == [
        ("https://www.youtube.com/results?search_query=socket+set", 2),
        ("https://www.google.com", 2),
    ]


# Verifies the audio queue helper drains pending items and safely ignores a missing queue.
def test_clear_audio_queue_drains_items():
    core = make_core()
    core.audio_in_queue = queue.Queue()
    core.audio_in_queue.put("a")
    core.audio_in_queue.put("b")

    core._clear_audio_queue()

    assert core.audio_in_queue.empty() is True
    core.audio_in_queue = None
    core._clear_audio_queue()


# Verifies the output transcription helper replaces overlapping chunks and appends new trailing text.
def test_update_output_transcription_rebuilds_incremental_output():
    core = make_core()

    assert core._update_output_transcription("Hello") == "Hello"
    assert core._update_output_transcription("Hello there") == "Hello there"
    assert (
        core._update_output_transcription("general kenobi")
        == "Hello there general kenobi"
    )


# Verifies camera payload encoding returns a base64 JPEG payload for successful frames and None for failures.
def test_read_camera_payload_encodes_frames(monkeypatch):
    frame = object()
    encoded_bytes = b"jpeg-bytes"
    cap = types.SimpleNamespace(read=lambda: (True, frame))

    monkeypatch.setattr(
        "voice_core.monday_voice_core.cv2.imencode",
        lambda extension, data, params: (True, encoded_bytes),
    )

    payload = MondayVoiceCore._read_camera_payload(cap)

    assert payload == {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(encoded_bytes).decode("utf-8"),
    }

    failing_cap = types.SimpleNamespace(read=lambda: (False, None))
    assert MondayVoiceCore._read_camera_payload(failing_cap) is None


# Verifies text turns include the latest camera payload before the text when camera support is active.
def test_send_text_turn_sends_camera_payload_before_text():
    sends = []
    core = make_core()
    core.session = types.SimpleNamespace(
        send=lambda **kwargs: sends.append(kwargs) or asyncio.sleep(0)
    )
    core._latest_image_payload = {"mime_type": "image/jpeg", "data": "abc"}

    asyncio.run(core._send_text_turn("hello"))

    assert sends == [
        {"input": {"mime_type": "image/jpeg", "data": "abc"}, "end_of_turn": False},
        {"input": "hello", "end_of_turn": True},
    ]


# Verifies voice-triggered browser commands are rate limited, deduplicated, and suppress audio after a successful open.
def test_maybe_execute_voice_open_command_handles_cooldown_and_dedupe(monkeypatch):
    core = make_core()
    clear_calls = []
    exec_calls = []
    timestamps = iter([10.0, 10.0, 11.0, 15.0, 15.0, 15.0])

    monkeypatch.setattr(
        "voice_core.monday_voice_core.time.time", lambda: next(timestamps)
    )
    monkeypatch.setattr(
        core,
        "parse_open_browser_command",
        lambda text: (
            {"name": "open_web_browser", "arguments": {"url": "https://example.com"}}
            if "open" in text
            else None
        ),
    )
    monkeypatch.setattr(
        core,
        "execute_function_json",
        lambda payload: (
            exec_calls.append(payload) or {"ok": True, "url": "https://example.com"}
        ),
    )
    monkeypatch.setattr(
        core, "_clear_audio_queue", lambda: clear_calls.append("cleared")
    )

    asyncio.run(core._maybe_execute_voice_open_command("open example.com"))
    asyncio.run(core._maybe_execute_voice_open_command("open example.com"))
    asyncio.run(core._maybe_execute_voice_open_command("open example.com"))

    assert len(exec_calls) == 1
    assert clear_calls == ["cleared"]
    assert core._last_voice_open_signature == "open_web_browser::https://example.com"
    assert core._last_voice_open_time == 10.0
    assert core._suppress_audio_until == 12.0


# Verifies the camera opener reuses an existing camera, releases failed backends, and raises when no backend works.
def test_open_camera_reuses_existing_or_finds_working_backend(monkeypatch):
    released = []

    class FakeCamera:
        def __init__(self, backend, opened):
            self.backend = backend
            self._opened = opened

        def isOpened(self):
            return self._opened

        def release(self):
            released.append(self.backend)

    created = []

    def fake_video_capture(index, backend):
        created.append((index, backend))
        return FakeCamera(backend, opened=backend == 2)

    core = make_core()
    existing = FakeCamera("existing", opened=True)
    core._camera = existing

    assert core._open_camera() is existing

    core._camera = None
    monkeypatch.setattr(
        "voice_core.monday_voice_core.cv2.VideoCapture", fake_video_capture
    )

    opened = core._open_camera()

    assert opened.backend == 2
    assert released == [1]
    assert created[:2] == [(0, 1), (0, 2)]

    core._camera = None
    monkeypatch.setattr(
        "voice_core.monday_voice_core.cv2.VideoCapture",
        lambda index, backend: FakeCamera(backend, opened=False),
    )

    with pytest.raises(RuntimeError, match="Could not open camera"):
        core._open_camera()
