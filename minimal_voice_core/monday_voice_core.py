import asyncio
import base64
import math
import os
import re
import struct
import sys
import time
import webbrowser
from urllib.parse import quote_plus, urlparse

import cv2
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types

if sys.version_info < (3, 11, 0):
    import exceptiongroup  # type: ignore
    import taskgroup  # type: ignore

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

DEFAULT_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

OPEN_BROWSER_FUNCTION = {
    "name": "open_web_browser",
    "description": "Open a website in the user's browser. Use this for direct web opening requests.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "url": {
                "type": "STRING",
                "description": "Absolute URL to open, for example https://www.youtube.com.",
            },
            "query": {
                "type": "STRING",
                "description": "Optional search query. If url is missing, query is used to build a search URL.",
            },
            "site": {
                "type": "STRING",
                "description": "Optional site hint, such as youtube or google.",
            },
        },
    },
}


class MondayVoiceCore:
    """
    Minimal Monday-style core:
    - voice in / voice out (Gemini Native Audio)
    - typed text input
    - camera frames
    - JSON function-call style browser opener
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        camera_index: int = 0,
        enable_camera: bool = True,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env or constructor.")

        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.camera_index = camera_index
        self.camera_enabled = enable_camera

        self.client = genai.Client(http_options={"api_version": "v1beta"}, api_key=self.api_key)
        self.audio = pyaudio.PyAudio()

        self.config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription={},
            output_audio_transcription={},
            system_instruction=(
                "You are Monday. Keep responses concise and practical. You will mimic the language, tone, and character of F.R.I.D.A.Y. from the Iron Man movies."
                "When open_web_browser is available you may use it, but direct local command routing may execute it before you respond."
            ),
            tools=[{"function_declarations": [OPEN_BROWSER_FUNCTION]}],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        )

        self.session = None
        self.stop_event = asyncio.Event()
        self.out_queue: asyncio.Queue | None = None
        self.audio_in_queue: asyncio.Queue | None = None

        self._mic_stream = None
        self._speaker_stream = None
        self._camera = None

        self._latest_image_payload = None
        self._is_speaking = False
        self._silence_start_time = None
        self._last_input_transcription = ""
        self._last_output_transcription = ""
        self._last_voice_open_signature = ""
        self._last_voice_open_time = 0.0
        self._suppress_audio_until = 0.0

    def close(self) -> None:
        self.stop_event.set()
        if self._mic_stream is not None:
            try:
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
        if self._speaker_stream is not None:
            try:
                self._speaker_stream.close()
            except Exception:
                pass
            self._speaker_stream = None
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None
        try:
            self.audio.terminate()
        except Exception:
            pass

    @staticmethod
    def _extract_url(text: str) -> str | None:
        match = re.search(r"(https?://[^\s]+|www\.[^\s]+)", text, flags=re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _normalize_url(url: str) -> str:
        candidate = url.strip().rstrip(".,)")
        if "://" not in candidate:
            candidate = f"https://{candidate}"
        parsed = urlparse(candidate)
        if not parsed.netloc:
            return "https://www.google.com"
        return parsed.geturl()

    def _build_search_url(self, query: str, site: str | None = None) -> str:
        q = (query or "").strip()
        if site and "youtube" in site.lower():
            return f"https://www.youtube.com/results?search_query={quote_plus(q)}"
        return f"https://www.google.com/search?q={quote_plus(q)}"

    def parse_open_browser_command(self, text: str) -> dict | None:
        original = (text or "").strip()
        lowered = original.lower()
        if not original:
            return None

        if not any(trigger in lowered for trigger in ["open", "go to", "visit", "play", "launch"]):
            return None

        explicit_url = self._extract_url(original)
        if explicit_url:
            return {
                "name": "open_web_browser",
                "arguments": {"url": self._normalize_url(explicit_url)},
            }

        youtube_query = None
        if "youtube" in lowered:
            match = re.search(
                r"(?:youtube(?:\s+video)?(?:\s+about|\s+on)?\s+)(.+)$",
                lowered,
                flags=re.IGNORECASE,
            )
            if not match:
                match = re.search(
                    r"(?:open|play|find)\s+(?:a\s+)?(?:youtube\s+)?(?:video\s+)?(?:about|on)\s+(.+)$",
                    lowered,
                    flags=re.IGNORECASE,
                )
            if match:
                youtube_query = match.group(1).strip()

        if youtube_query:
            if len(youtube_query) < 4:
                return None
            return {
                "name": "open_web_browser",
                "arguments": {
                    "url": self._build_search_url(youtube_query, site="youtube"),
                    "query": youtube_query,
                    "site": "youtube",
                },
            }

        match = re.search(r"(?:open|go to|visit|launch)\s+(.+)$", original, flags=re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            if len(target) < 4:
                return None
            if "." in target and " " not in target:
                return {
                    "name": "open_web_browser",
                    "arguments": {"url": self._normalize_url(target)},
                }
            return {
                "name": "open_web_browser",
                "arguments": {"url": self._build_search_url(target), "query": target},
            }
        return None

    def execute_function_json(self, function_call: dict) -> dict:
        name = function_call.get("name")
        args = function_call.get("arguments", {})
        if name != "open_web_browser":
            return {"ok": False, "error": f"Unsupported function: {name}"}

        url = args.get("url")
        query = args.get("query")
        site = args.get("site")
        if not url and query:
            url = self._build_search_url(query, site=site)
        if not url:
            url = "https://www.google.com"

        normalized = self._normalize_url(url)
        webbrowser.open(normalized, new=2)
        return {"ok": True, "url": normalized}

    def _clear_audio_queue(self) -> None:
        if self.audio_in_queue is None:
            return
        try:
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
        except Exception:
            pass

    async def _send_realtime(self) -> None:
        while not self.stop_event.is_set():
            msg = await self.out_queue.get()
            await self.session.send(input=msg, end_of_turn=False)

    async def _listen_microphone(self) -> None:
        self._mic_stream = await asyncio.to_thread(
            self.audio.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        vad_threshold = 800
        silence_duration = 0.5

        while not self.stop_event.is_set():
            try:
                data = await asyncio.to_thread(self._mic_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

                sample_count = len(data) // 2
                if sample_count > 0:
                    shorts = struct.unpack(f"<{sample_count}h", data)
                    rms = int(math.sqrt(sum(s * s for s in shorts) / sample_count))
                else:
                    rms = 0

                if rms > vad_threshold:
                    self._silence_start_time = None
                    if not self._is_speaking:
                        self._is_speaking = True
                        if self._latest_image_payload and self.camera_enabled:
                            await self.out_queue.put(self._latest_image_payload)
                else:
                    if self._is_speaking:
                        if self._silence_start_time is None:
                            self._silence_start_time = time.time()
                        elif time.time() - self._silence_start_time > silence_duration:
                            self._is_speaking = False
                            self._silence_start_time = None
            except Exception as exc:
                print(f"[mic] {exc}")
                await asyncio.sleep(0.1)

    async def _play_audio(self) -> None:
        self._speaker_stream = await asyncio.to_thread(
            self.audio.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while not self.stop_event.is_set():
            bytestream = await self.audio_in_queue.get()
            if time.time() < self._suppress_audio_until:
                continue
            await asyncio.to_thread(self._speaker_stream.write, bytestream)

    def _open_camera(self):
        if self._camera is not None and self._camera.isOpened():
            return self._camera

        backends = [getattr(cv2, "CAP_AVFOUNDATION", None), getattr(cv2, "CAP_DSHOW", None), cv2.CAP_ANY]
        for backend in backends:
            if backend is None:
                continue
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap.isOpened():
                self._camera = cap
                return cap
            cap.release()
        raise RuntimeError("Could not open camera. Check OS permissions and index.")

    async def _camera_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.camera_enabled:
                await asyncio.sleep(0.25)
                continue
            try:
                cap = self._open_camera()
                payload = await asyncio.to_thread(self._read_camera_payload, cap)
                if payload:
                    self._latest_image_payload = payload
            except Exception as exc:
                print(f"[camera] {exc}")
            await asyncio.sleep(1.0)

    @staticmethod
    def _read_camera_payload(cap):
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(bytes(encoded)).decode("utf-8"),
        }

    async def _send_text_turn(self, text: str) -> None:
        if self.camera_enabled and self._latest_image_payload:
            await self.session.send(input=self._latest_image_payload, end_of_turn=False)
        await self.session.send(input=text, end_of_turn=True)

    async def _maybe_execute_voice_open_command(self, transcript: str) -> None:
        now = time.time()
        if now - self._last_voice_open_time < 4.0:
            return

        function_call = self.parse_open_browser_command(transcript)
        if not function_call:
            return

        signature = f"{function_call.get('name')}::{function_call.get('arguments', {}).get('url', '')}"
        if signature == self._last_voice_open_signature:
            return
        self._last_voice_open_signature = signature
        self._last_voice_open_time = now

        result = self.execute_function_json(function_call)
        if result.get("ok"):
            self._clear_audio_queue()
            self._suppress_audio_until = time.time() + 2.0
            print(f"\n[tool] open_web_browser -> {result['url']}")

    async def _receive(self) -> None:
        while not self.stop_event.is_set():
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    if time.time() >= self._suppress_audio_until:
                        self.audio_in_queue.put_nowait(data)

                if response.server_content:
                    in_tx = getattr(response.server_content.input_transcription, "text", None)
                    out_tx = getattr(response.server_content.output_transcription, "text", None)

                    if in_tx and in_tx != self._last_input_transcription:
                        self._last_input_transcription = in_tx
                        print(f"\nYou (voice): {in_tx}")
                        await self._maybe_execute_voice_open_command(in_tx)

                    if out_tx and out_tx != self._last_output_transcription:
                        self._last_output_transcription = out_tx
                        if time.time() >= self._suppress_audio_until:
                            print(f"\nMonday: {out_tx}")

                if response.tool_call:
                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        if fc.name != "open_web_browser":
                            continue
                        args = dict(fc.args) if fc.args else {}
                        result = self.execute_function_json({"name": "open_web_browser", "arguments": args})
                        function_responses.append(
                            types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                        )
                    if function_responses:
                        await self.session.send_tool_response(function_responses=function_responses)

    async def _text_loop(self) -> None:
        print("Text commands: /camera on | /camera off | /quit")
        while not self.stop_event.is_set():
            user_text = await asyncio.to_thread(input, "\nYou (text): ")
            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text == "/quit":
                self.stop_event.set()
                return
            if user_text == "/camera on":
                self.camera_enabled = True
                print("Camera enabled.")
                continue
            if user_text == "/camera off":
                self.camera_enabled = False
                print("Camera disabled.")
                continue

            function_call = self.parse_open_browser_command(user_text)
            if function_call:
                result = self.execute_function_json(function_call)
                if result.get("ok"):
                    print(f"[tool] open_web_browser -> {result['url']}")
                else:
                    print(f"[tool] failed: {result.get('error')}")
                continue

            await self._send_text_turn(user_text)

    async def run(self) -> None:
        print(f"Connecting with model: {self.model}")
        async with (
            self.client.aio.live.connect(model=self.model, config=self.config) as session,
            asyncio.TaskGroup() as tg,
        ):
            self.session = session
            self.out_queue = asyncio.Queue(maxsize=20)
            self.audio_in_queue = asyncio.Queue()

            tg.create_task(self._send_realtime())
            tg.create_task(self._listen_microphone())
            tg.create_task(self._play_audio())
            tg.create_task(self._receive())
            tg.create_task(self._camera_loop())
            tg.create_task(self._text_loop())

            await self.stop_event.wait()
