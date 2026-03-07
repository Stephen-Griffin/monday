import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_google_genai_stub() -> None:
    google_module = sys.modules.setdefault("google", types.ModuleType("google"))

    genai_module = types.ModuleType("google.genai")
    genai_module.Client = object

    types_module = types.ModuleType("google.genai.types")

    class _Factory:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    for name in [
        "FunctionResponse",
        "LiveConnectConfig",
        "PrebuiltVoiceConfig",
        "SpeechConfig",
        "VoiceConfig",
    ]:
        setattr(types_module, name, _Factory)

    genai_module.types = types_module
    google_module.genai = genai_module

    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.types"] = types_module


def pytest_configure() -> None:
    cv2_module = types.ModuleType("cv2")
    cv2_module.CAP_ANY = 0
    cv2_module.CAP_AVFOUNDATION = 1
    cv2_module.CAP_DSHOW = 2
    cv2_module.IMWRITE_JPEG_QUALITY = 95
    cv2_module.VideoCapture = object
    cv2_module.imencode = lambda *args, **kwargs: (False, None)
    sys.modules.setdefault("cv2", cv2_module)

    pyaudio_module = types.ModuleType("pyaudio")
    pyaudio_module.paInt16 = 16

    class _PyAudio:
        def open(self, *args, **kwargs):
            return None

        def terminate(self) -> None:
            return None

    pyaudio_module.PyAudio = _PyAudio
    sys.modules.setdefault("pyaudio", pyaudio_module)

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    sys.modules.setdefault("dotenv", dotenv_module)

    _install_google_genai_stub()
