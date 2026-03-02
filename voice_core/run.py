import argparse
import asyncio

try:
    from .monday_voice_core import MondayVoiceCore
except ImportError:
    # Support direct script execution from inside the voice_core directory.
    from monday_voice_core import MondayVoiceCore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monday minimal voice + text + camera + browser opener"
    )
    parser.add_argument(
        "--model", default="models/gemini-2.5-flash-native-audio-preview-12-2025"
    )
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--no-camera", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    app = MondayVoiceCore(
        model=args.model,
        camera_index=args.camera_index,
        enable_camera=not args.no_camera,
    )
    try:
        await app.run()
    finally:
        app.close()


def cli() -> None:
    asyncio.run(main())
