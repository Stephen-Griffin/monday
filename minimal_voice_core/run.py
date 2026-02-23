import argparse
import asyncio

from monday_voice_core import MondayVoiceCore


def parse_args():
    parser = argparse.ArgumentParser(description="Monday minimal voice + text + camera + browser opener")
    parser.add_argument("--model", default="models/gemini-2.5-flash-native-audio-preview-12-2025")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--no-camera", action="store_true")
    return parser.parse_args()


async def main():
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


if __name__ == "__main__":
    asyncio.run(main())
