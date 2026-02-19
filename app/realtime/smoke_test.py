from __future__ import annotations

import argparse
from pathlib import Path
import time

from app.audio import AudioEngine
from app.config import load_config
from app.events import EventBus, StatusEvent, TranscriptEvent
from app.realtime import RealtimeClient


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime audio smoke test.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=3.0,
        help="How long to capture microphone audio before stopping.",
    )
    parser.add_argument(
        "--response-wait-seconds",
        type=float,
        default=8.0,
        help="How long to wait for transcript events after stopping recording.",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=10.0,
        help="How long to wait for websocket connection.",
    )
    return parser.parse_args(argv)


def _wait_for_connection(client: RealtimeClient, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if client.connected:
            return True
        time.sleep(0.1)
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.env_file))

    if not config.openai_api_key:
        print("OPENAI_API_KEY is missing. Set it in your .env before running smoke_test.")
        return 1

    event_bus = EventBus()
    realtime_client = RealtimeClient(event_bus=event_bus, model=config.model, api_key=config.openai_api_key)
    audio_engine = AudioEngine(event_bus=event_bus)

    def on_status(event: StatusEvent) -> None:
        detail = f" ({event.detail})" if event.detail else ""
        print(f"[status] {event.component}: {event.status}{detail}")

    def on_transcript(event: TranscriptEvent) -> None:
        kind = "final" if event.is_final else "partial"
        print(f"[transcript:{kind}] {event.source}: {event.text}")

    event_bus.subscribe(StatusEvent, on_status)
    event_bus.subscribe(TranscriptEvent, on_transcript)

    realtime_client.connect()
    if not _wait_for_connection(realtime_client, timeout_seconds=args.connect_timeout_seconds):
        print("Timed out waiting for Realtime websocket connection.")
        realtime_client.disconnect()
        return 2

    try:
        print(f"Recording mic for {args.record_seconds:.1f}s...")
        realtime_client.start_listening()
        audio_engine.start_listening()
        time.sleep(args.record_seconds)

        print("Stopping mic and waiting for model response...")
        audio_engine.stop_listening()
        realtime_client.stop_listening()
        time.sleep(args.response_wait_seconds)
    finally:
        audio_engine.stop_listening()
        realtime_client.stop_listening()
        realtime_client.disconnect()

    print("Smoke test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
