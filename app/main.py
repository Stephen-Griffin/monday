from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

from app.audio import AudioEngine
from app.config import AppConfig, load_config
from app.events import EventBus, StatusEvent
from app.logging_setup import setup_logging
from app.permissions import PermissionGate
from app.realtime import RealtimeClient
from app.state import ConversationState
from app.ui import MainWindow
from app.vision import CameraService


class AppController:
    """Connects UI controls to backend service stubs."""

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        permission_gate: PermissionGate,
        realtime_client: RealtimeClient,
        audio_engine: AudioEngine,
        camera_service: CameraService,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._permission_gate = permission_gate
        self._realtime_client = realtime_client
        self._audio_engine = audio_engine
        self._camera_service = camera_service
        self._logger = logging.getLogger(__name__)

    def connect_realtime(self) -> None:
        self._realtime_client.connect()

    def disconnect_realtime(self) -> None:
        self._audio_engine.stop_listening()
        self._realtime_client.stop_listening()
        self._realtime_client.disconnect()

    def start_listening(self) -> None:
        if not self._realtime_client.connected:
            self._event_bus.publish(
                StatusEvent(
                    component="audio",
                    status="blocked",
                    detail="Connect realtime before listening.",
                )
            )
            return
        self._realtime_client.start_listening()
        self._audio_engine.start_listening()

    def stop_listening(self) -> None:
        self._audio_engine.stop_listening()
        self._realtime_client.stop_listening()

    def shutdown(self) -> None:
        self._audio_engine.stop_listening()
        self._realtime_client.stop_listening()
        self._realtime_client.disconnect()
        self._logger.info(
            "Controller shutdown complete (model=%s, allowlist=%s).",
            self._config.model,
            ",".join(self._config.allowlist_domains),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monday-lite MVP scaffold.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never execute side effects (open URL, write notes).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log_paths = setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config(Path(args.env_file))

    logger.info("Starting monday-lite scaffold.")
    logger.info("Logs directory ready: %s", log_paths.base_dir)
    if not config.openai_api_key:
        logger.warning("OPENAI_API_KEY is empty; realtime integration remains stubbed.")

    event_bus = EventBus()
    state = ConversationState()
    event_bus.subscribe_all(state.handle_event)

    permission_gate = PermissionGate(dry_run=args.dry_run)
    realtime_client = RealtimeClient(
        event_bus=event_bus,
        model=config.model,
        api_key=config.openai_api_key,
    )
    audio_engine = AudioEngine(event_bus=event_bus)
    camera_service = CameraService()

    controller = AppController(
        config=config,
        event_bus=event_bus,
        permission_gate=permission_gate,
        realtime_client=realtime_client,
        audio_engine=audio_engine,
        camera_service=camera_service,
    )

    qapp = QApplication(sys.argv)
    window = MainWindow(
        config=config,
        dry_run=args.dry_run,
        event_bus=event_bus,
        controller=controller,
    )
    window.show()
    event_bus.publish(StatusEvent(component="system", status="ready", detail="UI launched"))
    return qapp.exec()


if __name__ == "__main__":
    raise SystemExit(main())
