from __future__ import annotations

import logging


class CameraService:
    """
    Optional camera scaffold for single-snapshot capture.

    TODO(Agent 2 or Agent 3): Implement OpenCV capture in snapshot mode.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def capture_snapshot(self) -> bytes | None:
        self._logger.info("Camera snapshot requested, but camera service is still a stub.")
        return None

