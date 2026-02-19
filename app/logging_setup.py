from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass(frozen=True)
class LogPaths:
    base_dir: Path
    app_log: Path
    actions_log: Path


def setup_logging(log_dir: Path | str = "logs") -> LogPaths:
    """Configure root logging and ensure the logs directory exists."""
    base_dir = Path(log_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    app_log = base_dir / "app.log"
    actions_log = base_dir / "actions.jsonl"
    actions_log.touch(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(app_log, encoding="utf-8"),
        ],
        force=True,
    )

    return LogPaths(base_dir=base_dir, app_log=app_log, actions_log=actions_log)

