from __future__ import annotations

from dataclasses import dataclass
import webbrowser


@dataclass
class ActionResult:
    executed: bool
    detail: str


class ActionExecutor:
    """
    Performs approved actions only.

    TODO(Agent 4): Wire validation and permission checks before invoking methods.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run

    def open_url(self, url: str) -> ActionResult:
        if self._dry_run:
            return ActionResult(executed=False, detail=f"dry-run prevented opening {url}")

        opened = webbrowser.open(url)
        return ActionResult(executed=opened, detail=f"open_url returned {opened}")

    def write_notes(self, current_text: str, content: str, mode: str = "append") -> str:
        if mode == "replace":
            return content
        return f"{current_text.rstrip()}\n{content}".strip()

