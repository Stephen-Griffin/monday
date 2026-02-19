from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    proposal_id: str
    tool: str
    args: dict[str, Any]
    reason: str | None = None

