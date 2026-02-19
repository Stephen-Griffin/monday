from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ToolProposal:
    tool: Literal["open_url", "write_notes"]
    args: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ToolCall:
    proposal_id: str
    tool: str
    args: dict[str, Any]
    reason: str | None = None
