from __future__ import annotations

from dataclasses import dataclass, field

from app.events import AppEvent, ToolProposalEvent, TranscriptEvent


@dataclass
class ConversationState:
    transcripts: list[TranscriptEvent] = field(default_factory=list)
    pending_tool_proposals: dict[str, ToolProposalEvent] = field(default_factory=dict)

    def handle_event(self, event: AppEvent) -> None:
        if isinstance(event, TranscriptEvent):
            self.transcripts.append(event)
            return
        if isinstance(event, ToolProposalEvent):
            self.pending_tool_proposals[event.proposal_id] = event

