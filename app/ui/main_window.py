from __future__ import annotations

from datetime import datetime
import json
from typing import Callable, Protocol

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.events import (
    ActionDecisionEvent,
    AudioFrameEvent,
    EventBus,
    NotesUpdateEvent,
    StatusEvent,
    ToolProposalEvent,
    TranscriptEvent,
)


class MainWindowController(Protocol):
    def connect_realtime(self) -> None: ...
    def disconnect_realtime(self) -> None: ...
    def start_listening(self) -> None: ...
    def stop_listening(self) -> None: ...
    def shutdown(self) -> None: ...


class ActionProposalCard(QFrame):
    def __init__(
        self,
        proposal: ToolProposalEvent,
        on_decision: Callable[[str, str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._proposal_id = proposal.proposal_id
        self._on_decision = on_decision
        self._submitted = False

        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel(f"Tool: {proposal.tool}  |  Action ID: {proposal.proposal_id}", self)
        layout.addWidget(title)

        reason = proposal.reason or "No reason provided."
        self.reason_label = QLabel(f"Reason: {reason}", self)
        self.reason_label.setWordWrap(True)
        layout.addWidget(self.reason_label)

        args_label = QLabel("Arguments:", self)
        layout.addWidget(args_label)

        args_text = QPlainTextEdit(self)
        args_text.setReadOnly(True)
        args_text.setMaximumHeight(120)
        try:
            pretty_args = json.dumps(proposal.args, indent=2, sort_keys=True, ensure_ascii=True)
        except TypeError:
            pretty_args = json.dumps({"raw": str(proposal.args)}, indent=2, ensure_ascii=True)
        args_text.setPlainText(pretty_args)
        layout.addWidget(args_text)

        self.decision_label = QLabel("Decision: pending", self)
        layout.addWidget(self.decision_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)

        self.approve_button = QPushButton("Approve", self)
        self.approve_button.clicked.connect(lambda: self._submit_decision("approved"))
        buttons.addWidget(self.approve_button)

        self.reject_button = QPushButton("Reject", self)
        self.reject_button.clicked.connect(lambda: self._submit_decision("rejected"))
        buttons.addWidget(self.reject_button)

        buttons.addStretch(1)
        layout.addLayout(buttons)

    def _submit_decision(self, decision: str) -> None:
        if self._submitted:
            return
        self._submitted = True
        self.approve_button.setEnabled(False)
        self.reject_button.setEnabled(False)
        self.decision_label.setText(f"Decision: {decision}")
        self._on_decision(self._proposal_id, decision)


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        dry_run: bool,
        event_bus: EventBus,
        controller: MainWindowController,
    ) -> None:
        super().__init__()
        self._event_bus = event_bus
        self._controller = controller
        self._connected = False
        self._listening = False
        self._speaking = False
        self._hold_to_talk_active = False
        self._subscriptions: list[Callable[[], None]] = []
        self._action_cards: dict[str, ActionProposalCard] = {}

        self.setWindowTitle("Monday Lite (MVP Scaffold)")
        self.resize(1040, 700)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.status_label = QLabel(
            f"Status: Idle | Model: {config.model} | Dry-run: {dry_run}",
            self,
        )
        layout.addWidget(self.status_label)

        indicators = QHBoxLayout()
        indicators.setSpacing(14)

        indicators.addWidget(QLabel("Connected:", self))
        self.connected_indicator = QLabel("OFF", self)
        indicators.addWidget(self.connected_indicator)

        indicators.addWidget(QLabel("Listening:", self))
        self.listening_indicator = QLabel("OFF", self)
        indicators.addWidget(self.listening_indicator)

        indicators.addWidget(QLabel("Speaking:", self))
        self.speaking_indicator = QLabel("OFF", self)
        indicators.addWidget(self.speaking_indicator)

        indicators.addWidget(QLabel("Streaming to cloud:", self))
        self.streaming_indicator = QLabel("OFF", self)
        indicators.addWidget(self.streaming_indicator)

        indicators.addStretch(1)
        layout.addLayout(indicators)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.connect_button = QPushButton("Connect Realtime", self)
        self.connect_button.clicked.connect(self._on_connect_clicked)
        controls_layout.addWidget(self.connect_button)

        self.listen_button = QPushButton("Start Listening", self)
        self.listen_button.setEnabled(False)
        self.listen_button.clicked.connect(self._on_listen_clicked)
        controls_layout.addWidget(self.listen_button)

        self.keyboard_hint = QLabel("Hold Space to talk (push-to-talk).", self)
        controls_layout.addWidget(self.keyboard_hint)

        controls_layout.addStretch(1)
        layout.addLayout(controls_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        transcript_box = QGroupBox("Transcript", self)
        transcript_layout = QVBoxLayout(transcript_box)
        self.transcript_pane = QPlainTextEdit(self)
        self.transcript_pane.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_pane)
        left_column.addWidget(transcript_box, 3)

        actions_box = QGroupBox("Action Proposals", self)
        actions_layout = QVBoxLayout(actions_box)
        self.actions_scroll = QScrollArea(self)
        self.actions_scroll.setWidgetResizable(True)

        self.actions_container = QWidget(self.actions_scroll)
        self.actions_cards_layout = QVBoxLayout(self.actions_container)
        self.actions_cards_layout.setContentsMargins(4, 4, 4, 4)
        self.actions_cards_layout.setSpacing(8)

        self.actions_empty_label = QLabel("No actions proposed yet.", self.actions_container)
        self.actions_cards_layout.addWidget(self.actions_empty_label)
        self.actions_cards_layout.addStretch(1)

        self.actions_scroll.setWidget(self.actions_container)
        actions_layout.addWidget(self.actions_scroll)
        left_column.addWidget(actions_box, 2)

        content_layout.addLayout(left_column, 2)

        notes_box = QGroupBox("Monday Notes", self)
        notes_layout = QVBoxLayout(notes_box)

        self.notes_banner = QLabel("", self)
        self.notes_banner.setVisible(False)
        notes_layout.addWidget(self.notes_banner)

        self.notes_editor = QTextEdit(self)
        self.notes_editor.setPlaceholderText("Assistant-approved notes updates will appear here.")
        notes_layout.addWidget(self.notes_editor)

        content_layout.addWidget(notes_box, 1)
        layout.addLayout(content_layout)
        self.setCentralWidget(central)

        self._speaking_decay_timer = QTimer(self)
        self._speaking_decay_timer.setSingleShot(True)
        self._speaking_decay_timer.timeout.connect(lambda: self._set_speaking(False))

        self._subscriptions.append(self._event_bus.subscribe(TranscriptEvent, self._on_transcript_event))
        self._subscriptions.append(self._event_bus.subscribe(ToolProposalEvent, self._on_tool_proposal_event))
        self._subscriptions.append(self._event_bus.subscribe(StatusEvent, self._on_status_event))
        self._subscriptions.append(self._event_bus.subscribe(NotesUpdateEvent, self._on_notes_update_event))
        self._subscriptions.append(self._event_bus.subscribe(AudioFrameEvent, self._on_audio_frame_event))

        self._refresh_status_indicators()

    def _on_connect_clicked(self) -> None:
        if not self._connected:
            self._controller.connect_realtime()
            self._set_connected(True)
            return

        if self._listening:
            self._controller.stop_listening()
            self._set_listening(False)

        self._controller.disconnect_realtime()
        self._set_connected(False)

    def _on_listen_clicked(self) -> None:
        if self._hold_to_talk_active:
            return

        if not self._listening:
            self._controller.start_listening()
            self._set_listening(True)
            return

        self._controller.stop_listening()
        self._set_listening(False)

    def _on_transcript_event(self, event: TranscriptEvent) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        speaker = (event.source or "unknown").strip() or "unknown"
        suffix = "" if event.is_final else " (partial)"
        line = f"[{timestamp}] {speaker.title()}: {event.text}{suffix}"

        QTimer.singleShot(0, lambda: self.transcript_pane.appendPlainText(line))

        if speaker.lower() in {"assistant", "jarvis", "model"}:
            QTimer.singleShot(0, self._mark_speaking_activity)

    def _on_tool_proposal_event(self, event: ToolProposalEvent) -> None:
        QTimer.singleShot(0, lambda: self._add_tool_proposal_card(event))

    def _on_status_event(self, event: StatusEvent) -> None:
        QTimer.singleShot(0, lambda: self._apply_status_event(event))

    def _on_notes_update_event(self, event: NotesUpdateEvent) -> None:
        QTimer.singleShot(0, lambda: self._apply_notes_update(event))

    def _on_audio_frame_event(self, event: AudioFrameEvent) -> None:
        if event.source.lower() != "mic":
            QTimer.singleShot(0, self._mark_speaking_activity)

    def _add_tool_proposal_card(self, proposal: ToolProposalEvent) -> None:
        if proposal.proposal_id in self._action_cards:
            return

        self.actions_empty_label.setVisible(False)

        card = ActionProposalCard(
            proposal=proposal,
            on_decision=self._emit_action_decision,
            parent=self.actions_container,
        )
        insert_index = max(0, self.actions_cards_layout.count() - 1)
        self.actions_cards_layout.insertWidget(insert_index, card)
        self._action_cards[proposal.proposal_id] = card

    def _emit_action_decision(self, action_id: str, decision: str) -> None:
        self._event_bus.publish(ActionDecisionEvent(action_id=action_id, decision=decision))

    def _apply_status_event(self, event: StatusEvent) -> None:
        detail = f" ({event.detail})" if event.detail else ""
        self.status_label.setText(f"Status: {event.component} -> {event.status}{detail}")

        if event.component == "realtime":
            if event.status == "connected":
                self._set_connected(True)
            elif event.status == "disconnected":
                self._set_connected(False)

        if event.component == "audio":
            if event.status == "listening":
                self._set_listening(True)
            elif event.status in {"idle", "blocked", "stopped"}:
                self._set_listening(False)

        if event.component in {"assistant", "tts", "speech"}:
            if event.status in {"speaking", "talking"}:
                self._set_speaking(True)
            elif event.status in {"idle", "silent", "stopped"}:
                self._set_speaking(False)

    def _apply_notes_update(self, event: NotesUpdateEvent) -> None:
        mode = event.mode.lower()
        current_text = self.notes_editor.toPlainText()

        if mode == "replace":
            self.notes_editor.setPlainText(event.content)
        else:
            if not current_text.strip():
                next_text = event.content
            elif not event.content:
                next_text = current_text
            else:
                next_text = f"{current_text.rstrip()}\n{event.content}"
            self.notes_editor.setPlainText(next_text)

        self._show_notes_banner(f"{event.source} updated notes ({mode}).")

    def _show_notes_banner(self, text: str) -> None:
        self.notes_banner.setText(text)
        self.notes_banner.setVisible(True)
        QTimer.singleShot(1800, lambda: self.notes_banner.setVisible(False))

    def _mark_speaking_activity(self) -> None:
        self._set_speaking(True)
        self._speaking_decay_timer.start(1400)

    def _set_connected(self, connected: bool) -> None:
        self._connected = connected
        self.connect_button.setText("Disconnect Realtime" if connected else "Connect Realtime")
        self.listen_button.setEnabled(connected)
        if not connected:
            self._set_listening(False)
        self._refresh_status_indicators()

    def _set_listening(self, listening: bool) -> None:
        self._listening = listening
        self.listen_button.setText("Stop Listening" if listening else "Start Listening")
        if not listening:
            self._hold_to_talk_active = False
        self._refresh_status_indicators()

    def _set_speaking(self, speaking: bool) -> None:
        self._speaking = speaking
        if not speaking:
            self._speaking_decay_timer.stop()
        self._refresh_status_indicators()

    def _refresh_status_indicators(self) -> None:
        self.connected_indicator.setText("ON" if self._connected else "OFF")
        self.listening_indicator.setText("ON" if self._listening else "OFF")
        self.speaking_indicator.setText("ON" if self._speaking else "OFF")
        self.streaming_indicator.setText("ON" if self._listening else "OFF")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            focus_widget = self.focusWidget()
            if isinstance(focus_widget, (QTextEdit, QPlainTextEdit)):
                super().keyPressEvent(event)
                return
            if self._connected and not self._listening:
                self._hold_to_talk_active = True
                self._controller.start_listening()
                self._set_listening(True)
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self._hold_to_talk_active:
                self._controller.stop_listening()
                self._set_listening(False)
                event.accept()
                return

        super().keyReleaseEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._controller.shutdown()
        event.accept()
