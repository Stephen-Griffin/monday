from __future__ import annotations

from typing import Callable, Protocol

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.events import EventBus, StatusEvent, ToolProposalEvent, TranscriptEvent


class MainWindowController(Protocol):
    def connect_realtime(self) -> None: ...
    def disconnect_realtime(self) -> None: ...
    def start_listening(self) -> None: ...
    def stop_listening(self) -> None: ...
    def shutdown(self) -> None: ...


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
        self._subscriptions: list[Callable[[], None]] = []

        self.setWindowTitle("Monday Lite (MVP Scaffold)")
        self.resize(1000, 680)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.status_label = QLabel(
            f"Status: Idle | Model: {config.model} | Dry-run: {dry_run}",
            self,
        )
        layout.addWidget(self.status_label)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.connect_button = QPushButton("Connect Realtime", self)
        self.connect_button.clicked.connect(self._on_connect_clicked)
        controls_layout.addWidget(self.connect_button)

        self.listen_button = QPushButton("Start Listening", self)
        self.listen_button.setEnabled(False)
        self.listen_button.clicked.connect(self._on_listen_clicked)
        controls_layout.addWidget(self.listen_button)
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

        actions_box = QGroupBox("Actions Queue", self)
        actions_layout = QVBoxLayout(actions_box)
        self.actions_list = QListWidget(self)
        self.actions_list.addItem("No actions proposed yet.")
        actions_layout.addWidget(self.actions_list)
        left_column.addWidget(actions_box, 2)
        content_layout.addLayout(left_column, 2)

        notes_box = QGroupBox("Monday Notes", self)
        notes_layout = QVBoxLayout(notes_box)
        self.notes_editor = QTextEdit(self)
        self.notes_editor.setPlaceholderText("Assistant-approved notes will appear here.")
        notes_layout.addWidget(self.notes_editor)
        content_layout.addWidget(notes_box, 1)

        layout.addLayout(content_layout)
        self.setCentralWidget(central)

        self._subscriptions.append(self._event_bus.subscribe(TranscriptEvent, self._on_transcript_event))
        self._subscriptions.append(self._event_bus.subscribe(ToolProposalEvent, self._on_tool_proposal_event))
        self._subscriptions.append(self._event_bus.subscribe(StatusEvent, self._on_status_event))

    def _on_connect_clicked(self) -> None:
        if not self._connected:
            self._controller.connect_realtime()
            self._connected = True
            self.connect_button.setText("Disconnect Realtime")
            self.listen_button.setEnabled(True)
            return

        if self._listening:
            self._controller.stop_listening()
            self._listening = False
            self.listen_button.setText("Start Listening")

        self._controller.disconnect_realtime()
        self._connected = False
        self.connect_button.setText("Connect Realtime")
        self.listen_button.setEnabled(False)

    def _on_listen_clicked(self) -> None:
        if not self._listening:
            self._controller.start_listening()
            self._listening = True
            self.listen_button.setText("Stop Listening")
            return

        self._controller.stop_listening()
        self._listening = False
        self.listen_button.setText("Start Listening")

    def _on_transcript_event(self, event: TranscriptEvent) -> None:
        line = f"[{event.source}] {event.text}"
        QTimer.singleShot(0, lambda: self.transcript_pane.appendPlainText(line))

    def _on_tool_proposal_event(self, event: ToolProposalEvent) -> None:
        if self.actions_list.count() == 1 and self.actions_list.item(0).text() == "No actions proposed yet.":
            self.actions_list.clear()
        summary = f"{event.proposal_id}: {event.tool} | args={event.args}"
        QTimer.singleShot(0, lambda: self.actions_list.addItem(summary))

    def _on_status_event(self, event: StatusEvent) -> None:
        detail = f" ({event.detail})" if event.detail else ""
        text = f"Status: {event.component} -> {event.status}{detail}"
        QTimer.singleShot(0, lambda: self.status_label.setText(text))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._controller.shutdown()
        event.accept()
