import unittest

from app.events import EventBus, StatusEvent, TranscriptEvent


class EventBusTests(unittest.TestCase):
    def test_typed_subscription_receives_matching_event(self) -> None:
        bus = EventBus()
        received: list[TranscriptEvent] = []

        bus.subscribe(TranscriptEvent, received.append)
        event = TranscriptEvent(source="assistant", text="hello", is_final=True)
        bus.publish(event)

        self.assertEqual(received, [event])

    def test_global_subscription_receives_all_events(self) -> None:
        bus = EventBus()
        received = []

        bus.subscribe_all(received.append)
        event = StatusEvent(component="system", status="ready")
        bus.publish(event)

        self.assertEqual(received, [event])


if __name__ == "__main__":
    unittest.main()

