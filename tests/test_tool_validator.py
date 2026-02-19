import unittest

from app.tools.validator import (
    ValidationError,
    is_allowed_domain,
    parse_tool_proposal,
)


class ToolValidatorTests(unittest.TestCase):
    def test_parse_open_url_valid(self) -> None:
        proposal = parse_tool_proposal(
            {
                "tool": "open_url",
                "args": {"url": "https://www.youtube.com/watch?v=123"},
                "reason": "User asked me to open the video.",
            },
            allowlist_domains=("youtube.com", "youtu.be"),
        )

        self.assertEqual(proposal.tool, "open_url")
        self.assertEqual(proposal.args, {"url": "https://www.youtube.com/watch?v=123"})
        self.assertEqual(proposal.reason, "User asked me to open the video.")

    def test_open_url_rejects_blocked_schemes(self) -> None:
        for blocked in ("file:///tmp/a", "javascript:alert(1)", "data:text/plain,hi"):
            with self.assertRaises(ValidationError):
                parse_tool_proposal(
                    {"tool": "open_url", "args": {"url": blocked}, "reason": "test"},
                    allowlist_domains=("youtube.com",),
                )

    def test_open_url_rejects_domain_outside_allowlist(self) -> None:
        with self.assertRaises(ValidationError):
            parse_tool_proposal(
                {
                    "tool": "open_url",
                    "args": {"url": "https://example.com"},
                    "reason": "User asked",
                },
                allowlist_domains=("youtube.com",),
            )

    def test_parse_write_notes_defaults_to_append(self) -> None:
        proposal = parse_tool_proposal(
            {
                "tool": "write_notes",
                "args": {"text": " summarize this "},
                "reason": "User asked to capture notes.",
            },
            allowlist_domains=("youtube.com",),
        )

        self.assertEqual(proposal.tool, "write_notes")
        self.assertEqual(proposal.args["text"], "summarize this")
        self.assertEqual(proposal.args["mode"], "append")

    def test_write_notes_rejects_excessive_text(self) -> None:
        with self.assertRaises(ValidationError):
            parse_tool_proposal(
                {
                    "tool": "write_notes",
                    "args": {"text": "a" * 11},
                    "reason": "User asked",
                },
                allowlist_domains=("youtube.com",),
                max_notes_text_chars=10,
            )

    def test_missing_required_top_level_key_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            parse_tool_proposal(
                {"tool": "write_notes", "args": {"text": "hello"}},
                allowlist_domains=("youtube.com",),
            )

    def test_allowlist_helper_accepts_exact_and_subdomain_only(self) -> None:
        self.assertTrue(is_allowed_domain("youtube.com", ("youtube.com",)))
        self.assertTrue(is_allowed_domain("www.youtube.com", ("youtube.com",)))
        self.assertFalse(is_allowed_domain("notyoutube.com", ("youtube.com",)))
        self.assertFalse(is_allowed_domain("youtube.com.evil.org", ("youtube.com",)))


if __name__ == "__main__":
    unittest.main()
