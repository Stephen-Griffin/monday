from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from app.tools.contracts import ToolProposal

KNOWN_TOOLS = frozenset({"open_url", "write_notes"})
BLOCKED_URL_SCHEMES = frozenset({"file", "javascript", "data"})
ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
DEFAULT_MAX_NOTES_TEXT_CHARS = 4000


@dataclass(frozen=True)
class ValidationError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


def parse_tool_proposal(
    raw_payload: str | dict[str, Any],
    allowlist_domains: Iterable[str],
    max_notes_text_chars: int = DEFAULT_MAX_NOTES_TEXT_CHARS,
) -> ToolProposal:
    payload = _coerce_payload(raw_payload)
    _require_exact_keys(payload, required_keys={"tool", "args", "reason"}, context="root")

    tool = payload["tool"]
    args = payload["args"]
    reason = payload["reason"]

    if not isinstance(tool, str) or tool not in KNOWN_TOOLS:
        raise ValidationError(f"unknown tool: {tool!r}")
    if not isinstance(args, dict):
        raise ValidationError("args must be an object")
    if not isinstance(reason, str) or not reason.strip():
        raise ValidationError("reason must be a non-empty string")

    normalized_reason = reason.strip()
    normalized_allowlist = tuple(_normalize_domain(domain) for domain in allowlist_domains if domain.strip())
    if not normalized_allowlist:
        raise ValidationError("allowlist is empty")

    if tool == "open_url":
        return ToolProposal(
            tool="open_url",
            args={"url": _validate_open_url_args(args, normalized_allowlist)},
            reason=normalized_reason,
        )

    return ToolProposal(
        tool="write_notes",
        args=_validate_write_notes_args(args, max_notes_text_chars),
        reason=normalized_reason,
    )


def is_allowed_domain(hostname: str, allowlist_domains: Iterable[str]) -> bool:
    normalized_host = _normalize_domain(hostname)
    normalized_allowlist = tuple(_normalize_domain(domain) for domain in allowlist_domains)
    return any(
        normalized_host == allowed or normalized_host.endswith(f".{allowed}")
        for allowed in normalized_allowlist
    )


def _coerce_payload(raw_payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_payload, dict):
        return raw_payload
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValidationError("proposal payload must be a JSON object")
    return parsed


def _validate_open_url_args(args: dict[str, Any], allowlist_domains: tuple[str, ...]) -> str:
    _require_exact_keys(args, required_keys={"url"}, context="open_url.args")
    url = args["url"]
    if not isinstance(url, str) or not url.strip():
        raise ValidationError("open_url.args.url must be a non-empty string")

    stripped_url = url.strip()
    split_url = urlsplit(stripped_url)
    scheme = split_url.scheme.lower()
    if scheme in BLOCKED_URL_SCHEMES:
        raise ValidationError(f"url scheme {scheme!r} is not allowed")
    if scheme not in ALLOWED_URL_SCHEMES:
        raise ValidationError("url must use http or https")
    if not split_url.hostname:
        raise ValidationError("url hostname is required")
    if not is_allowed_domain(split_url.hostname, allowlist_domains):
        raise ValidationError("url domain is not in allowlist")

    normalized_scheme = scheme
    normalized_netloc = split_url.netloc
    return urlunsplit(
        (normalized_scheme, normalized_netloc, split_url.path, split_url.query, split_url.fragment)
    )


def _validate_write_notes_args(args: dict[str, Any], max_notes_text_chars: int) -> dict[str, Any]:
    allowed_keys = {"text", "mode"}
    if not args.keys() <= allowed_keys:
        unexpected = sorted(set(args.keys()) - allowed_keys)
        raise ValidationError(f"write_notes.args has unexpected keys: {unexpected}")
    if "text" not in args:
        raise ValidationError("write_notes.args.text is required")

    text = args["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("write_notes.args.text must be a non-empty string")
    normalized_text = text.strip()
    if len(normalized_text) > max_notes_text_chars:
        raise ValidationError(
            f"write_notes.args.text exceeds max length ({max_notes_text_chars} characters)"
        )

    mode = args.get("mode", "append")
    if mode not in {"append", "replace"}:
        raise ValidationError("write_notes.args.mode must be 'append' or 'replace'")

    return {"text": normalized_text, "mode": mode}


def _require_exact_keys(payload: dict[str, Any], required_keys: set[str], context: str) -> None:
    actual_keys = set(payload.keys())
    missing = required_keys - actual_keys
    unexpected = actual_keys - required_keys
    if missing:
        raise ValidationError(f"{context} missing required keys: {sorted(missing)}")
    if unexpected:
        raise ValidationError(f"{context} has unexpected keys: {sorted(unexpected)}")


def _normalize_domain(raw_domain: str) -> str:
    return raw_domain.strip().lower().rstrip(".")
