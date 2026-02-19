from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-realtime-mini"
DEFAULT_ALLOWLIST_DOMAINS = ("youtube.com", "youtu.be")


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    model: str
    allowlist_domains: tuple[str, ...]


def _parse_allowlist(raw_value: str | None) -> tuple[str, ...]:
    if raw_value is None or raw_value.strip() == "":
        return DEFAULT_ALLOWLIST_DOMAINS

    domains = tuple(
        domain.strip().lower()
        for domain in raw_value.split(",")
        if domain.strip()
    )
    return domains or DEFAULT_ALLOWLIST_DOMAINS


def load_config(env_path: Path | None = None) -> AppConfig:
    """Load application settings from .env and process environment."""
    resolved_env_path = env_path or Path(".env")
    load_dotenv(dotenv_path=resolved_env_path, override=False)

    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("MODEL", DEFAULT_MODEL),
        allowlist_domains=_parse_allowlist(os.getenv("ALLOWLIST_DOMAINS")),
    )

