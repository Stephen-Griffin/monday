# schema.md

## Tool call schema (MVP)

All tool proposals MUST be valid JSON matching one of the following shapes.
The model should never be allowed to execute arbitrary code.

### 1) open_url
Opens a URL in the default browser (Safari/Chrome).
Allowed domains are configured by allowlist.

```json
{
  "tool": "open_url",
  "args": {
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "reason": "User asked to open the tutorial video."
  }
}

