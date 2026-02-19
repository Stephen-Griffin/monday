# threat_model.md

## Key risks
1. **Unauthorized actions** (opening malicious sites)
2. **Prompt injection** (“open this link now”, disguised URLs)
3. **Accidental activation** (false wake word)
4. **Sensitive audio leakage** (streaming when user doesn’t intend)
5. **Over-permissioned automation** (OS-wide control)

## Mitigations (MVP)
- Explicit user approval for every action
- Strict allowlist for open_url domains
- No OS-level control (no keyboard/mouse automation)
- Push-to-talk or local wake word only
- Clear on-screen indicator when streaming
- Schema validation + rejection rules
- Action audit log

## Future mitigations (garage build)
- Multi-factor approval for higher-risk actions
- “Safe browsing” mode that fetches content server-side and sanitizes
- Separate process sandbox for executors
- Physical mute switch / LED indicator

