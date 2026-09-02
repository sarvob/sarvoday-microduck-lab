# Privacy contract

This repository is designed for a public build stream.

- Commits use the project identity `Sarvoday Robotics <automation@sarvoday.invalid>`.
- Generated reports contain relative paths only.
- Secrets, tokens, personal email addresses, and local home-directory paths must never be committed.
- Run `./scripts/check_privacy.sh` before every commit and push.
- Do not record the full desktop. Capture only the dedicated Codex task and simulator window.

The privacy check intentionally fails on common personal-email domains, absolute macOS user paths, API-token prefixes, private keys, and non-project commit identities.
