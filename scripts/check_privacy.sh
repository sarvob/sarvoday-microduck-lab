#!/bin/sh
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

status=0

tracked=$(git ls-files -co --exclude-standard | grep -v '^scripts/check_privacy\.sh$' || true)
if [ -n "$tracked" ]; then
  if printf '%s\n' "$tracked" | xargs grep -nEI \
    '(/Users/[^/[:space:]]+)|([[:alnum:]._%+-]+@(gmail|outlook|hotmail|yahoo|icloud)\.[[:alpha:]]+)|(gh[pousr]_[[:alnum:]_]{20,})|(sk-[[:alnum:]_-]{20,})|(BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY)' \
    2>/dev/null; then
    echo "Privacy check failed: sensitive-looking content is present." >&2
    status=1
  fi
fi

author_name=$(git config --local user.name || true)
author_email=$(git config --local user.email || true)
if [ "$author_name" != "Sarvoday Robotics" ] || \
   [ "$author_email" != "automation@sarvoday.invalid" ]; then
  echo "Privacy check failed: local Git identity is not the project identity." >&2
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "Privacy check passed."
fi
exit "$status"
