#!/usr/bin/env bash
# Claude Code hook. Reads the hook JSON on stdin.
set -uo pipefail
input=$(cat)
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
f=$(jq -r '.tool_input.file_path // empty' <<<"$input")
case "$f" in *.py) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
ruff format -q "$f" >&2 || true
ruff check -q --fix "$f" >&2 || true
exit 0
