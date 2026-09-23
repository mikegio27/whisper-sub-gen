#!/usr/bin/env bash
# Claude Code hook. Reads the hook JSON on stdin.
set -uo pipefail
input=$(cat)
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
# Never loop: if we already blocked once this turn, let Claude stop.
[ "$(jq -r '.stop_hook_active // false' <<<"$input")" = "true" ] && exit 0
# Only when this repo has uncommitted changes to relevant files.
git status --porcelain -uall -- '*.py' 'pyproject.toml' | grep -q . || exit 0
if ! out=$({ ruff check -q . && ruff format --check -q . && .venv/bin/python -m unittest discover -s tests ; } 2>&1); then
  echo "ruff / unittest failed — fix before finishing:" >&2
  tail -n 60 <<<"$out" >&2
  exit 2
fi
