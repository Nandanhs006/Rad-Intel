#!/bin/bash
#
# Start Rad-Intel and keep the machine awake while it runs.
#
# Called by the launchd job in this folder, and runnable by hand to see what
# it does. It deliberately does not daemonise: launchd wants a process that
# stays in the foreground so it can restart it when it dies.

set -euo pipefail

# Where the project is. Change this if you put it somewhere else.
PROJECT="${RAD_INTEL_HOME:-$HOME/Documents/GitHub/Rad-Intel}"

cd "$PROJECT"

# Settings live in .env beside the code. Read here rather than in launchd, so
# changing one is editing a file rather than reinstalling a service. Lines
# that are blank or commented are skipped, and values may contain '='.
if [ -f .env ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|'#'*) continue ;;
      *=*) export "${line%%=*}=${line#*=}" ;;
    esac
  done < .env
fi

# After .env, not before: .env carries a PORT of its own, and reading it
# first meant that value silently replaced whatever had been chosen here.
# Order of preference: what the caller asked for, then .env, then 8000.
PORT="${RAD_INTEL_PORT:-${PORT:-8000}}"

# One worker. The rate limits and the one-training-run-at-a-time lock are
# counted inside a single process; two workers would be two separate sets of
# both, so a limit would be twice as generous as it reads.
exec caffeinate -i -s \
  .venv/bin/python -m uvicorn rad_intel.api.main:app \
    --host 127.0.0.1 --port "$PORT" --workers 1
