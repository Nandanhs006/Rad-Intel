#!/bin/bash
#
# Start Rad-Intel. Used by both services in this folder -- systemd on a Linux
# server, launchd on a Mac -- and runnable by hand to see what it does.
#
# It deliberately does not daemonise. Both service managers want a process
# that stays in the foreground, so they can tell when it dies and start it
# again.

set -euo pipefail

# Where the project is. The service file sets this; the fallback is for
# running it by hand.
PROJECT="${RAD_INTEL_HOME:-$HOME/Rad-Intel}"
[ -d "$PROJECT" ] || PROJECT="$HOME/Documents/GitHub/Rad-Intel"

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

# Listen on the loopback only. On a server something in front of this -- Caddy
# -- holds the certificate and forwards to it, and on a laptop a tunnel does.
# Either way nothing should be able to reach this port from outside the
# machine, whatever the firewall happens to say.
#
# One worker. The rate limits and the one-training-run-at-a-time lock are
# counted inside a single process; two workers would be two separate sets of
# both, so every limit would be twice as generous as it reads.
SERVE=(.venv/bin/python -m uvicorn rad_intel.api.main:app
       --host 127.0.0.1 --port "$PORT" --workers 1)

# A Mac has to be told not to fall asleep underneath it. A server does not,
# and has no such command.
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i -s "${SERVE[@]}"
fi
exec "${SERVE[@]}"
