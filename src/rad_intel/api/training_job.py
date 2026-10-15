"""
The single fine-tuning run.

There is one shared model, so there is one training run at a time. Two
reviewers starting a run at the same moment would have them competing for the
same GPU and both writing a new version from the same export, and whichever
finished second would silently be the one that counted.

So a run is a claim. The clinician who starts it holds it until it ends, and
while it is held:

* nobody else can start one, or write an export -- an export freezes records,
  and freezing them underneath a run that is reading them changes what is
  being trained on halfway through;
* everybody else can still upload cases and correct their own, because none of
  that touches the manifest the run already has;
* everybody can see that a run is going and who is holding it.

Starting and cancelling a run belong to the administrator. Clinicians upload,
correct and export; a run costs hours of machine time and freezes everybody's
exports, so it is not something to hand out. They can still see that a run is
going and who is holding it, because it is their cases being trained on.

It cancels itself if the holder stops reporting in. That is what "a network
problem should not leave a run orphaned" actually means in a system where
training happens on the server: the holder's connection cannot interrupt the
computation, but if their browser has gone away there is nobody watching the
result, and a run nobody is watching should not keep the queue locked.

None of this touches what the public tool serves. Training runs in its own
process while the web server keeps answering from the model it already has
loaded, so a run in progress is invisible to visitors except as slightly
slower predictions while the processor is busy.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How long the holder's browser may go quiet before the run is abandoned. The
# interface polls every few seconds, so this is many missed polls, not one.
#
# Thirty minutes, not three. Three was right when any clinician could start a
# run and the browser that did was the only thing watching it; now only the
# administrator can, and an administrator who switches tab, sleeps a laptop or
# loses wifi for a few minutes should not come back to a cancelled run and a
# wasted hour of compute. The watchdog still exists, because a run nobody is
# ever coming back to should not hold the queue for ever.
HEARTBEAT_GRACE_SECONDS = 30 * 60
_WATCHDOG_INTERVAL = 30


class TrainingJob:
    """Coordinates the one run that may be in progress."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._watchdog: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "running": False,
            "started_at": None,
            "finished_at": None,
            "manifest": None,
            "cases": 0,
            "epochs": 0,
            "holder_id": None,
            "holder_name": None,
            "holder_institution": None,
            "cancelled_by": None,
            "cancel_reason": None,
            "exit_code": None,
        }
        self._last_heartbeat = 0.0

    # --- state ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    def heartbeat(self, user_id: str) -> None:
        """Called when the holder asks for status. Only the holder's poll counts."""
        with self._lock:
            if self._state["running"] and self._state["holder_id"] == user_id:
                self._last_heartbeat = time.monotonic()

    def may_cancel(self, user: dict) -> bool:
        """The holder, or an administrator."""
        with self._lock:
            return bool(
                self._state["running"]
                and (self._state["holder_id"] == user["id"] or user.get("role") == "admin")
            )

    # --- lifecycle --------------------------------------------------------

    def claim(self, user: dict, manifest: Path, cases: int, epochs: int) -> bool:
        """Take the run. Returns False when somebody already holds it."""
        with self._lock:
            if self._state["running"]:
                return False
            self._state.update(
                running=True,
                started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=None,
                manifest=manifest.name,
                cases=cases,
                epochs=epochs,
                holder_id=user["id"],
                holder_name=user["full_name"],
                holder_institution=user.get("institution"),
                cancelled_by=None,
                cancel_reason=None,
                exit_code=None,
            )
            self._last_heartbeat = time.monotonic()
        return True

    def release(self, exit_code: int | None = None) -> None:
        with self._lock:
            self._state.update(
                running=False,
                finished_at=datetime.now(timezone.utc).isoformat(),
                exit_code=exit_code,
            )
            self._process = None

    def start(self, script: Path, base_dir: Path, manifest: Path, base_model: str,
              epochs: int) -> None:
        """Launch the fine-tuning script and watch it in the background."""
        command = [sys.executable, "-u", str(script), "--manifest", str(manifest),
                   "--base-model", base_model, "--epochs", str(epochs)]
        threading.Thread(target=self._run, args=(command, base_dir), daemon=True).start()
        self._ensure_watchdog()

    def _run(self, command: list[str], base_dir: Path) -> None:
        exit_code: int | None = None
        try:
            with self.log_path.open("w", encoding="utf-8") as log:
                log.write(f"$ {' '.join(command)}\n\n")
                log.flush()
                process = subprocess.Popen(
                    command, cwd=base_dir, stdout=log, stderr=subprocess.STDOUT,
                )
                with self._lock:
                    self._process = process
                exit_code = process.wait()
        except Exception as exc:
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nTraining process failed to run: {exc}\n")
            logger.exception("Fine-tuning run could not be started.")
        finally:
            snapshot = self.snapshot()
            if snapshot["cancel_reason"]:
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n--- cancelled: {snapshot['cancel_reason']} ---\n"
                              f"No new model version was written. The current model is "
                              f"untouched and the export is still there to retry.\n")
            self.release(exit_code)

    def cancel(self, reason: str, by: str | None = None) -> bool:
        """Stop the run. Returns False when there is nothing to stop."""
        with self._lock:
            if not self._state["running"]:
                return False
            self._state.update(cancelled_by=by, cancel_reason=reason)
            process = self._process

        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
            # The thread had not reached Popen yet, or it is already gone.
            self.release(None)
        return True

    # --- watchdog ---------------------------------------------------------

    def _ensure_watchdog(self) -> None:
        with self._lock:
            if self._watchdog and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(target=self._watch, daemon=True)
            self._watchdog.start()

    def _watch(self) -> None:
        while True:
            time.sleep(_WATCHDOG_INTERVAL)
            with self._lock:
                if not self._state["running"]:
                    return
                quiet_for = time.monotonic() - self._last_heartbeat
            if quiet_for > HEARTBEAT_GRACE_SECONDS:
                logger.warning(
                    "Cancelling the fine-tuning run: the browser that started it has "
                    "not reported in for %.0f seconds.", quiet_for,
                )
                self.cancel(
                    reason=("the connection to the clinician who started it was lost "
                            f"(no contact for {int(quiet_for)} seconds)"),
                    by="watchdog",
                )
                return
