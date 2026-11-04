"""The call as a Hermes session: one row per spoken utterance, and the id work lands in.

Two different doors, on purpose:

* **The session itself** is created through the api_server's own ``POST /api/sessions``,
  with ``source=desktop`` so the sidebar shows it in the ordinary recents list instead of
  the "API" section every delegated run used to fall into. No core change, no second
  board — the same door the Desktop app uses.
* **The rows** are appended through the session store, because there is no public
  "append a message" RPC. That is the one internal call in this plugin; it is wrapped so
  that a missing or moved symbol downgrades to "no transcript" with a warning, never to a
  failed call.

Everything here is best-effort by contract: the voice conversation must survive a store
that refuses to write.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import httpx

from . import agent_lane, config

logger = logging.getLogger(__name__)

SESSION_PREFIX = "geminilive_"
HTTP_TIMEOUT_S = 15.0


def board_path(session_id: str) -> Path:
    """Where this call's task board is written: one file per call, next to the settings."""
    from . import settings
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_") or "call"
    return settings._home() / "plugin-data" / "hermes-gemini-live" / "calls" / f"{safe}.json"


def _db_path() -> Path:
    """``state.db`` of the home this process serves."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "state.db"
    except Exception:
        import os
        raw = (os.environ.get("HERMES_HOME") or "").strip()
        return (Path(raw) if raw else Path.home() / ".hermes") / "state.db"


class CallLog:
    """One call's session, created lazily and written to opportunistically."""

    def __init__(self, session_id: str | None = None, profile: str | None = None) -> None:
        self.session_id = session_id
        self.profile = profile
        self._db = None
        self._db_failed = False
        self._warned = False
        # A session we do not own yet: the api_server only admits a run into a session its
        # own principal created, so in ``call_session`` mode the first delegated run makes
        # it and we adopt it. Rows said before that point wait here, in order.
        self._buffer: list[dict] = []

    # ── lifecycle ────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, profile: str | None = None,
               session_key: str | None = None) -> "CallLog":
        """Mint the call's session, or return an empty log when the lane is unavailable.

        The id is ours, not the server's: a call that reconnects (``auto_restart``) has
        to land in the same transcript, so the id is derived once and reused.
        """
        call_id = f"{SESSION_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        url = f"{agent_lane.base_url(profile)}/api/sessions"
        try:
            response = httpx.post(url, json={"id": call_id, "source": "desktop"},
                                  headers=agent_lane.headers(session_key, profile),
                                  timeout=HTTP_TIMEOUT_S)
        except httpx.HTTPError as exc:
            logger.info("hermes-gemini-live: no call session (%s); transcript stays unwritten",
                        type(exc).__name__)
            return cls(None, profile)
        if response.status_code // 100 != 2:
            logger.info("hermes-gemini-live: session create answered %s; transcript off",
                        response.status_code)
            return cls(None, profile)
        return cls(call_id, profile)

    def adopt(self, session_id: str, title: str | None = None) -> bool:
        """Take over a session the API created for us; returns False when we already have one.

        This is the honest version of "the work lands in the call's session": rather than
        minting a session the server will refuse to resume, the first run's session becomes
        the call's, and everything recorded before it (the speech that asked for it) is
        flushed into it.
        """
        session_id = (session_id or "").strip()
        if not session_id or self.session_id:
            return False
        self.session_id = session_id
        buffered, self._buffer = self._buffer, []
        if buffered:
            self.append(buffered)
        if title:
            self.rename(title)
        return True

    def rename(self, title: str) -> None:
        """Give the call a readable title. Best-effort, like every write here."""
        if not self.session_id:
            return
        url = f"{agent_lane.base_url(self.profile)}/api/sessions/{self.session_id}"
        try:
            httpx.patch(url, json={"title": title},
                        headers=agent_lane.headers(None, self.profile),
                        timeout=HTTP_TIMEOUT_S)
        except httpx.HTTPError:
            pass

    # ── rows ─────────────────────────────────────────────────────────────────
    def append(self, rows: list[dict]) -> int:
        """Write settled utterances. Returns how many landed; never raises.

        With no session yet the rows are kept in order instead of dropped: they belong to a
        conversation that is still being created.
        """
        if not rows:
            return 0
        if not self.session_id:
            self._buffer.extend(row for row in rows if str(row.get("content") or "").strip())
            return 0
        db = self._store()
        if db is None:
            return 0
        written = 0
        for row in rows:
            content = str(row.get("content") or "").strip()
            role = str(row.get("role") or "user")
            if not content or role not in ("user", "assistant"):
                continue
            try:
                db.append_message(self.session_id, role, content)
                written += 1
            except Exception as exc:  # a refused row is not a failed call
                if not self._warned:
                    logger.info("hermes-gemini-live: transcript write refused (%s); "
                                "continuing without it", type(exc).__name__)
                    self._warned = True
                break
        return written

    def note(self, text: str) -> None:
        """A control event (approval, board read, task start) as an assistant note."""
        body = (text or "").strip()
        if body:
            self.append([{"role": "assistant", "content": body}])

    def _store(self):
        if self._db is not None or self._db_failed:
            return self._db
        try:
            from hermes_state import SessionDB
            self._db = SessionDB(db_path=_db_path())
        except Exception as exc:
            self._db_failed = True
            logger.info("hermes-gemini-live: no session store (%s); transcript off",
                        type(exc).__name__)
            return None
        return self._db


def wanted(mode: str) -> bool:
    """Whether the configured transcript depth writes rows at all."""
    return mode in ("utterances", "full")


def call_title(stamp: str | None = None) -> str:
    when = stamp or time.strftime("%d.%m. %H:%M")
    return f"Sprachanruf {when}"


def mode_for(config_module=config) -> str:
    """Read the mode defensively: a broken setting must not stop a call from starting."""
    try:
        return config_module.transcript_mode()
    except Exception:
        return "delegations"
