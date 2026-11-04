"""The Hermes agent lane: one delegate tool call becomes a run on the api_server.

Hard rule taken from the voice domain: the relay pumps audio on this loop, so a tool that
waits is a microphone that stops. Nothing here blocks the caller — submission returns a
receipt instantly and the answer arrives later, spoken, from a worker thread.

Statuses are four named outcomes, never a bare exception, because the panel has to say
which one happened: ``ok`` (usable), ``absent`` (no server reachable), ``unauthorized``
(the key is wrong — a live server that refused us), ``error`` (it answered badly).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import httpx

from . import config

logger = logging.getLogger(__name__)

DEFAULT_API_SERVER_URL = "http://127.0.0.1:8642"
RUNS_PATH = "/v1/runs"
CAPABILITIES_PATH = "/v1/capabilities"

#: Read from core rather than guessed: ``api_server_run_idempotency.py:19``. ``interrupted``
#: is what a run becomes when the gateway restarts under it
#: (``api_server_runs.py:409-412``) — missing it here used to mean the voice lane waited the
#: full deadline for an answer that would never come.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
#: A live run that has stopped moving until a human answers it: the agent hit a command that
#: needs approval (``api_server_runs.py:857`` parks the status there with the redacted
#: ``approval`` event attached).
NEEDS_INPUT = "waiting_for_approval"
#: Approval choices the runs API accepts (``api_server_runs.py:1185``); a room-scoped grant
#: narrows this to once|deny, which is not our case — we are the submitting session.
APPROVAL_CHOICES = ("once", "session", "always", "deny")

STATUS_OK = "ok"
STATUS_ABSENT = "absent"
STATUS_UNAUTHORIZED = "unauthorized"
STATUS_ERROR = "error"

#: Spoken output has to be short enough to say aloud, and a 40 KB tool dump would end
#: the call as monotone reading. The tail is kept because that is where answers live.
MAX_SPEAKABLE_CHARS = 4000
POLL_BACKOFF_S = (1.0, 2.0, 4.0)
MAX_RUN_SECONDS = 900.0

#: Four, not one: a task parked on an approval holds its worker while it waits (the poll
#: loop is what notices the approval and speaks it), so a single unanswered task must not be
#: able to lock the lane.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gml-agent-lane")


class LaneUnavailable(RuntimeError):
    """The lane cannot take work; the reason is already speakable."""


def base_url(profile: str | None = None) -> str:
    """The api_server origin, routed to ``profile`` when one is named.

    A running gateway is multiplexed: the default listener answers
    ``/p/<profile>/v1/...`` for every served profile (``api_server.py:1174,4452``), and an
    UNPREFIXED run resumes in the DEFAULT profile's store — core says so outright at
    ``api_server_runs.py:559-562``. So a named profile is not a nicety, it is what keeps
    one profile's work out of another's memory.
    """
    raw = (os.environ.get("GEMINI_LIVE_API_SERVER_URL") or "").strip()
    root = (raw or DEFAULT_API_SERVER_URL).rstrip("/")
    if not profile or profile == "default":
        return root
    return f"{root}/p/{quote(profile, safe='')}"


@contextmanager
def _foreign_profile_scope(profile: str):
    """Bind another profile's secret scope for the duration of one read.

    The gateway's shared listener authenticates ``/p/<profile>/`` with that profile's own
    key, and one process serves every profile, so ``os.environ`` would answer with the
    launch profile's credential for all of them.
    """
    try:
        from agent.secret_scope import (build_profile_secret_scope, reset_secret_scope,
                                        set_secret_scope)
        from hermes_cli.profiles import get_profile_dir
        home = get_profile_dir(profile)
    except Exception:
        yield None
        return
    token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    try:
        yield home
    finally:
        reset_secret_scope(token)


def key(profile: str | None = None) -> str | None:
    """Bearer key for ``profile``, or None to send no Authorization.

    Set-but-blank is a deliberate opt-out, not a miss: the api_server accepts
    unauthenticated requests when it holds no key of its own, so "no key" is a real
    configuration rather than a broken one.
    """
    from .config import serving_profile
    own = serving_profile()
    if not profile or profile == own or not own:
        return _read_key()
    with _foreign_profile_scope(profile):
        return _read_key()


def _read_key() -> str | None:
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
        raw = get_secret("API_SERVER_KEY")
    except ImportError:
        raw = os.environ.get("API_SERVER_KEY")
    except UnscopedSecretError:
        raw = os.environ.get("API_SERVER_KEY")
    if raw is None:
        return None
    return raw.strip() or None


def headers(session_key: str | None = None, profile: str | None = None) -> dict:
    out = {}
    secret = key(profile)
    if secret:
        out["Authorization"] = f"Bearer {secret}"
    if session_key:
        out["X-Hermes-Session-Key"] = session_key
    return out


def probe(session_key: str | None = None, profile: str | None = None) -> tuple[str, str]:
    """(status, detail) for whether this run lane can be offered to the model.

    The refusal names the home to fix, because a multiplexed gateway authorises
    ``/p/<profile>/`` with that profile's own key and deliberately does not inherit the
    owner's (``api_server.py:1530-1545``) — so "set API_SERVER_KEY" without a location
    sends people to the wrong ``.env``.
    """
    if not key(profile):
        return STATUS_UNAUTHORIZED, (
            "no API_SERVER_KEY configured"
            if not profile
            else f"no API_SERVER_KEY for profile '{profile}' — set it in that profile's .env"
        )
    try:
        response = httpx.get(f"{base_url(profile)}{CAPABILITIES_PATH}",
                             headers=headers(session_key, profile), timeout=8.0)
    except httpx.HTTPError as exc:
        return STATUS_ABSENT, f"{type(exc).__name__}"
    if response.status_code in (401, 403):
        return STATUS_UNAUTHORIZED, f"the api server refused our key ({response.status_code})"
    if response.status_code != 200:
        return STATUS_ERROR, f"the api server answered {response.status_code}"
    try:
        features = (response.json().get("features") or {})
    except (ValueError, AttributeError):
        return STATUS_ERROR, "the api server returned a non-JSON capability list"
    if not features.get("run_submission"):
        return STATUS_ERROR, "this api server does not allow run submission"
    return STATUS_OK, ""


def submit(prompt: str, session_id: str | None = None,
           session_key: str | None = None, profile: str | None = None) -> str:
    """POST /v1/runs. Returns the run id; accepts any 2xx, not only 202.

    A refused *resume* is not a refused task. The api_server only admits a run into a
    session its own principal owns (``api_server_response_admissions.py`` answers
    ``permission_denied`` otherwise — surfaced as HTTP 409 through
    ``store_error_response``'s default). A call session created through
    ``POST /api/sessions`` is not such a session, so the run is retried once without the
    address: the work lands in its own session, which is what the user asked for in the
    first place. Losing the bookkeeping is cheaper than losing the task.
    """
    body: dict = {"input": prompt}
    if session_id:
        body["session_id"] = session_id
    try:
        response = httpx.post(f"{base_url(profile)}{RUNS_PATH}", json=body,
                              headers=headers(session_key, profile), timeout=30.0)
    except httpx.HTTPError as exc:
        raise LaneUnavailable(f"I couldn't reach the Hermes api server ({type(exc).__name__})") from exc
    if response.status_code // 100 != 2:
        detail = response.text[:200] or "no detail"
        if session_id and response.status_code in (403, 409) and "permission" in detail.lower():
            logger.info("hermes-gemini-live: the api server refused to resume %s (%s); "
                        "submitting the task in its own session", session_id, response.status_code)
            return submit(prompt, session_id=None, session_key=session_key, profile=profile)
        raise LaneUnavailable(f"the Hermes api server refused the run ({response.status_code}): "
                              f"{detail}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise LaneUnavailable("the Hermes api server returned a run that was not JSON") from exc
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise LaneUnavailable("the Hermes api server returned a run with no id")
    return run_id


def fetch(run_id: str, session_key: str | None = None,
          profile: str | None = None) -> dict:
    try:
        response = httpx.get(f"{base_url(profile)}{RUNS_PATH}/{run_id}",
                             headers=headers(session_key, profile), timeout=30.0)
    except httpx.HTTPError as exc:
        raise LaneUnavailable(f"I lost contact with the Hermes api server ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise LaneUnavailable(f"the Hermes api server answered {response.status_code} for that run")
    payload = response.json()
    if not isinstance(payload, dict):
        raise LaneUnavailable("the Hermes api server returned an invalid run status")
    return payload


def _control(run_id: str, verb: str, body: dict | None = None, *, session_key: str | None = None,
             profile: str | None = None) -> dict:
    """POST one of /approval, /steer, /stop for a run we started.

    Ownership is by idempotency scope, and the submitting session key IS that scope
    (``api_server_runs.py:1023-1028`` → ``_request_owns_run`` compares owner to
    ``_run_idempotency_scope(request)``). So every control call here must carry the same
    session key the run was submitted with, or the server answers 404 run-not-found.
    """
    url = f"{base_url(profile)}{RUNS_PATH}/{run_id}/{verb}"
    try:
        response = httpx.post(url, json=body or {}, headers=headers(session_key, profile),
                              timeout=30.0)
    except httpx.HTTPError as exc:
        raise LaneUnavailable(f"I couldn't reach the Hermes api server ({type(exc).__name__})") from exc
    if response.status_code // 100 != 2:
        detail = (response.json().get("error") or {}).get("message") if _is_json(response) else None
        raise LaneUnavailable(f"the Hermes api server refused to {verb} that run "
                              f"({response.status_code}): {detail or response.text[:160] or 'no detail'}")
    return _json_object(response)


def _is_json(response) -> bool:
    try:
        response.json()
        return True
    except (ValueError, AttributeError):
        return False


def _json_object(response) -> dict:
    try:
        value = response.json()
    except (ValueError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def approve(run_id: str, choice: str, request_id: str | None = None, *,
            session_key: str | None = None, profile: str | None = None) -> dict:
    """Resolve a pending approval. ``choice`` is one of APPROVAL_CHOICES; the request id is
    carried when the run parked more than one."""
    if choice not in APPROVAL_CHOICES:
        raise LaneUnavailable(f"'{choice}' is not an approval choice ({', '.join(APPROVAL_CHOICES)})")
    body: dict = {"choice": choice}
    if request_id:
        body["request_id"] = request_id
    return _control(run_id, "approval", body, session_key=session_key, profile=profile)


def steer(run_id: str, text: str, *, session_key: str | None = None,
          profile: str | None = None) -> dict:
    """Guide a run that is still working. Only a ``running`` run takes it
    (``api_server_runs.py:1228-1231`` answers 409 otherwise)."""
    if not (text or "").strip():
        raise LaneUnavailable("there was nothing to say to that run")
    return _control(run_id, "steer", {"input": text}, session_key=session_key, profile=profile)


def stop(run_id: str, *, session_key: str | None = None, profile: str | None = None) -> dict:
    return _control(run_id, "stop", None, session_key=session_key, profile=profile)


def speakable(text: str) -> str:
    """Prose the model can read aloud: no markdown furniture, no bare paths, no code."""
    if not text:
        return ""
    body = text if len(text) <= MAX_SPEAKABLE_CHARS else text[-MAX_SPEAKABLE_CHARS:]
    body = re.sub(r"```.*?```", " (code omitted) ", body, flags=re.S)
    body = re.sub(r"`([^`]*)`", r"\1", body)
    body = re.sub(r"^#{1,6}\s*", "", body, flags=re.M)
    body = re.sub(r"[*_]{1,3}", "", body)
    body = re.sub(r"[A-Za-z]:\\\\[^\s]+", "a file path", body)
    body = re.sub(r"(?<!\w)(?:/[\w.-]){2,}/", "a file path", body)
    body = re.sub(r"\s+", " ", body)
    return body.strip()[:MAX_SPEAKABLE_CHARS]


def await_result(run_id: str, session_key: str | None = None,
                 deadline_seconds: float = MAX_RUN_SECONDS,
                 profile: str | None = None, on_needs_input=None) -> tuple[str, str]:
    """Poll to a terminal status. Blocking — call it only from a worker thread.

    ``on_needs_input(run_id, question)`` fires once per parked approval, so the voice lane can
    ask the user about it while the run sits still. The approval event carries the flagged
    command already redacted server-side (``api_server.py:113-126``), which is the whole
    reason it is safe to speak.
    """
    started = time.monotonic()
    attempt = 0
    asked: set[str] = set()
    while True:
        status_row = fetch(run_id, session_key=session_key, profile=profile)
        state = str(status_row.get("status") or "")
        if state in TERMINAL_STATUSES:
            if state == "completed":
                return state, speakable(str(status_row.get("output") or ""))
            if state == "interrupted":
                # Adopted from hermes-live-voice's task contract: the gateway restarted
                # under the run, so the outcome is unknowable. Claiming failure would be
                # as wrong as claiming success.
                return "unknown", ("Hermes was restarted while that task was running, so its "
                                   "outcome is unknown — ask me to check the session.")
            detail = status_row.get("error") or state
            return state, speakable(str(detail))
        if state == NEEDS_INPUT and on_needs_input is not None:
            approval = status_row.get("approval") or {}
            request_id = str(approval.get("request_id") or "")
            if request_id not in asked:
                asked.add(request_id)
                on_needs_input(run_id, _approval_question(approval), request_id)
        if time.monotonic() - started > deadline_seconds:
            return "timed_out", f"Hermes was still working after {int(deadline_seconds)} seconds."
        time.sleep(POLL_BACKOFF_S[min(attempt, len(POLL_BACKOFF_S) - 1)])
        attempt += 1


def _approval_question(approval: dict) -> str:
    """The approval event in one speakable line, with the command last."""
    command = str(approval.get("command") or "").strip()
    reason = str(approval.get("reason") or approval.get("message") or "").strip()
    if command and reason:
        return f"{reason} It wants to run: {command}"
    return reason or (f"It wants to run: {command}" if command else "Hermes needs your approval")


class RunBook:
    """The board for one voice call: every task it started, what it was, and where it stands.

    Kept across a call so the model can be asked "what happened with the other one" without
    re-running it, and so a task the model addressed by id can be resolved back to its run.

    Durability, adopted from hermes-live-voice's task receipts: the board is written to disk
    after every change, because the things that outlive a socket are exactly the things the
    user asks about when they call back. A lost call loses the microphone, not the record of
    what was started.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._path = Path(path) if path else None

    def _save(self) -> None:
        """Write the board out. Best-effort: a full disk must not fail a call."""
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"tasks": list(self._tasks.values()), "at": time.time()}
            handle, tmp = tempfile.mkstemp(prefix="board-", suffix=".json",
                                           dir=str(self._path.parent))
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.info("hermes-gemini-live: board not saved (%s)", type(exc).__name__)

    def register(self, task_id: str, prompt: str) -> None:
        with self._lock:
            self._tasks[task_id] = {"id": task_id, "prompt": prompt[:120], "run_id": None,
                                    "state": "running", "question": "", "result": "",
                                    "request_id": ""}
            self._save()

    def run_id(self, task_id: str) -> str | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task["run_id"] if task else None

    def request_id(self, task_id: str) -> str:
        """The approval request a parked task waits on, when the server named one."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task["request_id"] if task else ""

    def attach_run(self, task_id: str, run_id: str) -> None:
        """Record the run behind a task — register() happens before submit, so it can't."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task["run_id"] = run_id
                self._save()

    def set_state(self, task_id: str, state: str, *, question: str = "",
                  result: str = "", request_id: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task["state"] = state
            if question:
                task["question"] = question
            if result:
                task["result"] = result[:MAX_SPEAKABLE_CHARS]
            if request_id:
                task["request_id"] = request_id
            self._save()

    def pending(self) -> list[str]:
        """Task ids still open — working or waiting on someone."""
        with self._lock:
            return sorted(key for key, task in self._tasks.items()
                          if task["state"] in ("queued", "running", "needs_input"))

    def board(self) -> str:
        """One line per task, oldest first — the text the model reads when it asks."""
        with self._lock:
            rows = list(self._tasks.values())
        if not rows:
            return "No tasks have been started in this call."
        return "\n".join(
            f"#{task['id']} [{task['state']}] {task['prompt']}"
            + (f" — waiting on you: {task['question']}" if task["state"] == "needs_input" else "")
            + (f" — {task['result'][:200]}" if task["state"] == "completed" and task["result"] else "")
            for task in rows)


class _Scheduler:
    """Admission for delegated runs: one at a time unless the operator allowed overlap.

    Adopted from hermes-live-voice, which defaults to ``exclusive`` for anything with side
    effects — two agents editing one repository is not a feature. The queue is in memory
    like the runs themselves; a body that survives a restart is the board on disk, not this.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque = deque()
        self._running = 0

    def admit(self, begin) -> bool:
        with self._lock:
            try:
                policy = config.concurrency()
            except Exception:
                policy = "exclusive"
            start_now = policy == "parallel" or self._running == 0
            if start_now:
                self._running += 1
            else:
                self._queue.append(begin)
        if start_now:
            begin()
        return start_now

    def finished(self) -> None:
        with self._lock:
            self._running = max(0, self._running - 1)
            following = self._queue.popleft() if (self._queue and self._running == 0) else None
            if following is not None:
                self._running += 1
        if following is not None:
            following()

    def queued(self) -> int:
        return len(self._queue)


_SCHEDULER = _Scheduler()


def start_task(prompt: str, on_finished, session_key: str | None = None,
               book: RunBook | None = None, profile: str | None = None,
               task_id: str | None = None, on_needs_input=None,
               session_id: str | None = None, on_run=None) -> str:
    """Hand a task to the pool and return immediately.

    ``on_finished(task_id, state, text)`` is called later from a worker thread, so the
    caller owns whatever it takes to get that result back onto its event loop. ``task_id`` is
    passed by the relay as the model's own call id, so the id the model hears in the receipt
    is the id it addresses later — one name per task, not two.

    ``session_id`` resumes that stored session instead of minting one per run: the call's
    own transcript is where the work belongs when the panel asks for it, and it is what
    lets a later task see what an earlier one already did.
    """
    task_id = task_id or uuid.uuid4().hex[:8]
    if book is not None:
        book.register(task_id, prompt)

    def begin() -> None:
        if book is not None:
            book.set_state(task_id, "running")

        def worker() -> tuple[str, str]:
            try:
                run_id = submit(prompt, session_id=session_id, session_key=session_key,
                                profile=profile)
                if book is not None:
                    book.attach_run(task_id, run_id)
                if on_run is not None:
                    # Called from this worker thread, once the run exists: the relay uses it
                    # to learn the session the api_server created for the first task.
                    try:
                        on_run(run_id)
                    except Exception:
                        logger.debug("hermes-gemini-live: on_run hook failed", exc_info=True)
                return await_result(run_id, session_key=session_key, profile=profile,
                                    on_needs_input=_parked(book, task_id, on_needs_input))
            except LaneUnavailable as exc:
                return "unavailable", speakable(str(exc))

        future = _POOL.submit(worker)

        def deliver() -> None:
            try:
                state, text = future.result()
                if book is not None:
                    book.set_state(task_id, state, result=text)
                on_finished(task_id, state, text)
            finally:
                # Releases the next queued task, whatever happened to this one.
                _SCHEDULER.finished()

        watcher = threading.Thread(target=deliver, name=f"gml-deliver-{task_id}", daemon=True)
        future.add_done_callback(lambda _f: watcher.start())

    if not _SCHEDULER.admit(begin) and book is not None:
        book.set_state(task_id, "queued")
    return task_id


def _parked(book: RunBook | None, task_id: str, on_needs_input):
    """Wrap the caller's callback so the board learns the task is waiting, not just the model."""
    def note(run_id: str, question: str, request_id: str) -> None:
        if book is not None:
            book.set_state(task_id, "needs_input", question=question, request_id=request_id)
        if on_needs_input is not None:
            on_needs_input(task_id, question, request_id)

    return note
