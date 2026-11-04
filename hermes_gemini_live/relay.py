"""Pump one browser WebSocket against one Live socket, and hand real work to Hermes.

Two tasks own the sockets, and either side dying closes both: a call with no browser is
a metered Live session with no listener.

The agent lane obeys one rule above all others: nothing about a run may be awaited on the
loop that carries audio. A tool call therefore answers the model with a receipt
immediately and the result arrives later, from a worker thread, as a spoken-note — the
same shape the voice domain learned a waiting tool is a dead microphone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from . import agent_lane, config, session_log, transcript, usage as usage_module, wire, tools as tools_module
from .config import ConfigError
from .live_client import LiveError, LiveSession

logger = logging.getLogger(__name__)

#: How many relays are live right now. The reload endpoint refuses while one is: swapping
#: module state under a running call is exactly the crash a restart used to avoid.
_active_calls = 0


def active_calls() -> int:
    return _active_calls


#: Audio frames arrive ~10/s; anything bigger than this is a bug or an attack.
MAX_BROWSER_FRAME_BYTES = 1 << 20
#: Forwarded, then the call ends — waiting for the socket to die on its own would only
#: turn an announced goodbye into a silent one.
TERMINAL_FRAME_TYPES = frozenset({"go_away"})
RESULT_NOTE = (
    "Hermes finished task #{task_id} ({state}). Result: {text}\n"
    "Tell the user what you found: the answer in your first sentence, then at most four more "
    "short spoken sentences. The full report stays in the Hermes session, so offer to answer "
    "questions about it instead of reading everything. Do not invent details, do not add "
    "commentary, and do not start another task about it."
)
NEEDS_NOTE = (
    "Task #{task_id} has stopped and needs the user: {question}\n"
    "Tell the user about it now in one short spoken sentence: only they can approve it, with "
    "the Approve button on the Gemini Live control. You cannot approve it. If they say no, "
    "call hermes_task_update on task {task_id} with action deny. Never start a second task "
    "about this one."
)
#: How long the model must have been silent before a note reaches it. Injecting a finished
#: result into the middle of its own speech makes it stop the sentence it was saying and read
#: the report instead, which is what the user heard as being cut off.
QUIET_BEFORE_NOTE_S = 1.0


@dataclass
class CallContext:
    """What this call carries besides its two sockets: the words, and where they go.

    ``transcript`` always exists (it is what a delegated task travels with).
    ``log`` only has a session id when the configured depth writes rows or the
    delegation target is the call's own session — a call that records nothing must not
    create an empty session in the user's sidebar.
    """

    transcript: "transcript.Transcript"
    log: "session_log.CallLog"
    mode: str = "delegations"
    target: str = "call_session"
    include_context: bool = True
    usage: "usage_module.Usage | None" = None
    #: ``call_session`` mode: no session is minted up front (the server refuses to resume
    #: one it did not create). The first delegated run's session becomes the call's.
    adopt_first: bool = False

    @property
    def writes_rows(self) -> bool:
        return session_log.wanted(self.mode)

    @property
    def call_session_id(self) -> str | None:
        return self.log.session_id

    def flush(self) -> int:
        """Write the utterances that settled since the last flush. Never raises."""
        if not self.writes_rows:
            return 0
        return self.log.append(self.transcript.pending_rows())

    def note(self, text: str) -> None:
        if self.mode == "full":
            self.log.note(text)

    def prompt_with_context(self, task: str) -> str:
        """A task, prefixed with what was just said — only when the rows are not already
        in the session the run resumes, so the worker never reads the same words twice.
        """
        if not self.include_context or self.writes_rows:
            return task
        window = transcript.labelled_block(self.transcript.context())
        return f"{window}\n\n{task}" if window else task



class NoteQueue:
    """Model-facing notes, released only in the gaps between the model's utterances.

    One at a time and first-in-first-out: several tasks can finish during a single long
    answer, and dumping all of them at once turns the report into noise. A note waiting on
    an approval is urgent, because the run behind it is standing still.
    """

    def __init__(self, live: LiveSession, clock=None) -> None:
        self._live = live
        # Injectable rather than patching ``time.monotonic``: that module is the event
        # loop's own clock, so a test that freezes it freezes asyncio with it.
        self._clock = clock or time.monotonic
        self._notes: deque[str] = deque()
        self._spoken_at = 0.0

    def speaking(self) -> None:
        """Stamp the model as busy; called for every audio frame forwarded."""
        self._spoken_at = self._clock()

    def push(self, text: str, *, urgent: bool = False) -> None:
        (self._notes.appendleft if urgent else self._notes.append)(text)

    def waiting(self) -> int:
        return len(self._notes)

    async def release_quietly(self) -> bool:
        """Hand over one note if the model has stopped speaking. True when it did.

        Popped only after the send succeeds: a socket that dies mid-delivery must not eat
        the result, because the pump tries again on its next wake-up.
        """
        if not self._notes or self._clock() - self._spoken_at < QUIET_BEFORE_NOTE_S:
            return False
        await self._live.send_text(self._notes[0])
        self._notes.popleft()
        return True


async def _browser_to_live(browser: Any, live: LiveSession, note: dict,
                           book: agent_lane.RunBook | None = None,
                           session_key: str | None = None, profile: str | None = None,
                           ctx: CallContext | None = None) -> None:
    while True:
        message = await browser.receive_text()
        if len(message) > MAX_BROWSER_FRAME_BYTES:
            continue
        kind = _frame_type(message)
        if kind == "audio":
            await live.send_audio(_b64(message))
        elif kind == "text":
            await live.send_text(_text(message))
        elif kind == "approval" and book is not None:
            value = _loads(message)
            if ctx is not None:
                # A note is a SQLite write: off the loop that carries audio, always.
                await asyncio.to_thread(
                    ctx.note, f"Nutzer-Entscheidung zu Aufgabe #{value.get('id')}: {value.get('choice')}")
            await _user_approval(value, live, browser, book, session_key, profile)
        elif kind == "session" and ctx is not None:
            sid = str(_loads(message).get("id") or "").strip()
            if sid and ctx.log.adopt(sid):
                logger.info("hermes-gemini-live: transcript follows the open session %s", sid)
        elif kind == "delegate_result" and ctx is not None:
            await _delegate_finished(_loads(message), browser, book, notes)
        elif kind == "close":
            note["reason"] = "browser closed the call"
            return


async def _handle_tool_call(call: dict, live: LiveSession, browser: Any,
                            book: agent_lane.RunBook, session_key: str | None,
                            profile: str | None, notes: NoteQueue,
                            ctx: CallContext | None = None) -> None:
    """Receipt first, work in the pool. The only await here is a small local socket send.

    The rule is about not waiting on a *run* (seconds to minutes), not about never
    awaiting: a receipt frame is what lets the model keep talking, so it goes out now.
    """
    ctx = ctx or CallContext(transcript.Transcript(), session_log.CallLog(None))
    call_id = str(call.get("id") or "")
    name = call.get("name")
    args = call.get("args") or {}
    if not call_id:
        logger.warning("hermes-gemini-live: tool call with no id was dropped")
        return
    if name == tools_module.BOARD_NAME:
        board = book.board()
        await asyncio.to_thread(ctx.note, f"Task-Board gelesen:\n{board}")
        await live.send_function_response(call_id, board, name)
        return
    if name == tools_module.UPDATE_NAME:
        await _update_task(call_id, args, live, browser, book, session_key, profile)
        return

    task = tools_module.compose_prompt(args)
    if not task:
        await live.send_function_response(call_id, "No task text was supplied.", name)
        return

    loop = asyncio.get_running_loop()
    await live.send_function_response(
        call_id,
        f"WORK_STARTED {call_id}: Hermes is running it. Do not describe any result yet.",
        name)
    # The panel shows what the user asked for, not the shaping instruction appended for the
    # worker — otherwise every task row ends with a paragraph about voice output.
    await browser.send_text(_dump({"type": "task_started", "id": call_id,
                                   "prompt": str(args.get("task") or "").strip()[:200]}))
    await asyncio.to_thread(
        ctx.note, f"Aufgabe #{call_id} gestartet: {str(args.get('task') or '').strip()[:400]}")
    if ctx.target == "active_session":
        # The renderer owns this one: it submits into the chat that is open, exactly the way
        # the built-in voice lane does (session.activate + prompt.submit), and reports back
        # with a ``delegate_result`` frame. Nothing here touches the api_server, so the
        # admission rule that refuses hand-made sessions never applies.
        book.register(call_id, str(args.get("task") or ""))
        book.set_state(call_id, "running")
        await browser.send_text(_dump({
            "type": "delegate",
            "id": call_id,
            "prompt": str(args.get("task") or "").strip(),
            "context": ctx.transcript.context() if ctx.include_context else "",
        }))
        return
    agent_lane.start_task(
        ctx.prompt_with_context(task),
        on_finished=_make_finisher(notes, browser, loop),
        session_key=session_key,
        book=book,
        profile=profile,
        # One name per task: the id in this receipt is the id the model addresses later.
        task_id=call_id,
        on_needs_input=_make_parker(notes, browser, loop),
        session_id=ctx.call_session_id if ctx.target == "call_session" else None,
        on_run=lambda run_id: _adopt_run_session(ctx, run_id, browser, loop, session_key, profile),
    )


async def _update_task(call_id: str, args: dict, live: LiveSession, browser: Any,
                       book: agent_lane.RunBook, session_key: str | None,
                       profile: str | None) -> None:
    """Answer, redirect or abandon a task the model already started.

    These hit the api_server, so they go off-loop with ``to_thread``: the audio loop keeps
    running, only this one tool call waits — which is what the model asked for.
    """
    task = str(args.get("task") or "").strip().lstrip("#")
    action = str(args.get("action") or "").strip().lower()
    answer = str(args.get("answer") or "").strip()
    if action not in tools_module.ACTIONS:
        await live.send_function_response(
            call_id, f"BAD_REQUEST unknown action '{action}'; "
                     f"use {', '.join(tools_module.ACTIONS)}.")
        return
    run_id = book.run_id(task)
    if not run_id:
        await live.send_function_response(
            call_id, f"NO_SUCH_TASK {task}: it is not on this call's board. "
                     f"Call hermes_tasks to see what is.")
        return

    try:
        if action == "deny":
            agent_lane.approve(run_id, "deny",
                               book.request_id(task), session_key=session_key, profile=profile)
        elif action == "steer":
            agent_lane.steer(run_id, answer or "The user has no further instruction.",
                             session_key=session_key, profile=profile)
        else:
            agent_lane.stop(run_id, session_key=session_key, profile=profile)
    except agent_lane.LaneUnavailable as exc:
        # A refused control is news the model must say out loud — silently retrying would
        # leave the user believing their answer landed.
        book.set_state(task, "needs_input", question=str(exc))
        await live.send_function_response(call_id, f"UPDATE_FAILED {task}: {exc}")
        await browser.send_text(_dump({"type": "task_failed", "id": task,
                                       "detail": str(exc)[:200]}))
        return

    book.set_state(task, "running" if action == "steer" else "cancelled")
    await live.send_function_response(call_id, f"UPDATE_DONE {task} {action}.")
    await browser.send_text(_dump({"type": "task_update", "id": task, "action": action}))


async def _user_approval(value: dict, live: LiveSession, browser: Any,
                         book: agent_lane.RunBook, session_key: str | None,
                         profile: str | None) -> None:
    """The user pressed Approve or Deny on the Desktop control.

    This is the only path that can approve a parked run: it arrives on the authenticated
    relay socket from the renderer, never from a Live ``toolCall``, so nothing the model
    hears or reads can trigger it. Only ``once`` is ever granted.
    """
    task = str(value.get("id") or "").strip().lstrip("#")
    action = "approve" if value.get("choice") == "approve" else "deny"
    run_id = book.run_id(task)
    if not run_id:
        await browser.send_text(_dump({"type": "task_failed", "id": task,
                                       "detail": "that task is not on this call's board"}))
        return
    try:
        await asyncio.to_thread(agent_lane.approve, run_id,
                                "once" if action == "approve" else "deny",
                                book.request_id(task), session_key=session_key, profile=profile)
    except agent_lane.LaneUnavailable as exc:
        await browser.send_text(_dump({"type": "task_failed", "id": task,
                                       "detail": str(exc)[:200]}))
        return
    book.set_state(task, "running" if action == "approve" else "cancelled")
    await browser.send_text(_dump({"type": "task_update", "id": task, "action": action}))
    await live.send_text(f"The user pressed {action.capitalize()} on task #{task}. "
                         "Acknowledge it in a few words.")


def _adopt_run_session(ctx: CallContext, run_id: str, browser: Any,
                       loop: asyncio.AbstractEventLoop, session_key: str | None,
                       profile: str | None) -> None:
    """Adopt the session the api_server created for this call's first run.

    Runs on a worker thread (``on_run`` fires from the pool). Reading the run status is one
    small local request; the row flush and the panel frame go back to the loop.
    """
    if not ctx.adopt_first or ctx.log.session_id:
        return
    try:
        status = agent_lane.fetch(run_id, session_key=session_key, profile=profile)
    except agent_lane.LaneUnavailable as exc:
        logger.info("hermes-gemini-live: could not read the run's session (%s)", exc)
        return
    session_id = str(status.get("session_id") or "").strip()
    if not session_id or not ctx.log.adopt(session_id, session_log.call_title()):
        return
    logger.info("hermes-gemini-live: adopted the call session %s from run %s",
                session_id, run_id)

    async def announce() -> None:
        await browser.send_text(_dump({"type": "call_session", "id": session_id}))

    _hand_off(announce(), loop, "session", "adopted")


async def _delegate_finished(value: dict, browser: Any, book: agent_lane.RunBook,
                            notes: NoteQueue) -> None:
    """The renderer finished (or gave up on) a task it submitted into the open chat."""
    task_id = str(value.get("id") or "").strip()
    state = str(value.get("state") or "unknown").strip() or "unknown"
    text = str(value.get("text") or "").strip()
    book.set_state(task_id, state, result=text)
    notes.push(RESULT_NOTE.format(task_id=task_id, state=state, text=text or "no output"))
    await browser.send_text(_dump({"type": "task_done", "id": task_id, "state": state}))
    await _release(notes, task_id, "finished")


def _hand_off(coro, loop: asyncio.AbstractEventLoop, task_id: str, why: str) -> None:
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        # The loop is gone: the call ended while Hermes was still working. The run keeps
        # going and its news simply goes unspoken, which is said here rather than dying
        # quietly on a worker thread.
        logger.info("hermes-gemini-live: task %s %s after its call closed", task_id, why)


def _make_parker(notes: NoteQueue, browser: Any, loop: asyncio.AbstractEventLoop):
    """A worker-thread callback that queues the question a parked run is waiting on."""
    def on_needs_input(task_id: str, question: str, request_id: str) -> None:
        async def deliver() -> None:
            notes.push(NEEDS_NOTE.format(task_id=task_id, question=question), urgent=True)
            await browser.send_text(_dump({"type": "task_needs_input", "id": task_id,
                                           "question": question[:200]}))
            await _release(notes, task_id, "is waiting")

        _hand_off(deliver(), loop, task_id, "was waiting")

    return on_needs_input


def _make_finisher(notes: NoteQueue, browser: Any,
                   loop: asyncio.AbstractEventLoop):
    """A worker-thread callback that queues a finished result for the next gap in speech."""
    def on_finished(task_id: str, state: str, text: str) -> None:
        async def deliver() -> None:
            notes.push(RESULT_NOTE.format(task_id=task_id, state=state, text=text or "no output"))
            await browser.send_text(_dump({"type": "task_done", "id": task_id,
                                           "state": state}))
            await _release(notes, task_id, "finished")

        _hand_off(deliver(), loop, task_id, "finished")

    return on_finished


async def _release(notes: NoteQueue, task_id: str, why: str) -> None:
    """Try to speak a note straight away if the model happens to be quiet.

    A dead socket here is not a lost result: the note stays queued and the pump retries it
    on its next wake-up, so the only thing to report is that the early attempt failed.
    """
    try:
        await notes.release_quietly()
    except LiveError as exc:
        logger.info("hermes-gemini-live: task %s %s but the call is closing (%s)",
                    task_id, why, exc)


async def _live_to_browser(browser: Any, live: LiveSession, note: dict,
                           book: agent_lane.RunBook, session_key: str | None,
                           profile: str | None, notes: NoteQueue,
                           ctx: CallContext | None = None) -> None:
    ctx = ctx or CallContext(transcript.Transcript(), session_log.CallLog(None))
    while True:
        try:
            # The timeout is not a liveness check, it is the other half of the queue: a
            # result that landed mid-sentence has to be released once the model stops, and
            # frames only arrive while it is talking.
            raw = await asyncio.wait_for(live.recv(), timeout=QUIET_BEFORE_NOTE_S)
        except asyncio.TimeoutError:
            await notes.release_quietly()
            continue
        raw_usage = raw.get("usageMetadata")
        if isinstance(raw_usage, dict) and ctx.usage is not None:
            ctx.usage.tokens(raw_usage)
        for frame in wire.browser_frames(raw):
            kind = frame["type"]
            if kind == "usage" and ctx.usage is not None:
                frame["usd"] = ctx.usage.usd
                frame["unpriced"] = ctx.usage.unpriced
            if kind == "audio":
                notes.speaking()
            if kind == "in_text":
                ctx.transcript.add(transcript.USER, str(frame.get("text") or ""))
            elif kind == "out_text":
                ctx.transcript.add(transcript.ASSISTANT, str(frame.get("text") or ""))
            elif kind == "speech_started":
                ctx.transcript.interrupted()
            elif kind == "turn_complete":
                ctx.transcript.turn_complete()
                # A row write is SQLite: it goes off the loop that carries audio, and a
                # refusal inside it is already absorbed by the log.
                await asyncio.to_thread(ctx.flush)
            elif kind == "resumption":
                handle = str(frame.get("handle") or "")
                if handle:
                    note["handle"] = handle
            elif kind == "go_away":
                await _on_going_away(note, ctx, frame)
                if note.get("stop_now"):
                    await browser.send_text(_dump(frame))
                    return
            await browser.send_text(_dump(frame))
            if kind == "error":
                note["reason"] = frame["detail"]
                return
            if kind in TERMINAL_FRAME_TYPES:
                note["reason"] = f"server ended the session ({frame.get('timeLeft') or 'now'})"
                return
            if kind == "tool_call":
                for call in frame["calls"]:
                    await _handle_tool_call(call, live, browser, book, session_key,
                                            profile, notes, ctx)
        await notes.release_quietly()


async def _on_going_away(note: dict, ctx: CallContext, frame: dict) -> None:
    """The server announced the end of this session: keep, restart, or hang up.

    ``auto_restart`` is what makes a long call usable: Gemini hands out a resumption
    handle with every update, and re-opening with it continues the same conversation
    instead of making the user dial again.
    """
    time_left = str(frame.get("timeLeft") or "")
    mode = "keep"
    try:
        mode = config.long_call()
    except Exception:
        pass
    note["reason"] = f"server ended the session ({time_left or 'now'})"
    if mode == "hang_up":
        note["stop_now"] = True
    elif mode == "auto_restart" and note.get("handle") and not note.get("restarted"):
        note["stop_now"] = True
        note["restart"] = True
    logger.info("hermes-gemini-live: session ending in %s (mode=%s, handle=%s)",
                time_left or "now", mode, bool(note.get("handle")))


def _loads(message: str) -> dict:
    try:
        value = json.loads(message)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _frame_type(message: str) -> str:
    value = _loads(message)
    kind = value.get("type")
    if kind == "audio" and not str(value.get("data") or "").strip():
        return ""
    if kind == "text" and not str(value.get("text") or "").strip():
        return ""
    if kind == "approval" and not (str(value.get("id") or "").strip()
                                   and value.get("choice") in ("approve", "deny")):
        return ""
    if kind == "delegate_result" and not (str(value.get("id") or "").strip()):
        return ""
    if kind == "session" and not str(value.get("id") or "").strip():
        return ""
    return kind if kind in ("audio", "text", "approval", "close", "delegate_result",
                            "session") else ""


def _b64(message: str) -> str:
    return str(_loads(message).get("data") or "")


def _text(message: str) -> str:
    return str(_loads(message).get("text") or "")[:4000]


def _dump(frame: dict) -> str:
    return json.dumps(frame, separators=(",", ":"))


async def run_relay(browser: Any, session_key: str | None = None,
                     profile: str | None = None) -> None:
    """Own the whole call. The caller has already accepted the upgrade."""
    global _active_calls
    _active_calls += 1
    try:
        await _run_relay(browser, session_key=session_key, profile=profile)
    finally:
        _active_calls = max(0, _active_calls - 1)


async def _run_relay(browser: Any, session_key: str | None = None,
                     profile: str | None = None) -> None:
    note: dict = {"reason": ""}
    # The serving home decides the profile, not a query parameter a client could send:
    # an unprefixed run resumes in the DEFAULT profile's store, so this is the one place
    # that must be right for delegated work to land in the right memory.
    profile = profile or config.serving_profile()
    # Off the audio loop: a probe is a network round trip, and a slow gateway must not
    # cost the user a dropped syllable.
    lane_status, lane_detail = await asyncio.to_thread(agent_lane.probe, session_key, profile)
    offer_tools = lane_status == agent_lane.STATUS_OK
    if not offer_tools:
        logger.warning("hermes-gemini-live: agent lane unavailable (%s: %s)",
                       lane_status, lane_detail)
    # One read per call: a settings change mid-call must not reshape a running session.
    mode = session_log.mode_for()
    try:
        target = config.delegation_target()
    except Exception:
        target = "call_session"
    try:
        include_context = config.include_voice_context()
    except Exception:
        include_context = True
    ctx = CallContext(transcript=transcript.Transcript(), log=session_log.CallLog(None),
                      mode=mode, target=target, include_context=include_context)
    # A call that records nothing must not mint an empty session in the user's sidebar;
    # delegations need one when they are meant to land in it.
    # ``call_session`` waits for the run to hand us its session (see _adopt_run_session);
    # only a recording mode that is not adopting needs one up front.
    ctx.adopt_first = target == "call_session"
    mint_here = session_log.wanted(mode) and not ctx.adopt_first and target != "active_session"
    if mint_here and agent_lane.key():
        ctx.log = await asyncio.to_thread(session_log.CallLog.create, profile, session_key)
        if ctx.log.session_id:
            logger.info("hermes-gemini-live: call session %s (mode=%s, target=%s)",
                        ctx.log.session_id, mode, target)
            await asyncio.to_thread(ctx.log.rename, session_log.call_title())
            await browser.send_text(_dump({"type": "call_session", "id": ctx.log.session_id}))
    # Durable board (hermes-live-voice's receipts): the file outlives the socket, so a
    # reconnect — or simply the next morning — can still answer "what did I start?".
    book = agent_lane.RunBook(session_log.board_path(ctx.log.session_id)
                              if ctx.log.session_id else None)
    tools = tools_module.declarations() if offer_tools else None
    attempt = 0
    while True:
        try:
            live = await LiveSession.open(tools, handle=note.get("handle") if attempt else None)
        except (LiveError, ConfigError) as exc:
            logger.warning("hermes-gemini-live: call refused: %s", exc)
            await browser.send_text(_dump({"type": "error", "detail": str(exc)}))
            break
        if not offer_tools and attempt == 0:
            await browser.send_text(_dump({"type": "lane", "status": lane_status,
                                           "detail": lane_detail}))
        notes = NoteQueue(live)
        ctx.usage = usage_module.Usage(getattr(live, "model", "") or "")
        if attempt:
            await live.send_text("The connection was resumed. Say in one short sentence that "
                                 "you are back, then wait for the user.")
        up = asyncio.create_task(_browser_to_live(browser, live, note, book, session_key,
                                                  profile, ctx))
        down = asyncio.create_task(_live_to_browser(browser, live, note, book, session_key,
                                                    profile, notes, ctx))
        try:
            done, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if not error:
                    continue
                note["reason"] = str(error)
                if task is up:
                    # Which pump failed says who left. The browser pump only dies when the page
                    # goes away — a navigation, not a fault — and there is nobody to tell, so
                    # this is logged as an ordinary end. The panel used to receive an error
                    # frame here, which read as "the system failed" every time the user clicked
                    # somewhere else mid-call.
                    logger.info("hermes-gemini-live: call ended (%s from the browser side)",
                                type(error).__name__)
                    continue
                logger.warning("hermes-gemini-live: relay pump failed: %s: %s",
                               type(error).__name__, error)
                await browser.send_text(_dump({"type": "error", "detail": _safe(error),
                                               "ended": True}))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            # A pump that lost the race can finish with an exception nobody asked for — the
            # cancel lands after it already raised. Retrieving it here keeps asyncio from
            # printing a bare traceback that looks like a crash in the log.
            for task in (up, down):
                if task.done() and not task.cancelled():
                    task.exception()
            await live.close()
            # Whatever was said last still belongs in the session, even if the words came
            # in without a turnComplete (a hang-up mid-sentence).
            await asyncio.to_thread(ctx.flush)
            if ctx.usage is not None:
                await asyncio.to_thread(usage_module.record, ctx.usage.close())
                ctx.usage = None
            if notes.waiting():
                # Queued and never spoken: the result is real, the user simply hung up first.
                logger.info("hermes-gemini-live: call ended with %d result(s) still queued",
                            notes.waiting())
        restart = bool(note.get("restart")) and attempt == 0
        note.pop("restart", None)
        note.pop("stop_now", None)
        if restart:
            attempt = 1
            note["restarted"] = True
            logger.info("hermes-gemini-live: session resumed on a fresh socket")
            continue
        break
    if book.pending():
        logger.info("hermes-gemini-live: call ended with %d run(s) still going: %s",
                    len(book.pending()), ",".join(book.pending()))
    logger.info("hermes-gemini-live: call ended (%s)", note["reason"] or "ended")



def _safe(error: BaseException) -> str:
    text = str(error) or type(error).__name__
    return text if "key=" not in text else "the Gemini Live socket failed"
