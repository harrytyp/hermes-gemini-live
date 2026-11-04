"""The relay owns two sockets and must tear both down when either side dies.

Nothing here touches Google: the Live side is a scripted peer, because what is asserted
is the plumbing and the shutdown, not the provider's behaviour. Each test ends through
exactly one pump, so no assertion depends on which task the loop happened to finish
first — a relay test that races its own two tasks proves nothing about either.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from hermes_gemini_live import agent_lane, relay
from hermes_gemini_live import tools as tools_module
from hermes_gemini_live.live_client import LiveError


class FakeLive:
    """A Live peer that either parks once its script is spent, or dies on purpose."""

    def __init__(self, incoming=(), *, end_after_script=True):
        self.incoming = list(incoming)
        self.audio_in: list[str] = []
        self.text_in: list[str] = []
        self.responses: list[dict] = []
        self.closed = False
        self._end_after_script = end_after_script

    async def send_audio(self, b64):
        self.audio_in.append(b64)

    async def send_text(self, text):
        self.text_in.append(text)

    async def send_function_response(self, call_id, result, name=None):
        self.responses.append({"id": call_id, "result": result, "name": name})

    async def recv(self):
        if self.incoming:
            await asyncio.sleep(0)
            return self.incoming.pop(0)
        if self._end_after_script:
            raise LiveError("the Gemini Live socket closed (1006)")
        await asyncio.sleep(3600)
        raise AssertionError("parked socket must be cancelled, not read to the end")

    async def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, incoming=()):
        self.incoming = [json.dumps(frame) for frame in incoming]
        self.sent: list[dict] = []
        self._parked = False

    async def receive_text(self):
        if self.incoming:
            await asyncio.sleep(0)
            return self.incoming.pop(0)
        self._parked = True
        await asyncio.sleep(3600)
        raise AssertionError("parked browser must be cancelled, not read to the end")

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))


def types(browser):
    return [frame["type"] for frame in browser.sent]


@pytest.fixture()
def opened(monkeypatch):
    """Bind a scripted Live peer to the session the relay opens, with the lane reachable.

    The relay probes the agent lane before it opens, so every test states that answer
    explicitly instead of hitting a real localhost port.
    """
    seen = {"tools": "unset", "probe": None}

    def fake_probe(session_key=None, profile=None):
        seen["probe"] = seen.get("probe") or (agent_lane.STATUS_OK, "")
        seen["profile"] = profile
        return seen["probe"]

    def install(live, *, lane=None):
        if lane is not None:
            seen["probe"] = lane

        class Stub:
            @staticmethod
            async def open(tools=None, handle=None):
                seen["tools"] = tools
                seen["handle"] = handle
                return live

        monkeypatch.setattr(relay, "LiveSession", Stub)
        monkeypatch.setattr(relay.agent_lane, "probe", fake_probe)
        return live

    install.seen = seen
    return install


@pytest.mark.asyncio
async def test_mic_and_type_frames_reach_the_live_socket(opened):
    # Parks after its script: the browser's own close is the only way this call can end.
    live = opened(FakeLive(end_after_script=False))
    browser = FakeBrowser([{"type": "audio", "data": "AAA="},
                           {"type": "text", "text": "hello"},
                           {"type": "close"}])
    await relay.run_relay(browser)
    assert live.audio_in == ["AAA="]
    assert live.text_in == ["hello"]
    assert live.closed is True


@pytest.mark.asyncio
async def test_live_events_are_relabelled_for_the_renderer(opened, monkeypatch):
    # This file tests the api_server lane; the renderer lane has its own tests.
    monkeypatch.setattr(relay.config, "delegation_target", lambda: "own_session")
    # This test is about labelling, so the run itself must not reach the network.
    monkeypatch.setattr(relay.agent_lane, "start_task", lambda *a, **k: "task-x")
    live = opened(FakeLive(incoming=[{"setupComplete": {}},
                                     {"serverContent": {"interrupted": True}},
                                     {"serverContent": {"turnComplete": True}},
                                     {"toolCall": {"functionCalls": [
                                         {"id": "c1", "name": "giao_viec_cho_hermes",
                                          "args": {"task": "list files"}}]}}]))
    browser = FakeBrowser([])
    await relay.run_relay(browser)
    # ...then the scripted socket ends, which is itself reported rather than swallowed.
    assert types(browser) == ["ready", "speech_started", "turn_complete", "tool_call",
                              "task_started", "error"]
    assert live.closed is True


@pytest.mark.asyncio
async def test_junk_and_blank_frames_never_reach_the_renderer(opened):
    opened(FakeLive(incoming=[{"someFutureField": {"a": 1}},
                              {"serverContent": {"modelTurn": {"parts": [{"text": "   "}]}}}]))
    browser = FakeBrowser([])
    await relay.run_relay(browser)
    # The only frame the renderer sees is the socket dying after the script; nothing the
    # server invented on its own gets forwarded as a message it cannot handle.
    assert types(browser) == ["error"]


@pytest.mark.asyncio
async def test_a_dead_live_socket_speaks_an_error_before_it_ends(opened):
    # Silence here would read to the user as a frozen app, which is the failure this
    # whole surface exists to avoid.
    live = opened(FakeLive(incoming=[]))
    browser = FakeBrowser([])
    await relay.run_relay(browser)
    assert "error" in types(browser)
    assert live.closed is True


@pytest.mark.asyncio
async def test_a_refused_open_reports_why_and_leaves_no_session(monkeypatch):
    from hermes_gemini_live.config import ConfigError

    class Refusing:
        @staticmethod
        async def open(tools=None, handle=None):
            raise ConfigError("no Gemini key for Live: set GEMINI_LIVE_API_KEY or GEMINI_API_KEY")

    monkeypatch.setattr(relay, "LiveSession", Refusing)
    browser = FakeBrowser([])
    await relay.run_relay(browser)
    assert browser.sent[0]["type"] == "error"
    assert "GEMINI_API_KEY" in browser.sent[0]["detail"]


@pytest.mark.asyncio
async def test_a_server_goodbye_ends_the_call_as_its_own_reason(opened):
    # Not an error: the session reached its cap, and calling that a failure would blame the
    # user's equipment for a documented limit.
    live = opened(FakeLive(incoming=[{"goAway": {"timeLeft": "30s"}}]))
    browser = FakeBrowser([])
    await relay.run_relay(browser)
    assert types(browser) == ["go_away"]
    assert live.closed is True


@pytest.mark.asyncio
async def test_a_tool_call_answers_with_a_receipt_and_starts_no_wait(opened, monkeypatch):
    monkeypatch.setattr(relay.config, "delegation_target", lambda: "own_session")
    started: list[tuple[str, object]] = []

    def fake_start_task(prompt, on_finished, session_key=None, book=None, profile=None,
                        task_id=None, on_needs_input=None, session_id=None, on_run=None):
        started.append((prompt, on_finished, profile, task_id, on_needs_input))
        return "task-1"

    monkeypatch.setattr(relay.agent_lane, "start_task", fake_start_task)
    live = opened(FakeLive(incoming=[
        {"toolCall": {"functionCalls": [{"id": "call_7", "name": "hermes_task",
                                         "args": {"task": "count the files in web/"}}]}}]))
    browser = FakeBrowser([])
    await relay.run_relay(browser)

    assert len(started) == 1
    prompt, on_finished, profile, task_id, on_needs_input = started[0]
    assert prompt.startswith("count the files")
    # The shaping instruction belongs to the run, not to the row the user reads.
    assert "read aloud" in prompt
    assert {"type": "task_started", "id": "call_7",
            "prompt": "count the files in web/"} in browser.sent
    assert callable(on_finished)
    assert profile is None
    # The receipt the model is handed is the id it must use later — one name per task.
    assert task_id == "call_7"
    assert callable(on_needs_input)
    assert live.responses and live.responses[0]["result"].startswith("WORK_STARTED call_7")


@pytest.mark.asyncio
async def test_an_empty_task_is_refused_without_starting_a_run(opened, monkeypatch):
    calls = []
    monkeypatch.setattr(relay.agent_lane, "start_task",
                        lambda *a, **k: calls.append(a))
    live = opened(FakeLive(incoming=[
        {"toolCall": {"functionCalls": [{"id": "c", "name": "hermes_task", "args": {}}]}}]))
    await relay.run_relay(FakeBrowser([]))
    assert calls == []
    assert live.responses == [{"id": "c", "result": "No task text was supplied.",
                               "name": "hermes_task"}]


@pytest.mark.asyncio
async def test_the_delegate_tool_is_offered_only_when_the_lane_answers(opened):
    live = opened(FakeLive(incoming=[]), lane=(agent_lane.STATUS_ABSENT, "nothing listening"))
    browser = FakeBrowser([])
    await relay.run_relay(browser)
    assert opened.seen["tools"] is None
    assert {"type": "lane", "status": "absent", "detail": "nothing listening"} in browser.sent


@pytest.mark.asyncio
async def test_the_delegate_tool_is_offered_when_the_lane_is_live(opened):
    opened(FakeLive(incoming=[]))
    await relay.run_relay(FakeBrowser([]))
    names = [entry["name"] for entry in opened.seen["tools"]]
    assert names == [tools_module.DELEGATE_NAME, tools_module.BOARD_NAME,
                     tools_module.UPDATE_NAME]


@pytest.mark.asyncio
async def test_reading_the_board_answers_from_memory_and_starts_nothing(opened, monkeypatch):
    # The point of the board verb is that a second request about running work costs no
    # second run, so this asserts the absence as much as the answer.
    started = []
    monkeypatch.setattr(relay.agent_lane, "start_task", lambda *a, **k: started.append(1))
    live = opened(FakeLive(incoming=[]))
    book = agent_lane.RunBook()
    book.register("t1", "list downloads")
    book.set_state("t1", "needs_input", question="It wants to run: rm -rf scratch")
    await relay._handle_tool_call(
        {"id": "c9", "name": tools_module.BOARD_NAME, "args": {}},
        live, FakeBrowser([]), book, None, None, relay.NoteQueue(live))
    assert started == []
    assert live.responses[0]["result"].startswith("#t1 [needs_input]")
    assert "rm -rf scratch" in live.responses[0]["result"]


@pytest.mark.asyncio
async def test_the_voice_model_cannot_approve_a_parked_run(opened, monkeypatch):
    # The model hears the room and reads run output, so an injected "approve it" must not
    # be able to unlock a dangerous command through a tool call.
    sent = []
    monkeypatch.setattr(relay.agent_lane, "approve",
                        lambda run_id, choice, request_id, **k: sent.append(
                            (run_id, choice, request_id)))
    live = opened(FakeLive(incoming=[]))
    book = agent_lane.RunBook()
    book.register("t1", "clean the scratch dir")
    book.attach_run("t1", "run_1")
    book.set_state("t1", "needs_input", question="may I?", request_id="req_9")
    browser = FakeBrowser([])
    await relay._update_task("c10", {"task": "t1", "action": "approve", "answer": "yes"},
                             live, browser, book, None, None)
    assert sent == []
    assert live.responses[0]["result"].startswith("BAD_REQUEST")
    assert "approve" not in tools_module.ACTIONS


@pytest.mark.asyncio
async def test_the_users_approve_press_reaches_the_lane_with_its_request(opened, monkeypatch):
    sent = []
    monkeypatch.setattr(relay.agent_lane, "approve",
                        lambda run_id, choice, request_id, **k: sent.append(
                            (run_id, choice, request_id)))
    live = opened(FakeLive(incoming=[]))
    book = agent_lane.RunBook()
    book.register("t1", "clean the scratch dir")
    book.attach_run("t1", "run_1")
    book.set_state("t1", "needs_input", question="may I?", request_id="req_9")
    browser = FakeBrowser([{"type": "approval", "id": "t1", "choice": "approve"},
                           {"type": "close"}])
    await relay._browser_to_live(browser, live, {"reason": ""}, book, None, None)
    assert sent == [("run_1", "once", "req_9")]
    assert {"type": "task_update", "id": "t1", "action": "approve"} in browser.sent
    assert book.pending() == ["t1"]


@pytest.mark.asyncio
async def test_a_browser_that_navigates_away_ends_the_call_without_an_error(opened):
    # The page disappearing mid-call used to come back through the same path as a real
    # fault, so every navigation painted "the system failed" on the panel.
    class Gone(FakeBrowser):
        async def receive_text(self):
            raise WebSocketDisconnect()

    live = opened(FakeLive(end_after_script=False))
    browser = Gone([])
    await relay.run_relay(browser)
    assert [f for f in browser.sent if f["type"] == "error"] == []
    assert live.closed is True


@pytest.mark.asyncio
async def test_a_finished_run_crosses_back_onto_the_loop_as_a_spoken_note():
    # The worker thread must not touch sockets itself; it hands the loop a coroutine.
    live = FakeLive(end_after_script=False)
    browser = FakeBrowser([])
    finish = relay._make_finisher(relay.NoteQueue(live), browser, asyncio.get_running_loop())

    finish("task-9", "completed", "4 files matched")
    await asyncio.sleep(0.1)

    assert any("Hermes finished task #task-9" in text and "4 files matched" in text
               for text in live.text_in)
    assert {"type": "task_done", "id": "task-9", "state": "completed"} in browser.sent


@pytest.mark.asyncio
async def test_a_result_that_lands_mid_sentence_waits_for_the_gap():
    # Injecting into the model's own speech made it stop the sentence it was saying and read
    # the report instead. The panel is still told at once; only the note waits.
    live = FakeLive(end_after_script=False)
    browser = FakeBrowser([])
    clock = {"now": 1000.0}
    notes = relay.NoteQueue(live, clock=lambda: clock["now"])
    relay._make_finisher(notes, browser, asyncio.get_running_loop())(
        "t1", "completed", "4 files matched")
    notes.speaking()
    await asyncio.sleep(0.05)

    assert live.text_in == [] and notes.waiting() == 1
    assert {"type": "task_done", "id": "t1", "state": "completed"} in browser.sent

    clock["now"] += relay.QUIET_BEFORE_NOTE_S + 0.1
    assert await notes.release_quietly() is True
    assert "Hermes finished task #t1" in live.text_in[0]
    assert notes.waiting() == 0


def test_the_report_note_asks_for_a_lead_and_not_for_an_interruption():
    # Both halves of the old wording were the bug: "now" ordered the model to cut itself
    # off, and "one short spoken sentence" is why a whole report arrived as one clause.
    ordering = relay.RESULT_NOTE.split("Result:")[0]
    assert "now" not in ordering
    assert "one short spoken sentence" not in relay.RESULT_NOTE
    assert "answer in your first sentence" in relay.RESULT_NOTE
    assert "at most four" in relay.RESULT_NOTE


@pytest.mark.asyncio
async def test_a_result_delivered_to_a_dead_socket_never_escapes_the_thread():
    # on_finished runs on a worker thread, where a raised exception would be swallowed by
    # the interpreter rather than reported, so the handling has to live where it happens.
    class Dead(FakeLive):
        async def send_text(self, text):
            raise LiveError("the Gemini Live socket closed (1006)")

    dead = Dead(end_after_script=False)
    notes = relay.NoteQueue(dead)
    finish = relay._make_finisher(notes, FakeBrowser([]), asyncio.get_running_loop())

    finish("t", "completed", "anything")
    await asyncio.sleep(0.1)

    assert dead.text_in == []
    # A failed send must not eat the result: the pump gets another turn at it.
    assert notes.waiting() == 1


@pytest.mark.asyncio
async def test_blank_frames_are_dropped_not_forwarded(opened):
    live = opened(FakeLive(end_after_script=False))
    browser = FakeBrowser([{"type": "audio", "data": ""},
                           {"type": "text", "text": "  "},
                           {"type": "voice_command"},
                           {"type": "close"}])
    await relay.run_relay(browser)
    assert live.audio_in == []
    assert live.text_in == []
