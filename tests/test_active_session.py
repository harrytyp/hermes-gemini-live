"""The renderer lane: a task that runs in the chat the user has open.

The api_server refuses to resume a session it did not create, so in ``active_session`` mode
the relay does not submit anything — it hands the work to the renderer, which uses the same
door the built-in voice lane uses (``session.activate`` + ``prompt.submit``). These tests
pin the two halves of that contract that live in Python: the frame the relay sends, and the
answer frame it accepts.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hermes_gemini_live import agent_lane, config, relay, session_log, settings, transcript


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_home", lambda: tmp_path)
    return tmp_path


class Browser:
    def __init__(self, incoming=None):
        self.sent: list[dict] = []
        self._incoming = list(incoming or [])
        self._wake = asyncio.Event()

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def receive_text(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        await asyncio.sleep(3600)
        return ""


class Live:
    model = config.DEFAULT_MODEL

    def __init__(self):
        self.notes: list[str] = []

    async def send_function_response(self, *a, **kw):
        return None

    async def send_text(self, text):
        self.notes.append(text)


def _ctx(target: str = "active_session") -> relay.CallContext:
    return relay.CallContext(transcript=transcript.Transcript(),
                             log=session_log.CallLog(None, None),
                             target=target, mode="delegations")


@pytest.mark.asyncio
async def test_a_tool_call_in_active_session_mode_is_handed_to_the_renderer(monkeypatch):
    started: list = []
    monkeypatch.setattr(relay.agent_lane, "start_task",
                        lambda *a, **kw: started.append((a, kw)) or "task-1")
    browser = Browser()
    ctx = _ctx()
    ctx.transcript.add(transcript.USER, "stell bitte einen Termin ein")
    notes = relay.NoteQueue(Live())
    book = agent_lane.RunBook(None)

    await relay._handle_tool_call(
        {"id": "call_9", "name": "hermes_task", "args": {"task": "Termin anlegen"}},
        Live(), browser, book, None, None, notes, ctx)

    assert started == [], "the api_server lane must not be used in active_session mode"
    frames = [frame for frame in browser.sent if frame["type"] == "delegate"]
    assert len(frames) == 1
    assert frames[0]["id"] == "call_9"
    assert frames[0]["prompt"] == "Termin anlegen"
    assert "stell bitte einen Termin ein" in frames[0]["context"]
    assert book.pending() == ["call_9"]


@pytest.mark.asyncio
async def test_the_renderers_answer_becomes_a_spoken_note():
    browser = Browser()
    live = Live()
    notes = relay.NoteQueue(live)
    book = agent_lane.RunBook(None)
    book.register("call_9", "Termin anlegen")

    await relay._delegate_finished({"id": "call_9", "state": "completed",
                                    "text": "Der Termin steht."}, browser, book, notes)

    assert book.pending() == []
    assert len(live.notes) == 1, "the model has to be told; a silent board is not an answer"
    assert "Der Termin steht." in live.notes[0]
    assert {"type": "task_done", "id": "call_9", "state": "completed"} in browser.sent


@pytest.mark.asyncio
async def test_the_open_chat_is_adopted_for_the_transcript(tmp_path):
    conn = {"type": "session", "id": "local-abc"}
    browser = Browser([json.dumps(conn), json.dumps({"type": "close"})])
    ctx = _ctx()
    ctx.log.append([{"role": "user", "content": "gesagt, bevor der Chat bekannt war"}])

    await relay._browser_to_live(browser, Live(), {"reason": ""}, agent_lane.RunBook(None),
                                 None, None, ctx)

    assert ctx.log.session_id == "local-abc"
    assert ctx.log._buffer == [], "buffered speech must be flushed into the adopted chat"


def test_the_frame_vocabulary_accepts_the_renderer_frames():
    assert relay._frame_type(json.dumps({"type": "delegate_result", "id": "x"})) == "delegate_result"
    assert relay._frame_type(json.dumps({"type": "delegate_result"})) == ""
    assert relay._frame_type(json.dumps({"type": "session", "id": "local-abc"})) == "session"
    assert relay._frame_type(json.dumps({"type": "session"})) == ""
