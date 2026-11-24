"""A Live model that accepts the setup and then ignores the microphone.

Measured against the real endpoint: identical bytes, gemini-3.8-live answered 1 of 3 calls
while gemini-3.1-flash-live-preview answered 3 of 3. The session is healthy the whole time —
voice activity arrives, no error frame, no goodbye — so neither pump can notice, and before
this the user's only symptom was "I can speak and hear nothing back".
"""

from __future__ import annotations

import asyncio
import json

import pytest

from hermes_gemini_live import agent_lane, relay


class Browser:
    def __init__(self, incoming=None):
        self.sent: list[dict] = []
        self._incoming = list(incoming or [])

    async def send_text(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def receive_text(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        await asyncio.sleep(3600)
        return ""


class Silent:
    """Hears the user, says nothing — the production symptom."""

    model = "deaf-model"

    def __init__(self):
        self.audio_in = []
        self.text_in = []

    async def send_audio(self, b64):
        self.audio_in.append(b64)

    async def send_text(self, text):
        self.text_in.append(text)

    async def send_function_response(self, *a, **kw):
        return None

    async def recv(self):
        await asyncio.sleep(3600)

    async def close(self):
        return None


def test_speech_is_recognised_from_the_pcm_the_renderer_sends():
    # 16-bit little-endian PCM, base64, exactly what the renderer puts on the socket.
    import base64
    import struct

    quiet = base64.b64encode(b"\x00\x00" * 400).decode()
    loud = base64.b64encode(struct.pack("<400h", *([8000] * 400))).decode()
    assert relay._speech_peak(quiet) == 0.0
    assert relay._speech_peak(loud) >= relay.SPEECH_PEAK
    # Anything undecodable never counts as speech, so a broken uplink cannot fake a user.
    assert relay._speech_peak("") == 0.0
    assert relay._speech_peak("!!!not base64!!!") == 0.0


@pytest.mark.asyncio
async def test_a_model_that_never_answers_is_reported():
    note = {"session_opened_at": 0.0, "heard_speech_at": 0.0, "model_answered": False}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(relay, "SILENT_MODEL_GRACE_S", 0.2)
    try:
        await asyncio.wait_for(relay._watch_for_a_deaf_model(Silent(), note), timeout=5)
    finally:
        monkeypatch.undo()
    assert note["switch_model"] is True


@pytest.mark.asyncio
async def test_a_model_that_answers_is_left_alone():
    note = {"session_opened_at": 0.0, "heard_speech_at": 0.0, "model_answered": True}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(relay, "SILENT_MODEL_GRACE_S", 0.2)
    try:
        task = asyncio.create_task(relay._watch_for_a_deaf_model(Silent(), note))
        await asyncio.sleep(1.0)
        assert task.done() is False, "a working call must not be torn down"
        task.cancel()
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_speech_from_before_this_session_does_not_trip_it():
    # A resumed socket inherits a conversation that was already heard; the watchdog must not
    # read that as a deaf model and restart the call the user just resumed.
    note = {"session_opened_at": 100.0, "heard_speech_at": 50.0, "model_answered": False}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(relay, "SILENT_MODEL_GRACE_S", 0.2)
    try:
        task = asyncio.create_task(relay._watch_for_a_deaf_model(Silent(), note))
        await asyncio.sleep(1.0)
        assert note.get("switch_model") is None
        task.cancel()
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_the_first_attempt_has_nothing_to_exclude_yet(monkeypatch):
    """The retry has to be able to say "not that one again"; the first dial has no history."""
    chosen: list[frozenset] = []

    class Router:
        @staticmethod
        async def open(tools=None, handle=None, exclude=None):
            chosen.append(exclude or frozenset())
            raise relay.LiveError("stop here")

    monkeypatch.setattr(relay, "LiveSession", Router)
    monkeypatch.setattr(relay.agent_lane, "probe",
                        lambda session_key=None, profile=None: (agent_lane.STATUS_OK, ""))
    await relay.run_relay(Browser([]))
    assert chosen == [frozenset()]
