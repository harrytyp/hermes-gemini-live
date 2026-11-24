"""The watchdog's own rule: the stamp is the FIRST words, not the latest.

A user who keeps talking into a model that never answers is exactly the case the watchdog
exists for. Stamping every loud frame would move the deadline forward forever and the call
would never be rescued.
"""

from __future__ import annotations

import asyncio
import json
import struct

import pytest

from hermes_gemini_live import agent_lane, relay

LOUD = None  # built below: a full frame of speech-level samples


def _frame(sample: int) -> str:
    import base64
    return base64.b64encode(struct.pack("<1600h", *([sample] * 1600))).decode()


LOUD = _frame(9000)
QUIET = _frame(0)


class Browser:
    """Delivers a script, then repeats its last frame forever."""

    def __init__(self, frames):
        self.sent: list[dict] = []
        self._frames = [json.dumps({"type": "audio", "data": data}) for data in frames]
        self._last = self._frames[-1] if self._frames else json.dumps(
            {"type": "audio", "data": QUIET})

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))

    async def receive_text(self):
        await asyncio.sleep(0.01)
        return self._frames.pop(0) if self._frames else self._last


class Live:
    model = "deaf"

    async def send_audio(self, b64):
        return None

    async def send_text(self, text):
        return None

    async def send_function_response(self, *a, **kw):
        return None


@pytest.mark.asyncio
async def test_continuing_speech_does_not_postpone_the_deadline(monkeypatch):
    note = {"session_opened_at": 0.0}
    browser = Browser([LOUD])          # loud forever: the user never stops talking
    live = Live()
    up = asyncio.create_task(relay._browser_to_live(browser, live, note,
                                                    agent_lane.RunBook(None)))
    monkeypatch.setattr(relay, "SILENT_MODEL_GRACE_S", 0.3)
    try:
        task = asyncio.create_task(relay._watch_for_a_deaf_model(live, note))
        await asyncio.wait_for(task, timeout=5)
    finally:
        up.cancel()
        await asyncio.gather(up, return_exceptions=True)
    assert note["switch_model"] is True, "a talking user must still be rescued"
    # The stamp stayed the first one: it is a start of speech, not a heartbeat.
    first = note["heard_speech_at"]
    await asyncio.sleep(0.5)
    up2 = asyncio.create_task(relay._browser_to_live(browser, live, note,
                                                     agent_lane.RunBook(None)))
    await asyncio.sleep(0.3)
    up2.cancel()
    await asyncio.gather(up2, return_exceptions=True)
    assert note["heard_speech_at"] == first


@pytest.mark.asyncio
async def test_quiet_frames_do_not_count_as_speech():
    note = {"session_opened_at": 0.0}
    browser = Browser([QUIET])
    up = asyncio.create_task(relay._browser_to_live(browser, Live(), note,
                                                    agent_lane.RunBook(None)))
    await asyncio.sleep(0.3)
    up.cancel()
    await asyncio.gather(up, return_exceptions=True)
    assert "heard_speech_at" not in note, "an open microphone is not a user asking something"
