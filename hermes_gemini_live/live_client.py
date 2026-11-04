"""One Gemini Live socket, owned by one browser connection.

The API key rides the socket URL, so every error string leaving this module is
redacted first — a raw ``websockets`` exception quotes the URI it failed on.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets

from . import config, wire

logger = logging.getLogger(__name__)

OPEN_TIMEOUT_S = 30.0
SETUP_TIMEOUT_S = 30.0


class LiveError(RuntimeError):
    """A Live-side failure, already safe to show or log."""


def uri_with(key: str) -> str:
    """The endpoint with the key in the query string.

    Live takes the credential only in the URL, so this is the single place the secret is
    ever written — and why every error leaving this module is redacted: a raw transport
    exception quotes the URI it failed on.
    """
    return f"{wire.WS_URI}?key={key}"


def _redactor(*secrets: str):
    needles = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)

    def redact(text: str) -> str:
        for needle in needles:
            text = text.replace(needle, "***")
        return text
    return redact


class LiveSession:
    """A thin async wrapper: connect, setup, append, receive.

    Deliberately no reconnection here — a dropped Live socket ends the voice call, and
    silently re-opening one would replay context the user believes was thrown away.
    """

    def __init__(self, socket: Any, key: str, model: str) -> None:
        self._socket = socket
        self._redact = _redactor(key)
        self.model = model
        self.closed_reason = ""

    @classmethod
    async def open(cls, tools: list[dict] | None = None,
                   handle: str | None = None) -> "LiveSession":
        """Open one session, with two adopted retries around the original single attempt.

        From gemini-live-bridge: a rejected resume handle is not a dead call — retry the
        same model fresh once — and a model that refuses the setup is not a dead call
        either when another Live model is available. Both retries are bounded, and both
        are visible in the log, because silently switching the voice model mid-dial would
        be worse than the failure it hides.
        """
        try:
            key = config.resolve_key()
            model = config.model()
            base = {
                "voice": config.voice(),
                "instructions": config.instructions(),
                "activity": config.activity_detection(),
                "tools": tools,
            }
        except config.ConfigError as exc:
            raise LiveError(str(exc)) from exc
        candidates = [model]
        if config.model_fallback():
            try:
                from . import models
                candidates += [name for name in models.catalog()[:4] if name != model]
            except Exception:
                pass  # the chosen model alone is still a valid attempt
        last: LiveError | None = None
        for candidate in candidates:
            plan = {**base, "model": candidate,
                    "thinking_level": config.thinking_level(candidate)}
            attempts = (handle, None) if handle else (None,)
            for resume in attempts:
                if resume is not None:
                    plan["resume_handle"] = resume
                else:
                    plan.pop("resume_handle", None)
                try:
                    return await cls._connect(key, plan, resume)
                except LiveError as exc:
                    last = exc
                    if resume is not None:
                        logger.info("hermes-gemini-live: resume handle refused (%s); "
                                    "retrying %s with a fresh session", exc, candidate)
                        continue
                    if len(candidates) > 1:
                        logger.info("hermes-gemini-live: %s refused the setup (%s); trying "
                                    "the next model", candidate, exc)
                    break
        raise last or LiveError("no Live model could be opened")

    @classmethod
    async def _connect(cls, key: str, plan: dict, handle: str | None) -> "LiveSession":
        """The single-socket path: connect, send setup, wait for setupComplete."""
        try:
            socket = await websockets.connect(uri_with(key), max_size=None,
                                              open_timeout=OPEN_TIMEOUT_S)
        except Exception as exc:
            raise LiveError(_redactor(key)(f"{type(exc).__name__}: {exc}")) from exc
        session = cls(socket, key, plan["model"])
        try:
            await session._send(wire.build_setup(**plan))
            await session._await_setup()
        except BaseException:
            await session.close()
            raise
        logger.info("hermes-gemini-live: Live session open (%s%s, silence=%dms)",
                    plan["model"], " resuming" if handle else "",
                    plan["activity"]["automaticActivityDetection"]["silenceDurationMs"])
        return session

    async def _await_setup(self) -> None:
        deadline = asyncio.get_running_loop().time() + SETUP_TIMEOUT_S
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise LiveError("Gemini Live did not acknowledge the session setup")
            frame = await self.recv()
            if "setupComplete" in frame:
                return
            if error := frame.get("error"):
                raise LiveError(str(error.get("message") or error)[:400])

    async def _send(self, frame: dict) -> None:
        try:
            await self._socket.send(json.dumps(frame))
        except Exception as exc:
            raise LiveError(self._redact(f"{type(exc).__name__}: {exc}")) from exc

    async def send_audio(self, b64: str) -> None:
        await self._send(wire.audio_chunk(b64))

    async def send_text(self, text: str) -> None:
        await self._send(wire.text_input(text))

    async def send_function_response(self, call_id: str, result: str,
                                      name: str | None = None) -> None:
        await self._send(wire.function_response(call_id, result, name))

    async def recv(self) -> dict:
        try:
            return json.loads(await self._socket.recv())
        except websockets.exceptions.ConnectionClosed as exc:
            self.closed_reason = self._redact(str(exc)) or "closed"
            raise LiveError(self._redact(f"the Gemini Live socket closed ({exc.code})")) from exc
        except Exception as exc:
            raise LiveError(self._redact(f"{type(exc).__name__}: {exc}")) from exc

    async def close(self) -> None:
        try:
            await self._socket.close()
        except Exception as exc:  # a close that fails is not a call that fails
            logger.debug("hermes-gemini-live: Live close raised %s: %s",
                         type(exc).__name__, self._redact(str(exc)))
