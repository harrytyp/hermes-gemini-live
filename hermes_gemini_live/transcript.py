"""The spoken conversation, as far as this lane is allowed to know it.

Gemini sends transcripts as deltas, in two separate streams (``inputTranscription`` and
``outputTranscription``), interleaved with audio and control frames. Nothing here
touches a socket: the relay feeds frames in, and asks two questions out —

* what did the last few seconds actually say (the context a delegated task travels with,
  exactly the way core's GPT-Live lane hands its own transcript window to ``prompt.submit``);
* which complete utterances still have to become rows in the session.

The window is a ring, deliberately: a call can run for an hour and only the tail is ever
useful as context. Settled utterances are handed out once and never re-read, so writing
them can neither duplicate a row nor lose one when a write is refused.
"""

from __future__ import annotations

import time

MAX_FRAGMENTS = 2000
KEEP_FRAGMENTS = 1500
#: Core's own cap for a spoken window riding a turn (``gateway/session_surface.py``).
MAX_CONTEXT_CHARS = 6000

USER = "user"
ASSISTANT = "assistant"


class Transcript:
    """Deltas in, readable turns out."""

    def __init__(self, max_fragments: int = MAX_FRAGMENTS,
                 keep: int = KEEP_FRAGMENTS) -> None:
        self._fragments: list[dict] = []
        self._max = max_fragments
        self._keep = keep
        self._written = 0

    # ── feeding ──────────────────────────────────────────────────────────────
    def add(self, speaker: str, text: str) -> None:
        """One delta. Same-speaker deltas merge into the turn already in progress."""
        if not text:
            return
        last = self._fragments[-1] if self._fragments else None
        if last is not None and last["speaker"] == speaker and not last["settled"]:
            last["text"] += text
            last["at"] = time.time()
        else:
            self._fragments.append({"speaker": speaker, "text": text,
                                    "settled": False, "at": time.time()})
        if len(self._fragments) > self._max:
            dropped = len(self._fragments) - self._keep
            self._fragments = self._fragments[dropped:]
            self._written = max(0, self._written - dropped)

    def interrupted(self) -> None:
        """The model was cut off: what it managed to say is a finished utterance."""
        last = self._fragments[-1] if self._fragments else None
        if last is not None and last["speaker"] == ASSISTANT:
            last["settled"] = True

    def turn_complete(self) -> None:
        """Everything spoken so far is a finished utterance."""
        for fragment in self._fragments:
            fragment["settled"] = True

    # ── reading ──────────────────────────────────────────────────────────────
    def context(self, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """The tail of the conversation, labelled and capped, for a worker to read."""
        lines: list[str] = []
        for fragment in reversed(self._fragments):
            label = "User" if fragment["speaker"] == USER else "Hermes"
            lines.append(f"{label}: {fragment['text'].strip()}")
            if sum(len(line) + 1 for line in lines) >= max_chars:
                break
        lines.reverse()
        body = "\n".join(lines).strip()
        if len(body) <= max_chars:
            return body
        # Trim from the front, then drop the partial first line: a cut inside a line would
        # leave "ser: …" and read as somebody else talking.
        tail = body[-max_chars:]
        return tail.split("\n", 1)[1] if "\n" in tail else tail

    def pending_rows(self) -> list[dict]:
        """Settled utterances not yet written, oldest first. Marks them as handed out."""
        rows = [{"role": ASSISTANT if fragment["speaker"] == ASSISTANT else USER,
                 "content": fragment["text"].strip(), "at": fragment["at"]}
                for fragment in self._fragments[self._written:]
                if fragment["settled"] and fragment["text"].strip()]
        self._written = len(self._fragments)
        return rows

    def pending_count(self) -> int:
        return len([fragment for fragment in self._fragments[self._written:]
                    if fragment["settled"] and fragment["text"].strip()])

    def __len__(self) -> int:
        return len(self._fragments)


def labelled_block(context: str) -> str:
    """A spoken window, fenced so the worker cannot mistake it for its own task text."""
    body = (context or "").strip()
    if not body:
        return ""
    return ("Recent spoken exchange from the voice call (the user was talking, not typing; "
            "the request itself is below):\n" + body)
