"""Which Live models this key has, which of them can hold a conversation, and which first.

The panel used to offer a hardcoded pair. That is wrong in both directions: it hides a
model the key can run, and it offers one the key may not have. ``ListModels`` answers for
the actual key, so the catalog is asked, not guessed.

**Everything here is derived, never enumerated.** Google ships new Live models, deprecates
old ones and renames families; a list of ids in this file would be stale the week it was
written. The catalog is therefore the API's own answer, and the only knowledge coded here
is the *shape* of the names — which families are conversational and which are not.

Measured on 2026-11-25 against a real key (8 live-capable models, one real call each):

    gemini-3.8-live                              answered   conversational
    gemini-3.8-live-extended-thinking             answered   conversational, needs a level
    gemini-3.1-flash-live-preview                 answered   conversational
    gemini-2.5-flash-native-audio-latest          answered   conversational
    gemini-3.5-live-translate-preview            answered   but translates, never converses
    gemini-3.5-transcribe-live                   1007       speech-to-text only
    gemini-robotics-er-2-streaming-preview       1007       needs robotics payloads
    genie-4                                      1011       audio-video generation

Which is why ``bidiGenerateContent`` alone is not a usable filter: Google lists eight models
on that method and only four of them will hold a call. The excluded ones are excluded by
*family*, so a newly released conversational model is picked up with no code change, and a
new transcriber or robotics model is left alone without one either.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request

from . import config

logger = logging.getLogger(__name__)

LIST_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
CACHE_TTL_S = 600.0

#: Families that answer ``bidiGenerateContent`` but cannot be a voice assistant. Matched
#: against the id, so a new revision of an excluded family stays excluded and an unforeseen
#: conversational family is not caught here. ``translate`` is deliberate: that model replies
#: with a translation of what it heard rather than a reply to it.
_NOT_CONVERSATIONAL = (
    "transcribe", "translate", "robotics", "genie", "embedding", "image", "video",
    "tts", "vision",
)

#: Used only when the key cannot be asked at all (offline startup, broken key): the ids that
#: were measured answering. A working key always wins, because the catalog is what both the
#: dropdown and the fallback chain read.
_LAST_RESORT = ("gemini-3.8-live", "gemini-3.8-live-extended-thinking")

_cache: dict = {"at": 0.0, "key": "", "models": []}


def is_conversational(name: str) -> bool:
    """Whether a Live model can hold a spoken conversation with us.

    Judged by family, not by a list of ids: Google adds models faster than this file could be
    updated, and what separates a voice assistant from a transcriber is what its name says
    it is for.
    """
    lowered = name.lower()
    return not any(word in lowered for word in _NOT_CONVERSATIONAL)


def needs_thinking_level(name: str) -> bool:
    """Whether this model's setup must carry a ``thinkingConfig``.

    Measured: ``...-extended-thinking`` answers "Thinking level must be specified for this
    model" without one, while ``gemini-3.8-live`` rejects the field outright ("Thinking level
    is not supported"). So this is a per-family wire requirement, not a preference — and it
    is read from the id rather than pinned, so a future thinking variant keeps working.
    """
    return "extended-thinking" in name.lower()


def _version(name: str) -> tuple:
    """Sort key from the numbers in an id: ``gemini-3.8-live`` -> (3, 8).

    An id with no number sorts last, so an unnumbered family never jumps ahead of a known one.
    """
    found = re.findall(r"(\d+)(?:\.(\d+))?", name)
    if not found:
        return (-1, -1)
    major, minor = found[0]
    return (int(major), int(minor or 0))


def _rank(name: str) -> tuple:
    """Best first, derived from the id alone.

    In order: a conversational model before anything else; a released id before a
    ``preview`` one (a preview can be withdrawn, a release stays); the newest version before
    an older one; then the plain variant before a specialised sibling. No id is named, so a
    new release lands in the right place without an edit here.
    """
    lowered = name.lower()
    return (
        0 if is_conversational(name) else 1,
        1 if "preview" in lowered else 0,
        tuple(-part for part in _version(lowered)),
        1 if "extended-thinking" in lowered else 0,
        lowered,
    )


def _fetch(key: str) -> list[str]:
    url = f"{LIST_MODELS_URL}?key={key}&pageSize=1000"
    request = urllib.request.Request(url, headers={"User-Agent": "hermes-gemini-live"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names: list[str] = []
    for model in payload.get("models", []):
        raw = str(model.get("name") or "")
        # The ONLY hard filter is the API's own answer: bidiGenerateContent is the protocol this
        # lane speaks, and Google publishes it as machine-readable metadata. Deliberately no
        # extra "must contain 'live'" test — that is a guess about naming, and it silently drops
        # a real model the day a family is called something else (a hypothetical
        # ``gemini-4-pro-realtime`` would vanish). Whether the model is *usable* as a voice
        # assistant is then a family question, answered by :func:`is_conversational`.
        if "bidiGenerateContent" not in (model.get("supportedGenerationMethods") or []):
            continue
        names.append(raw.split("/", 1)[1] if raw.startswith("models/") else raw)
    return names


def catalog(refresh: bool = False) -> list[str]:
    """Live model ids for this key, best first. Never raises; falls back to the last resort.

    Includes families that cannot converse — a transcriber really is Live-capable. Use
    :func:`voice_models` for the ones this lane may actually dial.
    """
    try:
        key = config.resolve_key()
    except config.ConfigError:
        return list(_LAST_RESORT)
    now = time.time()
    if (not refresh and _cache["models"] and _cache["key"] == key
            and now - _cache["at"] < CACHE_TTL_S):
        return list(_cache["models"])
    try:
        names = _fetch(key)
    except Exception as exc:
        logger.info("hermes-gemini-live: model catalog unavailable (%s); using the last resort",
                    type(exc).__name__)
        return list(_LAST_RESORT)
    if not names:
        return list(_LAST_RESORT)
    names.sort(key=_rank)
    _cache.update({"at": now, "key": key, "models": names})
    return list(names)


def voice_models(refresh: bool = False) -> list[str]:
    """The models this lane may dial: the conversational ones, best first.

    Falls back to the whole catalog when nothing classifies as conversational, so an
    unforeseen naming scheme degrades to "try them" instead of "no models at all".
    """
    names = catalog(refresh=refresh)
    usable = [name for name in names if is_conversational(name)]
    return usable or names


def choices() -> tuple[str, ...]:
    """The dropdown's contents: what this key can actually converse with."""
    return tuple(voice_models())


def fallback_order(current: str) -> list[str]:
    """The models to try after ``current`` goes deaf, best first, never ``current``.

    Reads the same catalog the dropdown does, so a model Google publishes tomorrow is
    trialled without a release here.
    """
    return [name for name in voice_models() if name != current]
