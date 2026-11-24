"""Which Live models this key actually has — asked, not guessed.

The panel used to offer a hardcoded pair. That is wrong in both directions: it hides a
model the key can run, and it offers one the key may not have. ``ListModels`` answers for
the actual key, so the dropdown is the truth.

Adopted from gemini-live-bridge (cruzlxyz), whose catalog filter and ranking are the
part that took the guessing out: keep only models that declare ``bidiGenerateContent``,
drop transcribe/translate variants, and order latest-first. The result is cached, because
a dropdown that costs a network round trip per render is worse than a slightly stale list.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request

from . import config

logger = logging.getLogger(__name__)

LIST_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
CACHE_TTL_S = 600.0
#: What is known to work, most-tested first — measured against the live endpoint, three calls
#: per model on 2026-11-24 with identical audio:
#:
#:     gemini-3.8-live                            1/3   (accepts the setup, ignores the mic)
#:     gemini-3.8-live-extended-thinking           0/3
#:     gemini-3.1-flash-live-preview               3/3
#:     gemini-2.5-flash-native-audio-latest        3/3
#:     gemini-2.5-flash-native-audio-preview-12-2025  3/3
#:
#: So the ranking is not "newest first" — it is "answers first", and the fallback chain now
#: walks a model that metered 3/3 before one that metered 0/3. Rewrite this table only with
#: fresh measurements; a guess here costs the user a silent call.
_PROVEN = (config.DEFAULT_MODEL, config.DEFAULT_MODEL + "-extended-thinking")
#: Tried in this order when the configured model goes deaf.
_ANSWERED_3_OF_3 = (
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)

_cache: dict = {"at": 0.0, "key": "", "models": []}


def _rank(name: str) -> tuple:
    """Best first. A model that answers beats a model that merely exists."""
    if name in _ANSWERED_3_OF_3:
        return (0, _ANSWERED_3_OF_3.index(name))
    if name == _PROVEN[0]:
        return (1, 0)
    if name == _PROVEN[1]:
        return (2, 0)
    if "native-audio-latest" in name:
        return (3, 0)
    if "native-audio-preview" in name:
        return (4, 0)
    return (5, 0)


def _fetch(key: str) -> list[str]:
    url = f"{LIST_MODELS_URL}?key={key}&pageSize=200"
    request = urllib.request.Request(url, headers={"User-Agent": "hermes-gemini-live"})
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names: list[str] = []
    for model in payload.get("models", []):
        raw = str(model.get("name") or "")
        lowered = raw.lower()
        if "live" not in lowered and "native-audio" not in lowered:
            continue
        if "bidiGenerateContent" not in (model.get("supportedGenerationMethods") or []):
            continue
        if "transcribe" in lowered or "translate" in lowered:
            continue
        names.append(raw.split("/", 1)[1] if raw.startswith("models/") else raw)
    return names


def catalog(refresh: bool = False) -> list[str]:
    """Live model ids for this key, best first. Never raises; falls back to the proven two."""
    try:
        key = config.resolve_key()
    except config.ConfigError:
        return list(_PROVEN)
    now = time.time()
    if (not refresh and _cache["models"] and _cache["key"] == key
            and now - _cache["at"] < CACHE_TTL_S):
        return list(_cache["models"])
    try:
        names = _fetch(key)
    except Exception as exc:
        logger.info("hermes-gemini-live: model catalog unavailable (%s); using the proven list",
                    type(exc).__name__)
        return list(_PROVEN)
    if not names:
        return list(_PROVEN)
    names.sort(key=lambda name: (_rank(name), name))
    _cache.update({"at": now, "key": key, "models": names})
    return list(names)


def choices() -> tuple[str, ...]:
    """The dropdown's contents: the catalog, with the proven ids always present."""
    names = catalog()
    for model in reversed(_PROVEN):
        if model not in names:
            names.insert(0, model)
    return tuple(names)
