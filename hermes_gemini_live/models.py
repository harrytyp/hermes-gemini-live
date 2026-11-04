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
#: The order the panel shows when the key answers: the model the spike proved first, then
#: its thinking variant, then the newest native-audio family, then anything else Live.
_PROVEN = (config.DEFAULT_MODEL, config.DEFAULT_MODEL + "-extended-thinking")

_cache: dict = {"at": 0.0, "key": "", "models": []}


def _rank(name: str) -> tuple:
    if name == _PROVEN[0]:
        return (0, 0)
    if name == _PROVEN[1]:
        return (1, 0)
    if "native-audio-latest" in name:
        return (2, 0)
    if "native-audio-preview" in name:
        return (3, 0)
    return (4, 0)


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
