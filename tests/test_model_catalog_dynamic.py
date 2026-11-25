"""The catalog is derived, not enumerated — so a model Google ships tomorrow still dials.

Every assertion here feeds the ranking and the classification a list of ids that does not
exist yet. If someone later "fixes" a bug by pinning an id into a list, one of these fails.
"""

from __future__ import annotations

import pytest

from hermes_gemini_live import models


def test_a_brand_new_conversational_model_is_dialable_without_a_code_change():
    # The shape Google has used for every Live release so far, one version further on.
    assert models.is_conversational("gemini-4.2-live")
    assert models.is_conversational("gemini-4.2-live-extended-thinking")


def test_newer_conversational_models_rank_before_older_ones():
    ranked = sorted(
        ["gemini-3.1-flash-live-preview", "gemini-3.8-live", "gemini-4.2-live"],
        key=models._rank,
    )
    assert ranked[-1] == "gemini-3.1-flash-live-preview", "the oldest preview must sort last"
    assert ranked.index("gemini-4.2-live") < ranked.index("gemini-3.8-live"), \
        "a released 4.2 must outrank a released 3.8"


def test_a_release_beats_a_preview_of_the_same_version():
    ranked = sorted(["gemini-3.9-live-preview", "gemini-3.9-live"], key=models._rank)
    assert ranked[0] == "gemini-3.9-live"


def test_single_purpose_families_are_never_dialled():
    # Each of these answers bidiGenerateContent on a real key; none of them holds a call.
    for name in ("gemini-3.5-transcribe-live", "gemini-3.5-live-translate-preview",
                 "gemini-robotics-er-2-streaming-preview", "genie-4"):
        assert not models.is_conversational(name), name


def test_a_future_single_purpose_family_is_caught_by_shape():
    # Not a list of known ids: the words that make a model wrong for a voice call.
    assert not models.is_conversational("gemini-5-live-transcribe")
    assert not models.is_conversational("gemini-5-live-tts")


def test_the_thinking_requirement_follows_the_family():
    assert models.needs_thinking_level("gemini-3.8-live-extended-thinking")
    assert models.needs_thinking_level("gemini-4.2-live-extended-thinking")
    assert not models.needs_thinking_level("gemini-3.8-live")
    assert not models.needs_thinking_level("gemini-4.2-live")


def test_an_unforeseen_naming_scheme_still_yields_models(monkeypatch):
    """Degrade to "try them", never to "no models at all"."""
    monkeypatch.setattr(models, "catalog", lambda refresh=False: ["weird-live-thing"])
    assert models.voice_models() == ["weird-live-thing"]
    assert models.choices() == ("weird-live-thing",)


def test_an_offline_catalog_never_leaves_the_lane_empty(monkeypatch):
    def boom(*a, **kw):
        raise OSError("no network")

    monkeypatch.setattr(models, "config", type("C", (), {"resolve_key": staticmethod(lambda: "k"),
                                                          "ConfigError": Exception}))
    monkeypatch.setattr(models, "_fetch", boom)
    models._cache.update({"at": 0.0, "key": "", "models": []})
    assert models.catalog() == list(models._LAST_RESORT)


def test_the_fallback_order_comes_from_the_catalog_not_a_list(monkeypatch):
    monkeypatch.setattr(models, "voice_models",
                        lambda refresh=False: ["new-a", "new-b", "new-c"])
    assert models.fallback_order("new-b") == ["new-a", "new-c"]


@pytest.mark.parametrize("name,expected", [
    ("gemini-3.8-live", (3, 8)),
    ("gemini-4-live", (4, 0)),
    ("gemini-2.5-flash-native-audio-latest", (2, 5)),
    ("nameless", (-1, -1)),
])
def test_version_key(name, expected):
    assert models._version(name) == expected


def test_a_live_model_called_something_else_is_still_found(monkeypatch):
    """The catalog must not require the word "live" in the id.

    A future family can be named "realtime" or anything else; the only reliable signal is the
    protocol the API says the model speaks. Requiring "live" in the name would silently drop
    a real model the day that happens.
    """
    payload = {"models": [
        {"name": "models/gemini-4-pro-realtime",
         "supportedGenerationMethods": ["bidiGenerateContent", "countTokens"]},
        {"name": "models/gemini-4-pro",                       # not a Live model at all
         "supportedGenerationMethods": ["generateContent"]},
    ]}

    class FakeResponse:
        def read(self):
            import json
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    assert models._fetch("key") == ["gemini-4-pro-realtime"]
    assert models.is_conversational("gemini-4-pro-realtime")


def test_the_catalog_carries_every_live_capable_model_not_a_subset(monkeypatch):
    payload = {"models": [
        {"name": "models/gemini-4.2-live", "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/genie-5", "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/gemini-4-tts-live", "supportedGenerationMethods": ["bidiGenerateContent"]},
    ]}

    class FakeResponse:
        def read(self):
            import json
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    names = models._fetch("key")
    assert names == ["gemini-4.2-live", "genie-5", "gemini-4-tts-live"]
    # The catalog keeps them (they ARE Live); only the voice lane filters.
    assert [n for n in names if models.is_conversational(n)] == ["gemini-4.2-live"]
