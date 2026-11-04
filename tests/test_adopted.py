"""Adopted pieces: the model catalog, the usage ledger, and the durable task board."""

from __future__ import annotations

import json

import pytest

from hermes_gemini_live import agent_lane, config, models, settings, usage


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_home", lambda: tmp_path)
    return tmp_path


# ── model catalog (gemini-live-bridge) ───────────────────────────────────────────────
def test_catalog_falls_back_to_the_proven_pair_without_a_key(monkeypatch):
    monkeypatch.setattr(config, "resolve_key", lambda: (_ for _ in ()).throw(
        config.ConfigError("no key")))
    assert models.catalog() == [config.DEFAULT_MODEL, config.DEFAULT_MODEL + "-extended-thinking"]


def test_catalog_filters_and_ranks_what_the_key_reports(monkeypatch):
    payload = {"models": [
        {"name": "models/gemini-9-live-preview", "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/gemini-3.8-live", "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/gemini-3.8-live-extended-thinking",
         "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/gemini-live-transcribe", "supportedGenerationMethods": ["bidiGenerateContent"]},
        {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
    ]}
    monkeypatch.setattr(models, "_fetch", lambda key: [
        m["name"].split("/", 1)[1] for m in payload["models"]
        if "live" in m["name"].lower()
        and "bidiGenerateContent" in m["supportedGenerationMethods"]
        and "transcribe" not in m["name"]
    ])
    monkeypatch.setattr(config, "resolve_key", lambda: "key-material-1234")
    models._cache.update({"at": 0.0, "key": "", "models": []})
    names = models.catalog()
    assert names[0] == "gemini-3.8-live"
    assert names[1].endswith("-extended-thinking")
    assert all("transcribe" not in name for name in names)


# ── usage ledger (hermes-live) ───────────────────────────────────────────────────────
def _report(prompt_audio: int, response_audio: int) -> dict:
    return {"promptTokenCount": prompt_audio, "candidatesTokenCount": response_audio,
            "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": prompt_audio}],
            "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": response_audio}]}


def test_usage_prices_a_complete_report(tmp_path):
    call = usage.Usage("gemini-3.8-live", started=1000.0)
    call.tokens(_report(1_000_000, 0))
    assert call.unpriced is False
    assert call.usd == pytest.approx(3.0)  # 1M audio input at $3


def test_usage_says_unpriced_when_the_breakdown_is_missing():
    call = usage.Usage("gemini-3.8-live", started=1000.0)
    call.tokens({"promptTokenCount": 5000, "candidatesTokenCount": 900})
    assert call.unpriced is True
    assert call.usd == 0.0


def test_usage_says_unpriced_for_an_unknown_model():
    call = usage.Usage("gemini-unknown-live", started=1000.0)
    call.tokens(_report(10, 10))
    assert call.unpriced is True


def test_a_finished_call_lands_in_the_ledger_and_the_totals(tmp_path):
    call = usage.Usage("gemini-3.8-live", started=1000.0)
    call.tokens(_report(2_000_000, 100_000))
    usage.record(call.close())
    summary = usage.summary()
    assert summary["all_time"]["calls"] == 1
    assert summary["all_time"]["input"] == 2_000_000
    assert summary["recent"][0]["model"] == "gemini-3.8-live"
    assert json.loads((tmp_path / "plugin-data" / "hermes-gemini-live" / "usage.json")
                      .read_text(encoding="utf-8"))["records"]


def test_only_numbers_are_persisted(tmp_path):
    call = usage.Usage("gemini-3.8-live", started=1000.0)
    call.tokens(_report(10, 10))
    usage.record(call.close())
    blob = (tmp_path / "plugin-data" / "hermes-gemini-live" / "usage.json").read_text(encoding="utf-8")
    for forbidden in ("audio", "transcript", "key", "device"):
        assert forbidden not in blob.lower()


# ── durable board (hermes-live-voice) ───────────────────────────────────────────────
def test_the_board_survives_the_process_that_wrote_it(tmp_path):
    path = tmp_path / "calls" / "gemini_live_x.json"
    book = agent_lane.RunBook(path)
    book.register("abc", "create the report")
    book.set_state("abc", "completed", result="done")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tasks"][0]["state"] == "completed"
    assert saved["tasks"][0]["result"] == "done"


def test_a_queued_task_counts_as_open(tmp_path):
    book = agent_lane.RunBook(tmp_path / "board.json")
    book.register("abc", "x")
    book.set_state("abc", "queued")
    assert book.pending() == ["abc"]


def test_exclusive_policy_queues_the_second_task(monkeypatch):
    monkeypatch.setattr(config, "concurrency", lambda: "exclusive")
    scheduler = agent_lane._Scheduler()
    ran = []
    assert scheduler.admit(lambda: ran.append("first")) is True
    assert scheduler.admit(lambda: ran.append("second")) is False
    assert ran == ["first"]
    scheduler.finished()
    assert ran == ["first", "second"]


def test_parallel_policy_starts_both(monkeypatch):
    monkeypatch.setattr(config, "concurrency", lambda: "parallel")
    scheduler = agent_lane._Scheduler()
    ran = []
    assert scheduler.admit(lambda: ran.append("first")) is True
    assert scheduler.admit(lambda: ran.append("second")) is True
    assert ran == ["first", "second"]
