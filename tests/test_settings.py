"""Settings: precedence, validation, and the promise that a broken file never fails a call."""

import json

import pytest

from hermes_gemini_live import config, settings


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_home", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_LIVE_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_LIVE_VOICE", raising=False)
    monkeypatch.delenv("GEMINI_LIVE_SILENCE_MS", raising=False)
    return tmp_path


def write(tmp_path, payload):
    path = tmp_path / "plugin-data" / settings.PLUGIN_ID / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_defaults_are_the_documented_ones():
    values = settings.effective()
    assert values["transcript_mode"] == "delegations"
    assert values["delegation_target"] == "own_session"
    assert values["model"] == config.DEFAULT_MODEL
    assert values["voice"] == config.DEFAULT_VOICE
    assert values["silence_ms"] == config.DEFAULT_SILENCE_MS


def test_saved_value_beats_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_LIVE_MODEL", settings.MODEL_CHOICES[1])
    settings.update({"model": settings.MODEL_CHOICES[0]})
    assert settings.value("model") == settings.MODEL_CHOICES[0]
    assert config.model() == settings.MODEL_CHOICES[0]


def test_environment_still_answers_when_the_file_is_silent(monkeypatch):
    monkeypatch.setenv("GEMINI_LIVE_SILENCE_MS", "900")
    # The file says nothing, so the settings lookup says nothing either — the environment
    # reader in config keeps its own refusals (a bad value must fail loudly, not be replaced).
    assert settings.value("silence_ms") is None
    assert config.silence_ms() == 900
    assert settings.effective()["silence_ms"] == 900


def test_unknown_choices_are_refused_before_anything_is_written(tmp_path):
    with pytest.raises(settings.SettingsError):
        settings.update({"transcript_mode": "everything"})
    assert not (tmp_path / "plugin-data" / settings.PLUGIN_ID / "settings.json").exists()


def test_one_bad_field_refuses_the_whole_save(tmp_path):
    settings.update({"voice": "Kore"})
    with pytest.raises(settings.SettingsError):
        settings.update({"voice": "Puck", "silence_ms": 99})
    assert settings.value("voice") == "Kore"


def test_integer_bounds_are_enforced():
    with pytest.raises(settings.SettingsError):
        settings.update({"silence_ms": config.SILENCE_MS_RANGE[1] + 1})


def test_unknown_keys_are_ignored_not_fatal():
    settings.update({"not_a_field": "x", "voice": "Fenrir"})
    assert settings.value("voice") == "Fenrir"
    assert "not_a_field" not in settings.stored()


def test_a_malformed_file_degrades_to_defaults(tmp_path):
    write(tmp_path, "{ this is not json")
    assert settings.value("transcript_mode") is None
    assert config.transcript_mode() == "delegations"


def test_a_stored_value_that_no_longer_validates_falls_back(monkeypatch):
    settings.update({"transcript_mode": "full"})
    # A hand-edited file naming a mode this build never accepted.
    settings._write({"transcript_mode": "everything"})
    assert settings.value("transcript_mode") is None
    assert config.transcript_mode() == "delegations"


def test_schema_carries_choices_for_every_dropdown():
    fields = {field["name"]: field for field in settings.schema()}
    assert fields["transcript_mode"]["choices"] == list(settings.TRANSCRIPT_MODES)
    assert fields["long_call"]["choices"] == list(settings.LONG_CALL_MODES)
    assert fields["voice"]["choices"] == list(config.LIVE_VOICES)
    assert fields["silence_ms"]["min"] == config.SILENCE_MS_RANGE[0]
