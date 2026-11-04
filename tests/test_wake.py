"""Owning the wake word: one lease, two claimants, no second detector.

Core's detector is a single process-wide lease. These tests pin the two outcomes that
matter: the plugin refuses when core holds it, and it takes the lease when core is off.
The detector itself is stubbed — no microphone, no engine install, no model download.
"""

from __future__ import annotations

import sys
import types

import pytest

from hermes_gemini_live import wake


class _WakeWordInUse(RuntimeError):
    pass


def _fake_tools(monkeypatch, *, busy: bool, problems: dict | None = None):
    calls: dict = {"started": 0, "stopped": 0, "on_wake": None}

    def start(on_wake, *, owner, config=None, external_audio=False):
        if busy:
            raise _WakeWordInUse("Wake-word microphone is already owned.")
        calls["started"] += 1
        calls["on_wake"] = on_wake
        calls["config"] = config
        return object()

    def stop(*, owner):
        calls["stopped"] += 1
        return True

    module = types.SimpleNamespace(
        WakeWordInUse=_WakeWordInUse,
        load_wake_word_config=lambda: {"enabled": False, "capture": "client", "phrase": "hey hermes"},
        check_wake_word_requirements=lambda cfg: problems or {"ok": True},
        start_listening=start,
        stop_listening=stop,
        pause_listening=lambda *, owner: True,
        resume_listening=lambda *, owner: True,
        is_listening=lambda: bool(calls["started"]) and not busy,
        _provider=lambda cfg: "sherpa",
    )
    monkeypatch.setitem(sys.modules, "tools.wake_word", module)
    monkeypatch.setitem(sys.modules, "tools", types.SimpleNamespace(wake_word=module))
    return calls


def test_arming_is_refused_while_the_core_listener_holds_the_microphone(monkeypatch):
    _fake_tools(monkeypatch, busy=True)
    wake.disarm()
    status = wake.arm()
    assert status["armed"] is False
    assert "Hermes" in status["reason"] or "hört" in status["reason"]
    assert status["pending"] == 0


def test_arming_takes_the_lease_and_a_hit_only_stamps_a_timestamp(monkeypatch):
    calls = _fake_tools(monkeypatch, busy=False)
    wake.disarm()
    status = wake.arm()
    assert status["armed"] is True
    assert calls["config"]["capture"] == "local", "a plugin-armed detector must open the mic itself"
    assert calls["config"]["enabled"] is True

    assert wake.pending() == 0
    calls["on_wake"]()
    assert wake.pending() > 0
    assert wake.take_pending() > 0
    assert wake.pending() == 0, "one hit must start exactly one call"


def test_a_missing_engine_is_reported_not_swallowed(monkeypatch):
    """Nothing listens when every engine fails to start, and the reason names each one."""
    calls = _fake_tools(monkeypatch, busy=False)

    def refuse(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'pypinyin'")
    monkeypatch.setitem(sys.modules["tools.wake_word"].__dict__, "start_listening", refuse)

    wake.disarm()
    status = wake.arm()

    assert status["armed"] is False
    assert "pypinyin" in status["reason"], "the real import error must reach the panel"


def test_a_second_engine_is_tried_when_the_first_cannot_start(monkeypatch):
    """sherpa without pypinyin must not leave the feature dead while porcupine works."""
    calls = _fake_tools(monkeypatch, busy=False)
    tried: list[str] = []

    def start(on_wake, *, owner, config=None, external_audio=False):
        tried.append(str((config or {}).get("provider")))
        if (config or {}).get("provider") == "sherpa":
            raise ModuleNotFoundError("No module named 'pypinyin'")
        calls["started"] += 1
        calls["on_wake"] = on_wake
        return object()

    monkeypatch.setitem(sys.modules["tools.wake_word"].__dict__, "start_listening", start)
    import types
    pm = types.SimpleNamespace(extras=types.SimpleNamespace(available=lambda extra: True))
    pm.available = pm.extras.available
    monkeypatch.setitem(sys.modules, "pm.extras", pm.extras)
    monkeypatch.setitem(sys.modules, "pm", pm)

    wake.disarm()
    status = wake.arm()

    assert status["armed"] is True
    assert status["provider"] != "sherpa"
    assert tried[0] == "sherpa" and len(tried) > 1


def test_disarming_is_idempotent(monkeypatch):
    calls = _fake_tools(monkeypatch, busy=False)
    wake.disarm()
    wake.disarm()
    assert wake.status()["armed"] is False
    assert calls["stopped"] >= 1


# ── turning Hermes' own listener off (the switch that "did not work") ────────────────
def _core_tools(monkeypatch, *, detector, cfg):
    """A stand-in for tools.wake_word plus the config write the core's setters use."""
    import sys, threading, types

    torn: list[str] = []
    wrote: list[tuple] = []

    class _Detector:
        engine = types.SimpleNamespace(close=lambda: torn.append("closed"))

    def _write_config_key(key, value):
        wrote.append((key, value))
        cfg["enabled"] = value

    module = types.SimpleNamespace(
        _detector_lock=threading.RLock(),
        _detector=_Detector() if detector else None,
        load_wake_word_config=lambda: dict(cfg),
        is_listening=lambda: getattr(module, "_detector", None) is not None,
        _teardown_locked=lambda close: (close(), setattr(module, "_detector", None)),
    )
    monkeypatch.setitem(sys.modules, "tools.wake_word", module)
    monkeypatch.setitem(sys.modules, "tools", types.SimpleNamespace(wake_word=module))
    monkeypatch.setitem(sys.modules, "tui_gateway.server",
                        types.SimpleNamespace(_write_config_key=_write_config_key))
    return module, torn, wrote


def test_stopping_the_core_listener_tears_down_and_persists(monkeypatch):
    module, torn, wrote = _core_tools(monkeypatch, detector=True, cfg={
        "enabled": True, "capture": "client", "phrase": "hey hermes"})

    report = wake.stop_core_listener()

    assert report["stopped"] is True
    assert torn == ["closed"], "the live listener must actually be closed, not just forgotten"
    assert wrote == [("wake_word.enabled", False)]
    assert report["config"] is True
    assert report["verified"] is True, "success is read back, not assumed"
    assert report["core"] == {"enabled": False, "listening": False}


def test_stopping_reports_when_nothing_was_running(monkeypatch):
    module, torn, wrote = _core_tools(monkeypatch, detector=False, cfg={"enabled": False})

    report = wake.stop_core_listener()

    assert report["stopped"] is False
    assert report["config"] is True and report["verified"] is True
    assert torn == []


def test_a_refused_config_write_is_not_reported_as_success(monkeypatch):
    import sys, threading, types
    module, _, _ = _core_tools(monkeypatch, detector=False, cfg={"enabled": True})

    def refuse(key, value):
        raise RuntimeError("locked")
    monkeypatch.setitem(sys.modules, "tui_gateway.server",
                        types.SimpleNamespace(_write_config_key=refuse))

    report = wake.stop_core_listener()

    assert report["config"] is False
    assert report["verified"] is False


def test_after_stopping_the_plugin_can_arm(monkeypatch):
    import sys, types
    module, _, _ = _core_tools(monkeypatch, detector=True, cfg={"enabled": True})

    def start(on_wake, *, owner, config=None, external_audio=False):
        if getattr(module, "_detector", None) is not None:
            raise _WakeWordInUse("already owned")
        module._detector = types.SimpleNamespace(engine=types.SimpleNamespace(close=lambda: None))
        return module._detector

    module.WakeWordInUse = _WakeWordInUse
    module.check_wake_word_requirements = lambda cfg: {"ok": True}
    module.start_listening = start
    module.stop_listening = lambda *, owner: True
    module.pause_listening = lambda *, owner: True
    module.resume_listening = lambda *, owner: True

    wake.disarm()
    assert wake.arm()["armed"] is False, "core's listener holds the microphone"

    wake.stop_core_listener()
    assert wake.arm()["armed"] is True, "once core is off, this lane may listen"


# ── the engine is not shipped with Hermes: it has to be installable on other machines ──
def test_install_engine_verifies_instead_of_trusting_pip(monkeypatch):
    import sys, types
    calls: list[list[str]] = []

    monkeypatch.setitem(sys.modules, "pm.extras", types.SimpleNamespace(available=lambda name: False))
    monkeypatch.setitem(sys.modules, "pm", types.SimpleNamespace(extras=sys.modules["pm.extras"]))

    class Done:
        returncode = 0
        stdout = "Successfully installed pyopen-wakeword-1.1.0"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a[0]) or Done())

    report = wake.ensure_engine()

    assert calls and "pyopen-wakeword==1.1.0" in calls[0]
    assert report["verified"] is False, "pip's exit code is not proof that the anchor imports"
    assert "fehlt weiterhin" in report["detail"]


def test_install_engine_is_a_no_op_when_the_anchor_already_imports(monkeypatch):
    import sys, types
    monkeypatch.setitem(sys.modules, "pm.extras", types.SimpleNamespace(available=lambda name: True))
    monkeypatch.setitem(sys.modules, "pm", types.SimpleNamespace(extras=sys.modules["pm.extras"]))
    report = wake.ensure_engine()
    assert report == {"installed": True, "verified": True, "detail": "Engine war schon da."}
