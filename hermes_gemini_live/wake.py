"""Own the wake word, instead of fighting the core for it.

Hermes' detector is a single lease in the gateway process: ``start_listening(on_wake,
owner=…)`` hands it to one owner and refuses everybody else with ``WakeWordInUse``. Core
takes that lease when its own wake word is on. The plugin takes the *same* lease with its
own owner object — so the two can never listen at once, which is exactly the mutual
exclusion the settings switch promises, enforced by the code that owns the microphone
rather than by a checkbox.

A hit does not start anything by itself: the detector callback only stamps a timestamp.
The renderer polls it (``GET /wake``) and starts its own call, which is what keeps the
detection on this lane instead of the core voice session.

Everything degrades to a spoken-out-loud reason: a missing engine, a lease held by the
core, a refused arm — the panel shows the text and the switch stays off.

Two processes, one microphone: the plugin's Python half runs in the plugin host, Hermes'
own listener may run in the gateway process. ``_teardown_locked`` can only stop a detector
in *this* process — the cross-process guarantee is the machine lock in tools.wake_word, and
it is what the panel reports: if Hermes' listener lives elsewhere, the switch persists
"off", arming is refused with ``WakeWordInUse``, and the reason says so instead of
pretending the microphone is free.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

#: The lease holder. Deliberately a plain object: core's own owner is the client transport,
#: so this can never collide with it and ``owns_listener`` answers for us alone.
_OWNER = object()

_state: dict = {"armed": False, "reason": "", "since": 0.0}
_pending: dict = {"at": 0.0}


class WakeUnavailable(RuntimeError):
    """The wake word cannot be armed here; the message is meant for the panel."""


def _raw_core_config() -> dict:
    """The stored section as-is. Never forces anything: this is what a report may claim."""
    from tools.wake_word import load_wake_word_config
    return dict(load_wake_word_config() or {})


def _core_config(provider: str | None = None) -> dict:
    """Core's ``wake_word`` section, adjusted to what a plugin can actually honour.

    Two changes, both forced by the situation rather than by taste:

    * ``capture: local`` — ``client`` means "the renderer feeds PCM"; when the plugin arms the
      listener itself, nobody is feeding, so the detector must open the microphone.
    * a provider whose engine is *installed*. Core's ``auto`` picks the first entry that the
      platform supports, not the first one that is present, so a supported-but-missing engine
      wins over an installed one and the wake word silently does nothing. Hermes usually ships
      sherpa or porcupine already; use that instead of installing anything.
    """
    from tools.wake_word import load_wake_word_config

    config = dict(load_wake_word_config() or {})
    config["enabled"] = True
    config["capture"] = "local"

    if provider:
        config["provider"] = provider
        return config

    from . import config as plugin_config
    chosen = str(plugin_config.wake_provider() or "auto").strip().lower()
    if chosen != "auto" and _provider_installed(chosen):
        config["provider"] = chosen
        return config
    # Whatever core would resolve, and whether that engine is actually importable. Core's
    # "auto" answers "supported on this platform", not "installed here", so a missing engine
    # can win over a present one and the wake word does nothing at all.
    try:  # core's own resolution, when it is there to ask
        from tools.wake_word import _provider as core_provider
        resolved = str(core_provider(config))
    except Exception:
        resolved = str(config.get("provider") or "auto")
    if not _provider_installed(resolved):
        fallback = _first_installed_provider()
        if fallback != "auto":
            config["provider"] = fallback
    return config


def _provider_installed(provider: str) -> bool:
    """Whether the engine behind a provider can be imported right now (``auto``: any)."""
    try:
        from pm.extras import available
    except Exception:
        return True  # no pm to ask: let core's own check decide
    extras = {"openwakeword": "wake-openwakeword", "sherpa": "wake-sherpa",
              "sherpa-onnx": "wake-sherpa", "kws": "wake-sherpa", "open": "wake-openwakeword",
              "porcupine": "wake-porcupine"}
    if provider == "auto":
        return any(available(extra) for extra in set(extras.values()))
    return available(extras.get(provider, "wake-openwakeword"))


def _first_installed_provider() -> str:
    """The engine Hermes already has, in the core's own order of preference."""
    try:
        from pm.extras import available
    except Exception:
        return "auto"
    for provider, extra in (("sherpa", "wake-sherpa"),
                            ("openwakeword", "wake-openwakeword"),
                            ("porcupine", "wake-porcupine")):
        if available(extra):
            return provider
    return "auto"


def _persist_enabled(value: bool) -> bool:
    """Write ``wake_word.enabled`` the way the core's own setters do.

    ``config.set`` has no setter for ``wake_word.*`` (it answers ``unknown config key``), so a
    plugin cannot go through the RPC. ``_write_config_key`` is the helper every core setter
    uses: one backend write, lock checks, and it drops the config cache. The backend is used
    directly as a second path, because the server module is not always importable.
    """
    try:
        from tui_gateway.server import _write_config_key
        _write_config_key("wake_word.enabled", bool(value))
        return True
    except Exception as exc:
        logger.info("hermes-gemini-live: gateway config write unavailable (%s)", type(exc).__name__)
    try:
        from hermes_cli.config_backend import get_config_backend, write_config_key
        from hermes_cli.config import config_home
        write_config_key(get_config_backend().config_path(config_home()),
                         "wake_word.enabled", bool(value))
        return True
    except Exception as exc:
        logger.warning("hermes-gemini-live: config write failed (%s: %s)", type(exc).__name__, exc)
        return False


def core_state() -> dict:
    """Whether Hermes' own wake word is enabled and whether a detector is listening now."""
    enabled = None
    try:
        enabled = bool(_raw_core_config().get("enabled"))
    except Exception:
        pass
    listening = False
    try:
        from tools.wake_word import is_listening
        listening = bool(is_listening())
    except Exception:
        pass
    return {"enabled": enabled, "listening": listening}


#: What to try, in order. Hermes ships sherpa and porcupine; openwakeword only when the extra
#: was installed. A provider that imports can still fail to start (a missing transitive dep, a
#: keyword porcupine does not know), so the list is walked until one actually listens.
_PROVIDER_ORDER = ("sherpa", "porcupine", "openwakeword")


def _candidates() -> list[str]:
    try:
        from . import config as plugin_config
        chosen = str(plugin_config.wake_provider() or "auto").strip().lower()
    except Exception:
        chosen = "auto"
    order: list[str] = []
    if chosen != "auto" and _provider_installed(chosen):
        order.append(chosen)
    order.extend(name for name in _PROVIDER_ORDER
                 if name not in order and _provider_installed(name))
    return order


def active_provider() -> str:
    """The engine that actually listens, once one has been started."""
    return str(_state.get("provider") or "")


def requirements() -> dict:
    """What is missing before the detector could run (engine, model, device)."""
    try:
        from tools.wake_word import check_wake_word_requirements
        return dict(check_wake_word_requirements(_core_config()) or {})
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def stop_core_listener() -> dict:
    """Stop whichever listener holds the microphone — Hermes' own included — and persist it.

    Deliberately in-process rather than over an RPC: the plugin runs inside the gateway
    process, so it can reach the detector singleton directly, and ``_teardown_locked`` also
    releases the machine lease (the file lock) that the RPC route cannot reach from here.
    Reaching into a private lock is a trade: the alternative is a public stop-anything call
    that does not exist, and the whole point of this switch is that the other listener ends.

    Reports what happened instead of assuming: ``stopped`` (a listener was running),
    ``config`` (the off state was written), ``reason`` (why not, if not).
    """
    report = {"stopped": False, "config": False, "reason": ""}
    try:
        from tools import wake_word as core
    except Exception as exc:
        report["reason"] = f"Wake-Modul nicht erreichbar ({type(exc).__name__})"
        return report
    try:
        lock = getattr(core, "_detector_lock", None)
        detector = getattr(core, "_detector", None)
        if lock is not None and detector is not None:
            with lock:
                detector = getattr(core, "_detector", None)
                if detector is not None:
                    core._teardown_locked(detector.engine.close)
                    report["stopped"] = True
        elif detector is not None:
            report["reason"] = "Zuhörer läuft in einem anderen Prozess"
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    report["config"] = _persist_enabled(False)
    if not report["config"]:
        report["reason"] = report["reason"] or (
            "Die Einstellung ließ sich nicht schreiben (config nicht erreichbar).")
    # Never claim success from the write alone: read the value back the way the core reads it.
    try:
        report["verified"] = _raw_core_config().get("enabled") is False
    except Exception:
        report["verified"] = False
    if report["config"] and not report["verified"]:
        report["reason"] = report["reason"] or "Einstellung geschrieben, wird aber noch nicht gelesen"
    report["core"] = core_state()
    if report["stopped"]:
        _state.update({"armed": False, "reason": "", "since": 0.0})
        logger.info("hermes-gemini-live: stopped the listener that held the microphone")
    return report


#: The engine package core's ``wake-openwakeword`` extra names. Pinned: it is the version the
#: extra pins, and a floating one would silently change what the detector loads.
ENGINE_REQUIREMENT = "pyopen-wakeword==1.1.0"


def _engine_installed() -> bool:
    try:
        from pm.extras import available
        return bool(available("wake-openwakeword"))
    except Exception:
        return False


def ensure_engine(timeout: float = 240.0) -> dict:
    """Install the wake engine into this install's venv, on an explicit user action.

    ``hermes pm install --extra wake-openwakeword`` is the intended path, but it can report
    success without the anchor becoming importable (its marker and the extras table disagreed
    on this machine). So this does the deterministic thing and then *verifies* by importing
    the anchor, which is exactly what core's own ``pm.available`` checks.

    Never called on load, only from an explicit click or a switch-on that would otherwise fail.
    """
    import subprocess
    import sys

    if _engine_installed():
        return {"installed": True, "verified": True, "detail": "Engine war schon da."}
    try:
        done = subprocess.run([sys.executable, "-m", "pip", "install", ENGINE_REQUIREMENT],
                              capture_output=True, text=True, timeout=timeout)
        tail = (done.stdout or "").strip().splitlines()[-2:]
    except Exception as exc:
        return {"installed": False, "verified": False,
                "detail": f"Installation nicht gestartet ({type(exc).__name__}: {exc})"}
    import importlib
    importlib.invalidate_caches()
    verified = _engine_installed()
    detail = "Engine installiert." if verified else (
        "Installation lief, aber das Modul fehlt weiterhin: " + " / ".join(tail or ["keine Ausgabe"]))
    if verified:
        logger.info("hermes-gemini-live: wake engine installed (%s)", ENGINE_REQUIREMENT)
    return {"installed": done.returncode == 0, "verified": verified, "detail": detail}


def arm() -> dict:
    """Take the microphone lease and listen, with whatever engine Hermes has.

    Walks the installed engines instead of insisting on one: an importable engine can still
    fail to start (sherpa needs pypinyin, porcupine knows only certain keywords), and a
    refusal to start is not a reason to leave the feature dead when another engine works.
    ``WakeWordInUse`` is different: that is the core holding the microphone, and no other
    engine will get past it either — so it stops the walk and says so.
    """
    from tools.wake_word import WakeWordInUse, is_listening, start_listening

    if _state["armed"] and is_listening():
        return status()
    attempts: list[str] = []
    for provider in _candidates():
        config = _core_config(provider)
        try:
            start_listening(_on_wake, owner=_OWNER, config=config)
        except WakeWordInUse:
            # Hermes' own listener holds it. This is the switch's whole point: say so, do not
            # start a second detector on the same microphone.
            return _fail("Hermes' eigenes Wake-Word hört gerade zu — erst dort ausschalten.")
        except Exception as exc:
            attempts.append(f"{provider}: {type(exc).__name__}: {exc}")
            continue
        _state.update({"armed": True, "reason": "", "since": time.time(), "provider": provider})
        logger.info("hermes-gemini-live: wake word armed via %s", provider)
        return status()
    if not attempts:
        return _fail("Keine Wake-Engine installiert (sherpa, porcupine oder openwakeword).")
    return _fail("Keine Engine startete — " + " | ".join(attempts))


def disarm() -> dict:
    """Give the microphone back. Safe to call when nothing is armed."""
    try:
        from tools.wake_word import stop_listening
        stop_listening(owner=_OWNER)
    except Exception as exc:
        logger.info("hermes-gemini-live: wake disarm raised %s", type(exc).__name__)
    _state.update({"armed": False, "reason": "", "since": 0.0})
    return status()


def pause() -> bool:
    """Release the microphone for a call without losing the lease."""
    try:
        from tools.wake_word import pause_listening
        return bool(pause_listening(owner=_OWNER))
    except Exception:
        return False


def resume() -> bool:
    try:
        from tools.wake_word import resume_listening
        return bool(resume_listening(owner=_OWNER))
    except Exception:
        return False


def status() -> dict:
    return {"armed": bool(_state["armed"]), "reason": _state["reason"],
            "since": _state["since"], "pending": _pending["at"],
            "provider": active_provider() or (requirements().get("provider") or ""),
            "candidates": _candidates(),
            "core": core_state(), "requirements": requirements()}


def take_pending() -> float:
    """The timestamp of a detection, once: the renderer starts exactly one call per hit."""
    at, _pending["at"] = _pending["at"], 0.0
    return at


def pending() -> float:
    return _pending["at"]


def _fail(reason: str) -> dict:
    _state.update({"armed": False, "reason": reason, "since": 0.0})
    logger.info("hermes-gemini-live: wake word not armed — %s", reason)
    return status()


def _on_wake() -> None:
    """Detector callback. Runs on the detector's thread; only stamps the hit."""
    _pending["at"] = time.time()
    logger.info("hermes-gemini-live: wake word matched")
