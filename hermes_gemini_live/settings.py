"""Settings the panel can change: one schema, one file, one validator.

Everything here is deliberately boring and explicit — the panel renders a dropdown
per field, so a name, its choices and its bounds are the only things it needs. A
value is stored as JSON under the plugin's data dir, which core mounts as
``$HERMES_HOME/plugin-data/<plugin-id>/``; nothing else in the plugin writes there.

Precedence per field: the settings file, then the environment variable the plugin
already documented, then the built-in default. A file that was never written or
holds an unknown key never fails a call: unknown keys are dropped on save, and a
malformed file degrades to "not stored" (the env/default path) with a warning,
because a settings typo must not be able to take the voice lane down.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

PLUGIN_ID = "hermes-gemini-live"

#: Wire-supported Live model ids the panel may offer. Kept to what the spike actually
#: measured: a third id offered from memory would be a guess the panel makes for the user.
MODEL_CHOICES = (config.DEFAULT_MODEL, config.DEFAULT_MODEL + "-extended-thinking")

TRANSCRIPT_MODES = ("delegations", "utterances", "full")
DELEGATION_TARGETS = ("call_session", "own_session")
LONG_CALL_MODES = ("keep", "auto_restart", "hang_up")
#: Adopted from hermes-live-voice: writes and side effects run one at a time unless the
#: operator explicitly allows overlap, because two agents editing the same repo at once is
#: not a feature. ``parallel`` keeps the old behaviour (up to four workers).
CONCURRENCY_MODES = ("exclusive", "parallel")


@dataclass(frozen=True)
class Field:
    """One setting: what kind of control the panel draws, and what it accepts."""

    name: str
    kind: str  # enum | int | bool | text
    default: Any
    label: str
    help: str = ""
    choices: tuple[str, ...] = ()
    low: int | None = None
    high: int | None = None
    max_len: int = 2000
    env: str = ""

    def to_wire(self) -> dict:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind, "default": self.default,
                               "label": self.label, "help": self.help}
        if self.choices:
            out["choices"] = list(self.choices)
        if self.low is not None:
            out["min"] = self.low
        if self.high is not None:
            out["max"] = self.high
        if self.kind == "text":
            out["max_len"] = self.max_len
        return out


FIELDS: tuple[Field, ...] = (
    Field("transcript_mode", "enum", "delegations",
          "Protokolltiefe",
          "delegations: nur die delegierten Aufgaben; utterances: zusätzlich jede "
          "gesprochene Äußerung als Zeile; full: zusätzlich Steuerbefehle und Status.",
          choices=TRANSCRIPT_MODES),
    Field("delegation_target", "enum", "own_session",
          "Wohin die Arbeit geht",
          "own_session: jede Aufgabe bekommt ihre eigene Session (Standard, weil der "
          "api_server ein Fortsetzen in eine von Hand angelegte Session mit "
          "permission_denied ablehnt); call_session: es wird versucht, die Arbeit in die "
          "Session des Anrufs zu legen — lehnt der Server ab, läuft die Aufgabe in ihrer "
          "eigenen Session weiter statt zu scheitern.",
          choices=DELEGATION_TARGETS),
    Field("include_voice_context", "bool", True,
          "Gesprochenen Kontext mitgeben",
          "Der Arbeiter sieht die letzten gesprochenen Sätze, nicht nur den Auftrag.",
          env=config.ENV_PREFIX + "VOICE_CONTEXT"),
    Field("concurrency", "enum", "exclusive",
          "Aufgaben gleichzeitig",
          "exclusive: eine Aufgabe nach der anderen (sicher bei Schreibzugriffen); "
          "parallel: bis zu vier gleichzeitig, wie bisher.",
          choices=CONCURRENCY_MODES),
    Field("long_call", "enum", "auto_restart",
          "Wenn die Session volläuft",
          "keep: Anruf endet, wie der Server es sagt; auto_restart: einmal mit dem "
          "Fortsetzungs-Handle neu verbinden; hang_up: sofort auflegen.",
          choices=LONG_CALL_MODES),
    Field("model", "enum", config.DEFAULT_MODEL, "Gemini-Modell", "",
          choices=MODEL_CHOICES, env=config.ENV_PREFIX + "MODEL"),
    Field("model_fallback", "bool", True, "Modell-Ausweichen",
          "Wenn das gewählte Modell die Session nicht öffnet: das nächste aus der Liste "
          "versuchen, statt den Anruf abzubrechen."),
    Field("input_device", "text", "", "Mikrofon",
          "Geräte-ID aus der Auswahl. Leer = Systemstandard.", max_len=200),
    Field("output_device", "text", "", "Lautsprecher",
          "Geräte-ID aus der Auswahl. Leer = Systemstandard.", max_len=200),
    Field("voice", "enum", config.DEFAULT_VOICE, "Stimme", "",
          choices=config.LIVE_VOICES, env=config.ENV_PREFIX + "VOICE"),
    Field("thinking_level", "enum", config.DEFAULT_THINKING_LEVEL, "Denkstufe",
          "Nur das extended-thinking-Modell liest diesen Wert.",
          choices=config.THINKING_LEVELS, env=config.ENV_PREFIX + "THINKING_LEVEL"),
    Field("silence_ms", "int", config.DEFAULT_SILENCE_MS, "Stille bis Turn-Ende (ms)",
          "Wie lange es still sein muss, bevor das Modell den Turn als beendet ansieht.",
          low=config.SILENCE_MS_RANGE[0], high=config.SILENCE_MS_RANGE[1],
          env=config.ENV_PREFIX + "SILENCE_MS"),
    Field("prefix_ms", "int", config.DEFAULT_PREFIX_MS, "Vorlauf (ms)",
          "Wie viel Ton vor dem erkannten Sprechbeginn mitgeschnitten wird.",
          low=config.PREFIX_MS_RANGE[0], high=config.PREFIX_MS_RANGE[1],
          env=config.ENV_PREFIX + "PREFIX_MS"),
    Field("echo_cancellation", "bool", True, "Echo-Unterdrückung",
          "Aus ist die häufigste Ursache dafür, dass sich das Modell selbst unterbricht.",
          env=config.ENV_PREFIX + "ECHO_CANCELLATION"),
    Field("noise_suppression", "bool", True, "Rauschunterdrückung", "",
          env=config.ENV_PREFIX + "NOISE_SUPPRESSION"),
    Field("auto_gain", "bool", True, "Automatische Verstärkung", "",
          env=config.ENV_PREFIX + "AUTO_GAIN"),
    Field("half_duplex", "bool", False, "Halbduplex",
          "Mikrofon stumm, während das Modell spricht — für laute Lautsprecher.",
          env=config.ENV_PREFIX + "HALF_DUPLEX"),
    Field("extra_instructions", "text", "", "Zusätzliche Anweisung",
          "Wird an die Systemanweisung angehängt.", env=config.ENV_PREFIX + "INSTRUCTIONS"),
)

BY_NAME = {spec.name: spec for spec in FIELDS}


class SettingsError(ValueError):
    """A refused value, worded so the panel can show it verbatim."""


def _home() -> Path:
    """The Hermes home this process serves, or ``~/.hermes`` when core is unavailable."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        raw = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(raw) if raw else Path.home() / ".hermes"


def settings_path() -> Path:
    return _home() / "plugin-data" / PLUGIN_ID / "settings.json"


def _read_file() -> dict:
    path = settings_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("hermes-gemini-live: settings unreadable (%s); using env/defaults",
                       type(exc).__name__)
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        logger.warning("hermes-gemini-live: settings.json is not JSON; using env/defaults")
        return {}
    return value if isinstance(value, dict) else {}


def stored() -> dict:
    """What the panel last saved, unknown keys dropped. Never raises."""
    data = _read_file()
    return {name: data[name] for name in BY_NAME if name in data}


def _coerce(spec: Field, value: Any) -> Any:
    if spec.kind == "bool":
        if isinstance(value, bool):
            return value
        raise SettingsError(f"{spec.label}: erwartet true oder false")
    if spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{spec.label}: erwartet eine ganze Zahl")
        number = int(value)
        if spec.low is not None and number < spec.low:
            raise SettingsError(f"{spec.label}: {number} ist kleiner als {spec.low}")
        if spec.high is not None and number > spec.high:
            raise SettingsError(f"{spec.label}: {number} ist größer als {spec.high}")
        return number
    if spec.kind == "text":
        if not isinstance(value, str):
            raise SettingsError(f"{spec.label}: erwartet Text")
        text = value.strip()
        if len(text) > spec.max_len:
            raise SettingsError(f"{spec.label}: höchstens {spec.max_len} Zeichen")
        return text
    if not isinstance(value, str) or value.strip() not in spec.choices:
        raise SettingsError(f"{spec.label}: '{value}' ist nicht "
                            f"{', '.join(spec.choices)}")
    return value.strip()


def _from_env(spec: Field) -> Any:
    """The plugin's documented environment variable, read the way config reads it."""
    if not spec.env:
        return None
    raw = (os.environ.get(spec.env) or "").strip()
    if not raw:
        return None
    if spec.kind == "int":
        try:
            raw = int(raw)
        except ValueError:
            return None
    elif spec.kind == "bool":
        folded = raw.lower()
        if folded in ("1", "true", "yes", "on"):
            raw = True
        elif folded in ("0", "false", "no", "off"):
            raw = False
        else:
            return None
    try:
        return _coerce(spec, raw)
    except SettingsError:
        return None


def value(name: str) -> Any:
    """What the panel saved for *name*, or ``None`` when it never saved one.

    Deliberately file-only: the environment already has its own reader in ``config`` that
    refuses bad values loudly (a case-folded voice name must fail a call, not be quietly
    replaced). A settings lookup that also read the environment would swallow exactly
    those refusals — so the precedence lives in ``config`` (file first, then env, then
    default) and this function only ever answers for the file.
    """
    spec = BY_NAME.get(name)
    if spec is None:
        raise KeyError(name)
    data = _read_file()
    if name not in data:
        return None
    try:
        return _coerce(spec, data[name])
    except SettingsError as exc:
        logger.warning("hermes-gemini-live: stored %s refused (%s); falling back", name, exc)
        return None


def effective() -> dict:
    """Every field as the lane will see it: file, then environment, then default.

    This is the panel's view, not the validator's — a bad environment value shows the
    default here, while ``config`` keeps refusing it at call time (and ``status`` says so).
    """
    out: dict[str, Any] = {}
    for spec in FIELDS:
        stored = value(spec.name)
        if stored is not None:
            out[spec.name] = stored
            continue
        from_env = _from_env(spec)
        out[spec.name] = spec.default if from_env is None else from_env
    return out


def schema() -> list[dict]:
    """The panel's field list. The model dropdown asks the key what it can run."""
    out: list[dict] = []
    for spec in FIELDS:
        wire = spec.to_wire()
        if spec.name == "model":
            try:
                from . import models
                wire["choices"] = list(models.choices())
            except Exception:
                pass  # keep the static pair rather than a dropdown with no options
        out.append(wire)
    return out


def update(payload: dict) -> dict:
    """Validate and persist *payload*, returning the effective values afterwards.

    All-or-nothing: one bad field refuses the whole save, so the panel can point at it
    instead of leaving half of a change applied.
    """
    if not isinstance(payload, dict):
        raise SettingsError("erwartet ein Objekt")
    clean: dict[str, Any] = {}
    for name, raw in payload.items():
        spec = BY_NAME.get(name)
        if spec is None:
            continue  # a newer panel talking to an older backend: ignore, do not refuse
        clean[name] = _coerce(spec, raw)
    merged = {**stored(), **clean}
    _write(merged)
    return effective()


def _write(data: dict) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix="settings-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
