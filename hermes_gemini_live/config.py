"""Configuration for hermes-gemini-live. Reads, never installs, never guesses.

Fail-closed everywhere: a key that is SET but blank is a refusal rather than a
fall-through, because a blank key on the Live URL produces an opaque 401 mid-call.
"""

from __future__ import annotations

import os

ENV_PREFIX = "GEMINI_LIVE_"

#: Proven against the live endpoint by the spike: setup accepted, audio returned at
#: audio/pcm;rate=24000. ``...-extended-thinking`` additionally REQUIRES a thinking level
#: ("Thinking level must be specified for this model") and accepts high|low — but hands work
#: to Hermes in only 2 of 5 turns against this model's 8 of 8, which is the difference
#: between a voice lane that acts and one that narrates. See the README table.
DEFAULT_MODEL = "gemini-3.8-live"
THINKING_LEVELS = ("high", "low")
DEFAULT_THINKING_LEVEL = "high"

#: Google's Live voice list. Case-sensitive on the wire, so an unknown or
#: case-folded name refuses instead of sending a value the API may reject. The whitelist is
#: kept short on purpose: the endpoint answers ``setupComplete`` to ANY ``voiceName``, so a
#: typo would silently fall back to a default voice instead of failing — measured with 20
#: candidate names, every one of them "accepted". These five are the documented Live set.
#: ``Aoede`` and ``Kore`` are the two described as female; ``Aoede`` is the softer of them.
LIVE_VOICES = ("Puck", "Charon", "Kore", "Fenrir", "Aoede")
DEFAULT_VOICE = "Aoede"

#: End-of-turn tuning, proven against the endpoint as accepted fields of
#: ``realtimeInputConfig.automaticActivityDetection``. There is NO sensitivity/threshold
#: knob there (``threshold``, ``voiceActivityConfig`` and friends are rejected), so silence
#: length is the only server-side lever — hence the generous default.
DEFAULT_SILENCE_MS = 1200
DEFAULT_PREFIX_MS = 300
SILENCE_MS_RANGE = (400, 5000)
PREFIX_MS_RANGE = (100, 1500)

KEY_CANDIDATES = (ENV_PREFIX + "API_KEY", "GEMINI_API_KEY")

#: The delegate only fires if the model believes it cannot answer itself. Measured: offered
#: the same tool and the same "current time in Tokyo, do not guess" question, plain
#: ``gemini-3.8-live`` called it from the first wording, while
#: ``gemini-3.8-live-extended-thinking`` talked its way to an invented answer until the
#: instruction stated that it holds no present-tense knowledge at all — then it called too.
#: Keep the prohibition, not just the invitation.
DEFAULT_INSTRUCTIONS = (
    "You are Hermes, a spoken assistant. Keep answers short and speakable: no markdown, "
    "no bullet lists, no code blocks, no long paths. You have no knowledge of the present: "
    "no current time or date, no files, no terminal, no web, nothing about this machine. "
    "For anything real or current, your first action is a call to hermes_task — never guess "
    "and never claim you cannot reach it. After calling it, say one brief sentence that you "
    "are on it, then stop talking and wait for the result."
)


class ConfigError(RuntimeError):
    """A configuration refusal, worded to be read aloud or shown as-is."""


def _lookup(name: str) -> str | None:
    """Profile-scoped secret read.

    Under multiplex ``os.environ`` holds the LAUNCH profile's values, so a bare
    ``os.getenv`` would answer for the wrong home. ``agent.secret_scope`` is the
    documented reader; its unscoped path raises ``UnscopedSecretError``, and only
    there is the process env a faithful read of this profile's own ``.env``.
    """
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except ImportError:
        return os.environ.get(name)
    try:
        return get_secret(name)
    except UnscopedSecretError:
        return os.environ.get(name)


def resolve_key() -> str:
    for name in KEY_CANDIDATES:
        raw = _lookup(name)
        if raw is None:
            continue
        value = raw.strip()
        if not value:
            raise ConfigError(f"{name} is set but empty — set a real key or unset it")
        return value
    raise ConfigError("no Gemini key for Live: set GEMINI_LIVE_API_KEY or GEMINI_API_KEY")


def _setting(name: str):
    """A panel setting, or None when the panel has never spoken.

    Imported late: ``settings`` reads this module for its schema, so a module-level
    import would be a cycle. A settings failure must never take the lane down, so any
    exception degrades to "not stored" and the env/default path answers.
    """
    try:
        from . import settings
        return settings.value(name)
    except Exception:
        return None


def model() -> str:
    stored = _setting("model")
    if stored:
        return str(stored)
    return (os.environ.get(ENV_PREFIX + "MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def voice() -> str:
    stored = _setting("voice")
    if stored:
        return str(stored)
    raw = (os.environ.get(ENV_PREFIX + "VOICE") or DEFAULT_VOICE).strip() or DEFAULT_VOICE
    if raw not in LIVE_VOICES:
        raise ConfigError(
            f"{ENV_PREFIX}VOICE '{raw}' is not a Gemini Live voice "
            f"({', '.join(LIVE_VOICES)}; case-sensitive)"
        )
    return raw


def _ms(name: str, default: int, bounds: tuple[int, int]) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} '{raw}' is not a whole number of milliseconds") from None
    low, high = bounds
    if not low <= value <= high:
        raise ConfigError(f"{name} {value} is outside {low}-{high} ms")
    return value


def silence_ms() -> int:
    stored = _setting("silence_ms")
    if isinstance(stored, int):
        return stored
    return _ms(ENV_PREFIX + "SILENCE_MS", DEFAULT_SILENCE_MS, SILENCE_MS_RANGE)


def prefix_ms() -> int:
    stored = _setting("prefix_ms")
    if isinstance(stored, int):
        return stored
    return _ms(ENV_PREFIX + "PREFIX_MS", DEFAULT_PREFIX_MS, PREFIX_MS_RANGE)


def activity_detection() -> dict:
    """The only turn control this endpoint accepts, in the shape it accepts it.

    Google exposes no speech-sensitivity threshold here; the honest knobs are how long
    silence must run before the turn ends and how much lead-in to keep.
    """
    return {"automaticActivityDetection": {
        "silenceDurationMs": silence_ms(),
        "prefixPaddingMs": prefix_ms(),
    }}


def _flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} '{raw}' is not a boolean")


def _tuned(name: str, default: bool) -> bool:
    stored = _setting(name)
    if isinstance(stored, bool):
        return stored
    return _flag(ENV_PREFIX + name.upper(), default)


def audio_tuning() -> dict:
    """Capture rules the renderer must apply, shipped from here so one profile can
    change them without touching the plugin bundle.

    Echo cancellation defaults ON because it is the actual cause of the model talking to
    itself: on a loudspeaker its own reply reaches the mic, Live's server VAD reads that
    as an interruption, and the reply stops mid-sentence. Half-duplex is the blunt
    fallback for a speaker loud enough to beat the canceller — the uplink mutes while the
    model speaks, which costs talk-over interruption and buys back a call that finishes.
    """
    return {
        "echoCancellation": _tuned("echo_cancellation", True),
        "noiseSuppression": _tuned("noise_suppression", True),
        "autoGainControl": _tuned("auto_gain", True),
        "halfDuplex": _tuned("half_duplex", False),
    }


def thinking_level(model_id: str) -> str | None:
    """The level extended-thinking models demand, or None for models that take none.

    Only ever set for the ids the spike proved require it: sending the field to a
    model that has not been probed with it would guess at a wire value. Measured here,
    because it looks like an available knob and is not: ``gemini-3.8-live`` refuses
    ``thinkingLevel`` outright ("Thinking level is not supported for this model"), parses
    ``thinkingBudget`` but ignores it — thought tokens were 941 with no config at all and 939
    with ``thinkingBudget: -1``, which is the same number — and it already spends ~940 thought
    tokens per turn on its own. So None for this model is its maximum, not an omission.
    """
    if "extended-thinking" not in model_id:
        return None
    stored = _setting("thinking_level")
    if stored:
        return str(stored)
    raw = (os.environ.get(ENV_PREFIX + "THINKING_LEVEL") or DEFAULT_THINKING_LEVEL).strip().lower()
    if raw not in THINKING_LEVELS:
        raise ConfigError(
            f"{ENV_PREFIX}THINKING_LEVEL '{raw}' is not supported "
            f"({', '.join(THINKING_LEVELS)}); this model refuses minimal and auto"
        )
    return raw


def instructions() -> str:
    extra = str(_setting("extra_instructions") or "").strip() or \
        (os.environ.get(ENV_PREFIX + "INSTRUCTIONS") or "").strip()
    return f"{DEFAULT_INSTRUCTIONS} {extra}".strip() if extra else DEFAULT_INSTRUCTIONS


def transcript_mode() -> str:
    """How much of the call is written into the session: delegations | utterances | full."""
    return str(_setting("transcript_mode") or "delegations")


def delegation_target() -> str:
    """Where delegated work lands: the call's own session, or a fresh one per task."""
    return str(_setting("delegation_target") or "active_session")


def long_call() -> str:
    """What to do when the Live session reports it is running out: keep | auto_restart | hang_up."""
    return str(_setting("long_call") or "auto_restart")


def include_voice_context() -> bool:
    """Whether a delegated task carries the recent spoken exchange with it."""
    stored = _setting("include_voice_context")
    return True if stored is None else bool(stored)


def model_fallback() -> bool:
    """Whether a failed session setup may try the next model instead of ending the call."""
    stored = _setting("model_fallback")
    return True if stored is None else bool(stored)


def concurrency() -> str:
    """How many delegated tasks may run at once: exclusive (one) or parallel (four)."""
    return str(_setting("concurrency") or "exclusive")


def wake_provider() -> str:
    """Which engine to ask for: ``auto`` means "whatever Hermes ships and is present"."""
    stored = _setting("wake_provider")
    return str(stored or "auto")


def wake_word() -> bool:
    """Whether this plugin may own "hey Hermes" — only with the core wake word switched off."""
    stored = _setting("wake_word")
    return False if stored is None else bool(stored)


def input_device() -> str:
    """Preferred microphone device id, or '' for the system default."""
    return str(_setting("input_device") or "")


def output_device() -> str:
    """Preferred speaker device id, or '' for the system default."""
    return str(_setting("output_device") or "")


def api_server_session_key() -> str | None:
    """The memory scope delegated runs read and write.

    This is deliberately not inferred from the launch profile: one process serves many
    homes, and a guessed scope would quietly write one profile's memories into another.
    Unset means "no header", which lets the api_server use its own default.
    """
    return (os.environ.get(ENV_PREFIX + "SESSION_KEY") or "").strip() or None


def serving_profile() -> str | None:
    """The profile this process is serving, or None for the default home.

    Derived from the home, not from a query parameter a client could send: the api_server
    resumes an unprefixed run in the DEFAULT profile's store, so guessing wrong here
    quietly writes one profile's memories into another's. ``profiles/<name>`` is the
    on-disk convention; an arbitrary HERMES_HOME has no parent named ``profiles`` and
    therefore reads as the default.
    """
    override = (os.environ.get(ENV_PREFIX + "API_SERVER_PROFILE") or "").strip()
    if override:
        return None if override == "default" else override
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        return None
    return home.name if home.parent.name == "profiles" else None


def status_dict() -> dict:
    """Readiness for the panel. Never contains the key.

    Each part degrades on its own and the first refusal is reported: a bad tuning knob
    must not look like a missing key, and a missing key must not hide a bad voice name.
    """
    problems: list[str] = []
    m = model()
    try:
        v, lvl = voice(), thinking_level(m)
    except ConfigError as exc:
        v, lvl = "", None
        problems.append(str(exc))
    try:
        audio = audio_tuning()
        turn = dict(activity_detection()["automaticActivityDetection"])
    except ConfigError as exc:
        audio, turn = {}, {}
        problems.append(str(exc))
    try:
        key_present = bool(resolve_key())
    except ConfigError as exc:
        key_present = False
        problems.append(str(exc))
    return {
        "ok": key_present and not problems,
        "model": m,
        "voice": v,
        "thinking_level": lvl,
        "key_present": key_present,
        "audio": audio,
        "turn": turn,
        "detail": problems[0] if problems else "",
    }
