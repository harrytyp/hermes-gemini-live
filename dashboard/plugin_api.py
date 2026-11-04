"""Backend half of hermes-gemini-live, mounted by the Hermes web server.

Landed at ``/api/plugins/hermes-gemini-live/`` by
``hermes_cli/web_server.py::_mount_plugin_api_routes``, which imports THIS FILE BY PATH.
There is no parent package at that point, so the plugin root goes on ``sys.path`` and
the package is imported by absolute name — the same shim hermes-talk's dashboard half
uses, and the reason this plugin keeps its code in one prefixed package.

Two invariants:

- **The key never leaves this process.** It is read per call, spent only inside
  :mod:`hermes_gemini_live.live_client`, and never enters a response body or a log line.
- **The WebSocket authorizes itself.** Starlette's HTTP middleware never runs for a WS
  upgrade, so the runtime "plugin is disabled" gate cannot protect this endpoint; it
  calls core's canonical upgrade gate rather than inventing a second answer.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from hermes_gemini_live import agent_lane, config, models, relay, session_log, settings, usage  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter()


def _upgrade_authorized(ws: WebSocket) -> bool:
    """Core's own gate, so this endpoint can never drift from dashboard auth.

    ``import_module`` is deliberate: ``from hermes_cli import web_server_chat`` resolves
    through the package attribute and would keep serving the real module even when a
    harness replaced it in ``sys.modules``.

    Every way this can break fails CLOSED, not open, because the route mints a live
    provider session and can start Hermes runs: the module missing (older hosts that
    predate ``web_server_chat`` still serve a real dashboard with real auth), core renaming
    or removing ``_ws_auth_ok``, or the call raising.
    """
    try:
        gate = importlib.import_module("hermes_cli.web_server_chat")
    except ModuleNotFoundError:
        logger.warning("hermes-gemini-live: this host has no hermes_cli.web_server_chat "
                       "auth gate; refusing the upgrade")
        return False
    except Exception as exc:
        logger.warning("hermes-gemini-live: cannot load the dashboard auth gate (%s); "
                       "refusing the upgrade", type(exc).__name__)
        return False
    authorize = getattr(gate, "_ws_auth_ok", None)
    if not callable(authorize):
        logger.warning("hermes-gemini-live: the host no longer exposes _ws_auth_ok; "
                       "refusing the upgrade rather than guessing")
        return False
    try:
        return bool(authorize(ws))
    except Exception as exc:
        logger.warning("hermes-gemini-live: auth gate raised %s; refusing", type(exc).__name__)
        return False


def _profile_of(ws: WebSocket) -> str:
    """Named profile for multiplexed hosts: the route's own ``?profile=``, never env."""
    return (ws.query_params.get("profile") or "").strip()


@router.get("/status")
async def status(profile: str = "") -> dict:
    """Readiness for the panel. Key material is never reported, only presence.

    Logged on every read: when the panel's button goes dark, the question is whether this
    line appears in the backend log at all. Absent here means core's per-home gate
    refused the request before the plugin's router ever ran.

    The agent lane is reported as *configured*, never as *reachable*: probing it is a
    network round trip and this handler runs on the web server's event loop.
    """
    payload = config.status_dict()
    payload["lane"] = {
        "configured": bool(agent_lane.key()),
        "url": agent_lane.base_url(),
        "profile": config.serving_profile() or "default",
        "scope": config.api_server_session_key() or "default",
    }
    # What a call will actually do, decided once per call by the relay: the panel shows it
    # besides the model and voice, so "why is there no transcript" has an answer on screen.
    # Device ids are a client concern, but they belong to the profile's settings: the relay
    # only ships them so the renderer can honour them (deviceId constraint + setSinkId).
    try:
        payload["devices"] = {"input": config.input_device(), "output": config.output_device()}
    except Exception:
        payload["devices"] = {"input": "", "output": ""}
    try:
        payload["policy"] = {"transcript_mode": config.transcript_mode(),
                             "delegation_target": config.delegation_target(),
                             "long_call": config.long_call(),
                             "voice_context": config.include_voice_context()}
    except Exception:
        payload["policy"] = {}
    logger.info(
        "hermes-gemini-live: /status profile=%s ok=%s model=%s lane=%s detail=%r",
        profile or "-",
        payload["ok"],
        payload["model"],
        payload["lane"]["configured"],
        payload["detail"],
    )
    return payload


@router.get("/settings")
async def get_settings() -> dict:
    """The panel's form: every field's control, and the value in force right now.

    ``effective`` is what the lane will actually use (file, then env, then default), so the
    form never shows a value the call would not honour. ``stored`` is only what the panel
    itself saved — the difference is what a "reset" would reveal.
    """
    return {"fields": settings.schema(), "values": settings.effective(),
            "stored": settings.stored(), "path": str(settings.settings_path())}


@router.put("/settings")
async def put_settings(payload: dict) -> dict:
    """Save the form. All-or-nothing: one refused field leaves the previous values whole."""
    try:
        values = settings.update(payload or {})
    except settings.SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500,
                            detail=f"settings could not be written ({type(exc).__name__})") from exc
    logger.info("hermes-gemini-live: settings saved (%s)", ", ".join(sorted((payload or {}).keys())))
    return {"fields": settings.schema(), "values": values, "stored": settings.stored(),
            "path": str(settings.settings_path())}


@router.get("/models")
async def available_models() -> dict:
    """Live models this key can run, best first (``ListModels``, cached).

    The dropdown reads this instead of a hardcoded pair — adopted from gemini-live-bridge,
    which settled the filter (``bidiGenerateContent``, no transcribe/translate) and order.
    """
    return {"models": list(models.choices()), "default": config.model()}


@router.get("/usage")
async def usage_summary() -> dict:
    """Call count, duration, tokens and an estimated cost — numbers only, stored locally.

    Adopted from hermes-live's ledger. The figure is an estimate from published rates at
    ``price_date``, not a bill; ``unpriced`` counts the calls whose report was too
    incomplete to price, and delegated Hermes work is never in it.
    """
    return usage.summary()


#: Reload order: leaves first, ``relay`` last. A module object is mutated in place, so the
#: imports other code already holds stay valid — and reloading relay last rebinds the names
#: it imported by value (``ConfigError`` and friends), which would otherwise go stale.
_RELOAD_ORDER = ("config", "settings", "models", "usage", "transcript", "wire",
                 "live_client", "tools", "session_log", "agent_lane", "relay")


@router.post("/reload")
async def reload_plugin() -> dict:
    """Re-import this plugin's own modules, so a code change needs no app restart.

    Deliberately narrow: only ``hermes_gemini_live.*`` is reloaded (never core), and only
    between calls — swapping module state under a live relay would break the call that is
    using it, which is the failure mode a restart "solves" by throwing everything away.
    """
    if relay.active_calls():
        raise HTTPException(status_code=409,
                            detail="Ein Anruf läuft gerade — Neuladen erst nach dem Auflegen.")
    import importlib
    reloaded: list[str] = []
    failed: dict[str, str] = {}
    for name in _RELOAD_ORDER:
        try:
            module = importlib.import_module(f"hermes_gemini_live.{name}")
            importlib.reload(module)
            reloaded.append(name)
        except Exception as exc:  # a broken edit must leave the old modules answering
            failed[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("hermes-gemini-live: reload of %s failed (%s)", name, failed[name])
            break
    if failed:
        raise HTTPException(status_code=500, detail={"reloaded": reloaded, "failed": failed})
    logger.info("hermes-gemini-live: modules reloaded (%s)", ", ".join(reloaded))
    return {"reloaded": reloaded}


@router.websocket("/relay")
async def live_relay(ws: WebSocket) -> None:
    if not _upgrade_authorized(ws):
        logger.warning("hermes-gemini-live: refused an unauthorized /relay upgrade")
        await ws.close(code=4401)
        return
    if _profile_of(ws):
        logger.debug("hermes-gemini-live: relay serving profile %s", _profile_of(ws))
    await ws.accept()
    try:
        # No client-supplied profile reaches the lane: relay derives it from the home this
        # process actually serves. The query value is log-only, to spot a mis-pooled backend.
        await relay.run_relay(ws, session_key=config.api_server_session_key())
    except WebSocketDisconnect:
        logger.debug("hermes-gemini-live: browser went away mid-call")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
