"""The dashboard half loads the way the host loads it, and answers /status safely."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
_API = _ROOT / "dashboard" / "plugin_api.py"
_PREFIX = "/api/plugins/hermes-gemini-live"


def _load_plugin_api():
    """Byte-for-byte the host's mount path: importlib by file, no parent package."""
    name = "hermes_dashboard_plugin_hermes-gemini-live"
    spec = importlib.util.spec_from_file_location(name, _API)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-material-1234")
    module = _load_plugin_api()
    app = FastAPI()
    app.include_router(module.router, prefix=_PREFIX)
    return TestClient(app), module


def test_api_file_exists_and_declares_a_router():
    assert _API.is_file()
    assert _load_plugin_api().router is not None


def test_routes_are_mounted_at_the_namespace_the_panel_uses(client):
    _, module = client
    paths = {route.path for route in module.router.routes}
    assert "/status" in paths and "/relay" in paths


def test_status_reports_readiness_without_key_material(client):
    http, _ = client
    body = http.get(f"{_PREFIX}/status")
    assert body.status_code == 200
    payload = body.json()
    assert payload["ok"] is True
    assert payload["model"]
    assert "test-key-material" not in body.text


def test_status_explains_a_missing_key(client, monkeypatch):
    http, module = client
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    payload = http.get(f"{_PREFIX}/status").json()
    assert payload["ok"] is False
    assert "GEMINI_API_KEY" in payload["detail"]
    assert module.config.status_dict()["thinking_level"] in (None, "high", "low")


@pytest.mark.parametrize("verdict", [True, False])
def test_relay_authorizes_through_core_gate_not_a_private_copy(client, monkeypatch, verdict):
    # The invariant is the delegation, not the verdict: a WS route that invented its own
    # auth would silently drift from dashboard auth the first time core changes it.
    http, module = client
    seen = []

    class Gate:
        @staticmethod
        def _ws_auth_ok(ws):
            seen.append(ws)
            return verdict

    monkeypatch.setitem(sys.modules, "hermes_cli.web_server_chat", Gate)
    marker = object()
    assert module._upgrade_authorized(marker) is verdict
    assert seen == [marker]


def test_authorize_gate_refuses_when_the_host_has_no_gate_module(client, monkeypatch):
    # Hosts that predate hermes_cli.web_server_chat still serve a real, authenticated
    # dashboard; a missing gate there must not open the relay to any local process.
    http, module = client

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    assert module._upgrade_authorized(object()) is False


def test_authorize_gate_refuses_when_the_host_moved_the_symbol(client, monkeypatch):
    # Fail-closed on drift: the route mints a live provider session, and silently
    # trusting every upgrade after a core rename would be the worst way to notice.
    http, module = client

    class Renamed:
        pass

    monkeypatch.setattr(module.importlib, "import_module", lambda name: Renamed())
    assert module._upgrade_authorized(object()) is False


def test_authorize_gate_refuses_when_the_gate_itself_raises(client, monkeypatch):
    http, module = client

    class Exploding:
        @staticmethod
        def _ws_auth_ok(ws):
            raise RuntimeError("backend gone")

    monkeypatch.setattr(module.importlib, "import_module", lambda name: Exploding())
    assert module._upgrade_authorized(object()) is False


def test_settings_route_answers_the_form_the_panel_draws(client):
    api, _module = client
    payload = api.get(f"{_PREFIX}/settings").json()
    fields = {field["name"]: field for field in payload["fields"]}
    assert fields["transcript_mode"]["choices"] == ["delegations", "utterances", "full"]
    assert fields["delegation_target"]["choices"] == ["call_session", "own_session"]
    assert payload["values"]["transcript_mode"] == "delegations"
    assert payload["path"].endswith("settings.json")


def test_saving_settings_is_visible_on_the_next_read(client, tmp_path):
    api, _module = client
    response = api.put(f"{_PREFIX}/settings",
                       json={"transcript_mode": "full", "voice": "Kore", "silence_ms": 900})
    assert response.status_code == 200
    assert response.json()["values"]["transcript_mode"] == "full"
    again = api.get(f"{_PREFIX}/settings").json()
    assert again["values"]["voice"] == "Kore"
    assert again["values"]["silence_ms"] == 900
    assert (tmp_path / "plugin-data" / "hermes-gemini-live" / "settings.json").is_file()


def test_a_refused_value_is_a_400_and_changes_nothing(client):
    api, _module = client
    api.put(f"{_PREFIX}/settings", json={"long_call": "hang_up"})
    response = api.put(f"{_PREFIX}/settings", json={"long_call": "explode"})
    assert response.status_code == 400
    assert "long_call" in response.json()["detail"] or "Wenn die Session" in response.json()["detail"]
    assert api.get(f"{_PREFIX}/settings").json()["values"]["long_call"] == "hang_up"


def test_status_reports_the_policy_the_call_will_use(client):
    api, _module = client
    api.put(f"{_PREFIX}/settings", json={"transcript_mode": "utterances"})
    policy = api.get(f"{_PREFIX}/status").json()["policy"]
    assert policy["transcript_mode"] == "utterances"
    assert policy["delegation_target"] == "own_session"


def test_models_route_answers_a_usable_dropdown(client):
    api, _module = client
    payload = api.get(f"{_PREFIX}/models").json()
    # Without a reachable Google endpoint the proven pair is the honest answer.
    assert payload["models"]
    assert payload["default"] in payload["models"]


def test_usage_route_is_numeric_only(client):
    api, _module = client
    payload = api.get(f"{_PREFIX}/usage").json()
    assert payload["all_time"]["calls"] == 0
    assert payload["price_date"]
    assert set(payload) == {"last_24h", "all_time", "recent", "price_date", "price_url"}


def test_status_reports_the_devices_the_renderer_must_use(client):
    api, _module = client
    api.put(f"{_PREFIX}/settings", json={"input_device": "mic-1", "output_device": "spk-2"})
    devices = api.get(f"{_PREFIX}/status").json()["devices"]
    assert devices == {"input": "mic-1", "output": "spk-2"}
