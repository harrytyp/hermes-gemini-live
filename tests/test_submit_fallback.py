"""A refused resume must not become a refused task.

The api_server admits a run into a session only when its own principal owns it
(``api_server_response_admissions.py`` → ``permission_denied``, surfaced as HTTP 409 by
``store_error_response``'s default mapping). A call session created through
``POST /api/sessions`` is not such a session, so resuming into it is refused — and the task
must still run.
"""

from __future__ import annotations

import pytest

from hermes_gemini_live import agent_lane, config, settings


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_home", lambda: tmp_path)
    monkeypatch.setenv("API_SERVER_KEY", "test-key-material")


class _Response:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self) -> dict:
        return self._body


def test_a_refused_resume_retries_standalone(monkeypatch):
    seen: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(dict(json or {}))
        if "session_id" in (json or {}):
            return _Response(409, {"error": {"code": "permission_denied",
                                             "message": "permission_denied"}})
        return _Response(202, {"run_id": "run_abc", "status": "queued"})

    monkeypatch.setattr(agent_lane.httpx, "post", fake_post)
    run_id = agent_lane.submit("do the thing", session_id="geminilive_x")

    assert run_id == "run_abc"
    assert seen[0]["session_id"] == "geminilive_x"
    assert "session_id" not in seen[1], "the retry must not address the refused session again"


def test_a_genuine_refusal_is_still_an_error(monkeypatch):
    monkeypatch.setattr(agent_lane.httpx, "post",
                        lambda *a, **kw: _Response(401, {"error": {"message": "bad key"}}))
    with pytest.raises(agent_lane.LaneUnavailable) as excinfo:
        agent_lane.submit("do the thing", session_id="geminilive_x")
    assert "401" in str(excinfo.value)


def test_a_conflict_is_not_silently_swallowed(monkeypatch):
    """Only a permission refusal is retried: a busy/conflicting session is real news."""
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json or {}))
        return _Response(409, {"error": {"code": "admission_conflict",
                                         "message": "admission_conflict"}})

    monkeypatch.setattr(agent_lane.httpx, "post", fake_post)
    with pytest.raises(agent_lane.LaneUnavailable):
        agent_lane.submit("do the thing", session_id="geminilive_x")
    assert len(calls) == 1


def test_the_default_target_is_the_one_that_works_out_of_the_box():
    assert settings.BY_NAME["delegation_target"].default == "own_session"
    assert config.delegation_target() == "own_session"


# ── call_session: adopt the run's own session (root cause, not a workaround) ─────────
def test_rows_wait_until_a_session_exists(tmp_path, monkeypatch):
    from hermes_gemini_live import session_log
    wrote: list = []
    monkeypatch.setattr(session_log.CallLog, "_store", lambda self: type(
        "Store", (), {"append_message": lambda _s, *a: wrote.append(a)})())
    log = session_log.CallLog(None)
    assert log.append([{"role": "user", "content": "hallo"}]) == 0
    assert wrote == [], "nothing may be written before the session exists"
    assert len(log._buffer) == 1


def test_adopting_a_session_flushes_what_was_said_before_it(tmp_path, monkeypatch):
    from hermes_gemini_live import session_log
    written: list[tuple] = []
    monkeypatch.setattr(session_log.CallLog, "_store", lambda self: type(
        "Store", (), {"append_message": lambda _s, sid, role, content: written.append((sid, role, content))})())
    log = session_log.CallLog(None)
    log.append([{"role": "user", "content": "mach das mal"}])
    assert log.adopt("gemini_run_session") is True
    assert written == [("gemini_run_session", "user", "mach das mal")]
    assert log.adopt("another_one") is False, "a call adopts exactly one session"


def test_reload_refuses_while_a_call_is_live(monkeypatch):
    import importlib.util, sys
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_gemini_live import relay, settings

    monkeypatch.setattr(settings, "_home", lambda: Path("."))
    monkeypatch.setattr(relay, "active_calls", lambda: 1)

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_reload_test",
                                                  root / "dashboard" / "plugin_api.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/hermes-gemini-live")
    client = TestClient(app)

    response = client.post("/api/plugins/hermes-gemini-live/reload")
    assert response.status_code == 409
