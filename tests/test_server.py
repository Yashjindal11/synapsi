from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synapsi import Agent, Council
from synapsi.providers import MockProvider
from synapsi.server import create_app


def _factory(strategy: str | None, mode: str | None) -> Council:
    agents = [Agent("A", model=MockProvider()), Agent("B", "skeptic", MockProvider())]
    return Council(agents, strategy=strategy or "debate", mode=mode or "fast")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(runs_dir=tmp_path, council_factory=_factory))


def test_meta_and_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    meta = client.get("/api/meta").json()
    assert "debate" in meta["strategies"] and meta["auth_required"] is False
    assert all("api_key" not in str(a) for a in meta["council"]["agents"])


def test_run_lifecycle_and_stream(client: TestClient) -> None:
    with client:
        started = client.post("/api/runs", json={"question": "Q?", "options": ["a", "b"]}).json()
        run_id = started["run_id"]
        with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "end":
                    break
        assert messages[-1]["status"] == "completed"
        assert any(m.get("event", {}).get("type") == "judgment_created" for m in messages)
        body = client.get(f"/api/runs/{run_id}").json()
        assert body["status"] == "completed"
        assert body["result"]["metadata"]["strategy"] == "debate"
        assert any(r["run_id"] == run_id for r in client.get("/api/runs").json())


def test_validation_and_path_safety(client: TestClient) -> None:
    assert client.post("/api/runs", json={"question": ""}).status_code == 422
    assert client.post("/api/runs", json={"question": "q", "strategy": "x"}).status_code == 422
    bad = client.post("/api/runs", json={"question": "q", "model": "openai:x"})
    assert bad.status_code == 422
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/api/runs/does_not_exist").status_code == 404


def test_token_required_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNAPSI_API_TOKEN", "t0ken")
    client = TestClient(create_app(runs_dir=tmp_path, council_factory=_factory))
    assert client.post("/api/runs", json={"question": "q"}).status_code == 401
    ok = client.post("/api/runs", json={"question": "q"}, headers={"Authorization": "Bearer t0ken"})
    assert ok.status_code == 200
