"""REST + WebSocket API for running and inspecting deliberations.

Intended for local use. It binds to 127.0.0.1 by default and has no user
accounts; set ``SYNAPSI_API_TOKEN`` to require a bearer token for starting
runs. Requests may choose a strategy and mode but never models, URLs, or keys:
those come only from the server-side configuration.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from synapsi.agents.roles import list_roles
from synapsi.config import build_council, load_config
from synapsi.core.problem import Problem
from synapsi.council import Council
from synapsi.modes import MODES
from synapsi.observability.events import Event
from synapsi.result import SynapSIResult
from synapsi.strategies import list_strategies

RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    options: list[str] | None = Field(default=None, max_length=12)
    context: str | None = Field(default=None, max_length=20000)
    facts: list[str] = Field(default_factory=list, max_length=20)
    strategy: str | None = None
    mode: Literal["fast", "balanced", "deep"] | None = None
    seed: int | None = None


class RunSummary(BaseModel):
    run_id: str
    status: Literal["running", "completed", "failed"]
    question: str
    strategy: str | None = None
    verdict: str | None = None
    answer: str | None = None
    error: str | None = None


@dataclass
class ActiveRun:
    request: RunRequest
    status: Literal["running", "completed", "failed"] = "running"
    events: list[dict[str, Any]] = field(default_factory=list)
    listeners: list[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=list)
    error: str | None = None
    task: asyncio.Task[None] | None = None

    def publish(self, item: dict[str, Any] | None) -> None:
        if item is not None:
            self.events.append(item)
        for queue in self.listeners:
            queue.put_nowait(item)


class RunStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id):
            raise HTTPException(400, "invalid run id")
        return self.directory / f"{run_id}.json"

    def save(self, result: SynapSIResult) -> None:
        result.save(self.path(result.run_id))

    def load(self, run_id: str) -> SynapSIResult | None:
        p = self.path(run_id)
        return SynapSIResult.load(p) if p.exists() else None

    def summaries(self) -> list[RunSummary]:
        out = []
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:200]:
            try:
                r = SynapSIResult.load(p)
            except Exception:
                continue
            out.append(
                RunSummary(
                    run_id=r.run_id,
                    status="completed",
                    question=r.problem.question,
                    strategy=r.metadata.strategy,
                    verdict=r.judgment.verdict.value if r.judgment else None,
                    answer=r.answer,
                )
            )
        return out


def create_app(
    *,
    config_path: Path | None = None,
    runs_dir: Path = Path("runs"),
    static_dir: Path | None = None,
    council_factory: Callable[[str | None, str | None], Council] | None = None,
    max_concurrent_runs: int = 2,
) -> FastAPI:
    store = RunStore(runs_dir)
    active: dict[str, ActiveRun] = {}
    slots = asyncio.Semaphore(max_concurrent_runs)
    token = os.environ.get("SYNAPSI_API_TOKEN")

    def make_council(strategy: str | None, mode: str | None) -> Council:
        if council_factory is not None:
            return council_factory(strategy, mode)
        if config_path is not None:
            return build_council(
                load_config(config_path),
                base_dir=config_path.parent,
                strategy_override=strategy,
                mode_override=mode,
            )
        return Council.preset(mode or "fast", "mock", strategy=strategy or "independent_panel")

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(401, "missing or invalid bearer token")

    app = FastAPI(
        title="SynapSI", version="0.1", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        council = make_council(None, None)
        return {
            "strategies": list_strategies(),
            "modes": list(MODES),
            "roles": [r.key for r in list_roles()],
            "council": council.describe(),
            "auth_required": token is not None,
        }

    @app.post("/api/runs", dependencies=[Depends(authorize)])
    async def start_run(req: RunRequest) -> RunSummary:
        if req.strategy is not None and req.strategy not in list_strategies():
            raise HTTPException(422, f"unknown strategy {req.strategy!r}")
        if slots.locked():
            raise HTTPException(429, "too many concurrent runs")
        council = make_council(req.strategy, req.mode)
        problem = Problem(
            question=req.question, context=req.context, options=req.options, facts=req.facts
        )
        run = ActiveRun(request=req)
        run_id = f"run_{os.urandom(6).hex()}"
        active[run_id] = run

        def on_event(event: Event) -> None:
            run.publish(event.model_dump(mode="json"))

        async def execute() -> None:
            async with slots:
                try:
                    result = await council.run(
                        problem, run_id=run_id, seed=req.seed, event_handlers=[on_event]
                    )
                    store.save(result)
                    run.status = "completed"
                except Exception as exc:
                    run.status = "failed"
                    run.error = f"{type(exc).__name__}: {exc}"
                finally:
                    run.publish(None)

        run.task = asyncio.create_task(execute())
        return RunSummary(
            run_id=run_id, status="running", question=req.question, strategy=req.strategy
        )

    @app.get("/api/runs")
    async def list_runs() -> list[RunSummary]:
        running = [
            RunSummary(
                run_id=rid,
                status=r.status,
                question=r.request.question,
                strategy=r.request.strategy,
                error=r.error,
            )
            for rid, r in active.items()
            if r.status != "completed"
        ]
        return running + store.summaries()

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        result = store.load(run_id)
        if result is not None:
            return {"status": "completed", "result": result.model_dump(mode="json")}
        run = active.get(run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return {"status": run.status, "error": run.error, "events": run.events}

    @app.websocket("/api/runs/{run_id}/stream")
    async def stream(ws: WebSocket, run_id: str) -> None:
        await ws.accept()
        run = active.get(run_id)
        if run is None:
            await ws.send_json({"type": "end", "status": "unknown"})
            await ws.close()
            return
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        for past in run.events:
            await ws.send_json({"type": "event", "event": past})
        if run.status == "running":
            run.listeners.append(queue)
            try:
                while (item := await queue.get()) is not None:
                    await ws.send_json({"type": "event", "event": item})
            except WebSocketDisconnect:
                return
            finally:
                run.listeners.remove(queue)
        await ws.send_json({"type": "end", "status": run.status, "error": run.error})
        await ws.close()

    if static_dir is not None and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app
