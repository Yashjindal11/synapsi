import json
from pathlib import Path

from synapsi.core.pricing import PricingTable
from synapsi.core.usage import TokenUsage
from synapsi.observability import Event, EventBus, EventType, JSONLSink, Recorder, UsageTracker
from synapsi.providers import Completion


def test_bus_fans_out_and_isolates_failures(tmp_path: Path) -> None:
    recorder = Recorder()

    def broken(_: Event) -> None:
        raise RuntimeError("subscriber bug")

    bus = EventBus([broken, recorder, JSONLSink(tmp_path / "trace.jsonl")])
    bus.emit(Event(type=EventType.AGENT_STARTED, run_id="r", agent="A"))
    assert len(recorder.of_type(EventType.AGENT_STARTED)) == 1
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["agent"] == "A"


def test_usage_tracker_buckets() -> None:
    tracker = UsageTracker(PricingTable({"p:m": (1.0, 1.0)}))
    c = Completion(text="", model="p:m", usage=TokenUsage(prompt_tokens=10, completion_tokens=10))
    tracker.record(c, agent="A", task="analyze")
    tracker.record(c.model_copy(update={"cached": True}), agent="B", task="analyze")
    report = tracker.report()
    assert report.total.calls == 2 and report.total.cached_calls == 1
    assert report.by_agent["A"].cost_usd == 20 / 1_000_000
    assert report.by_agent["B"].cost_usd == 0.0
    assert report.by_task["analyze"].total_tokens == 40
