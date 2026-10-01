from pathlib import Path

from synapsi import Agent, Council
from synapsi.memory import InMemoryStore, JSONLMemoryStore
from synapsi.providers import MockProvider


async def test_memory_is_recalled_as_prior_conclusions_not_evidence() -> None:
    memory = InMemoryStore()
    provider = MockProvider()
    council = Council([Agent("a", model=provider)], memory=memory, judge="structural")
    await council.run("Should the airline add a second daily flight to Denver?", context="SECRET")
    assert len(memory.records()) == 1
    assert "SECRET" not in memory.records()[0].model_dump_json()

    await council.run("Should the airline add a daily flight to Denver?")
    prompt = provider.requests[-1].messages[-1].content
    assert "PRIOR CONCLUSIONS" in prompt and "not evidence" in prompt


async def test_memory_disabled_by_default() -> None:
    provider = MockProvider()
    council = Council([Agent("a", model=provider)])
    await council.run("q one")
    await council.run("q one")
    assert "PRIOR CONCLUSIONS" not in provider.requests[-1].messages[-1].content


def test_jsonl_store_roundtrip(tmp_path: Path) -> None:
    store = JSONLMemoryStore(tmp_path / "mem.jsonl", min_similarity=0.5)
    from synapsi.memory import MemoryRecord

    store.add(MemoryRecord(run_id="r1", question="hub delay causes", summary="weather"))
    store.add(MemoryRecord(run_id="r2", question="unrelated topic", summary="x"))
    assert [r.run_id for r in store.recall("hub delay causes")] == ["r1"]
    store.clear()
    assert store.records() == []
