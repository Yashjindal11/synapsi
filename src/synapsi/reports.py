"""Markdown and self-contained HTML reports for a :class:`SynapSIResult`."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synapsi.result import SynapSIResult
    from synapsi.synthesis.models import Finding


def _fmt_cost(cost: float | None) -> str:
    return "unknown (no pricing configured)" if cost is None else f"${cost:.4f}"


def _md_findings(title: str, findings: list[Finding]) -> list[str]:
    lines = [f"### {title}", ""]
    if not findings:
        return [*lines, "_None._", ""]
    for f in findings:
        refs = ", ".join(f.claim_ids + f.evidence_ids)
        note = f" — {f.note}" if f.note else ""
        lines.append(f"- {f.statement}{note}" + (f" `[{refs}]`" if refs else ""))
    return [*lines, ""]


def to_markdown(result: SynapSIResult) -> str:
    j, s, u, m = result.judgment, result.synthesis, result.uncertainty, result.metadata
    out = [f"# SynapSI report: {result.problem.question}", ""]
    out += [
        f"- **Strategy:** {m.strategy} ({m.mode})",
        f"- **Verdict:** {j.verdict.value if j else 'n/a'}"
        + (f" — answer **{j.answer}**" if j and j.answer else ""),
        f"- **Confidence:** {j.confidence:.2f} ({j.method})" if j else "- **Confidence:** n/a",
        f"- **Evidence strength:** {j.evidence_strength.value}" if j else "",
        "",
    ]
    if s:
        out += ["## Synthesis", "", s.summary, ""]
        out += _md_findings("Established", s.established)
        out += _md_findings("Probable", s.probable)
        out += _md_findings("Disputed", s.disputed)
        out += _md_findings("Unknown", s.unknown)
        if s.recommendations:
            out += ["### Recommended next steps", ""] + [f"- {r}" for r in s.recommendations] + [""]
    if j:
        out += ["## Judgment", "", f"**Decision:** {j.decision}", "", j.reasoning_summary, ""]
        if j.minority_positions:
            out += ["**Minority positions:** " + "; ".join(j.minority_positions), ""]
        if j.unresolved_issues:
            out += ["**Unresolved:**", ""] + [f"- {i}" for i in j.unresolved_issues] + [""]
    out += ["## Perspectives", ""]
    for name, p in result.final_perspectives.items():
        tag = "independent" if p.independent else f"revised (round {p.round})"
        out += [
            f"### {name} — {p.role} ({p.model})",
            "",
            f"**Answer:** {p.answer} · **confidence** {p.confidence:.2f} · {tag}",
            "",
            p.position,
            "",
        ]
        if p.reasoning_summary:
            out += [f"_Summary:_ {p.reasoning_summary}", ""]
    out += [
        "## Claims",
        "",
        "| ID | Agent | Type | Status | Statement | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for c in result.claims:
        if c.withdrawn:
            continue
        stmt = c.statement.replace("|", "\\|")
        out.append(
            f"| {c.id} | {c.agent} | {c.type.value} | {c.status.value} | {stmt} | "
            f"{', '.join(c.evidence_ids) or '—'} |"
        )
    out += ["", "## Evidence", ""]
    for e in result.evidence:
        src = e.provenance.source_id or e.provenance.claimed_source or ""
        out.append(
            f"- **{e.id}** [{e.provenance.source_kind.value}] {e.content[:240]}"
            + (f" — _{src}_" if src else "")
        )
    if result.challenges:
        out += ["", "## Challenges", ""]
        rebuttals = {r.challenge_id: r for r in result.rebuttals}
        for ch in result.challenges:
            out.append(
                f"- **{ch.id}** {ch.challenger} → {ch.target_claim_id} ({ch.target_agent}) "
                f"[{ch.kind.value}, {ch.status.value}]: {ch.problem}"
            )
            if (r := rebuttals.get(ch.id)) is not None:
                out.append(f"  - {r.response_type.value}: {r.response}")
    if result.disagreements:
        out += ["", "## Disagreements", ""]
        for d in result.disagreements:
            out.append(f"- **{d.id}** [{d.kind}, {d.status}] {d.topic} — {d.evidence_balance}")
    out += [
        "",
        "## Uncertainty signals",
        "",
        f"- Initial agreement: {u.initial_agreement_rate} → final: {u.agreement_rate}",
        f"- Position changes: {u.position_changes} "
        f"({u.unsupported_position_changes} without new evidence or concession)",
        f"- External evidence items: {u.external_evidence}; model-knowledge items: "
        f"{u.model_knowledge_items}",
        f"- Shared-source ratio: {u.source_dependence.shared_source_ratio:.2f}",
        f"- Unresolved disagreements: {u.unresolved_disagreements}",
        f"- Invalid references ignored: {u.invalid_references}",
        "",
        "## Run metadata",
        "",
        f"- Run: `{result.run_id}` · SynapSI {m.synapsi_version} · prompts v{m.prompt_version}",
        f"- Model calls: {m.usage.total.calls} · tokens: {m.usage.total.total_tokens} · "
        f"cost: {_fmt_cost(m.usage.total.cost_usd)} · wall time: {m.latency_s:.2f}s",
        "- Agents: " + ", ".join(f"{a['name']} ({a['model']})" for a in m.agents),
    ]
    if m.stopped_reason:
        out.append(f"- Stopped early: {m.stopped_reason}")
    if m.errors:
        out += ["- Errors:"] + [f"  - {e}" for e in m.errors]
    return "\n".join(line for line in out if line is not None) + "\n"


_CSS = """
body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#1b1f24}
h1{font-size:1.5rem}h2{border-bottom:1px solid #ddd;padding-bottom:.2rem;margin-top:2rem}
.badge{display:inline-block;padding:.1rem .45rem;border-radius:.4rem;font-size:.8rem;background:#eef}
.supported{background:#d9f5df}.partially_supported{background:#eef7d4}.contradicted{background:#fbd5d5}
.uncertain{background:#fff1c9}.unverified,.unsupported{background:#eee}
table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #ddd;padding:.3rem;vertical-align:top}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem}
.card{border:1px solid #ddd;border-radius:.5rem;padding:.7rem}
small{color:#555}
"""


def to_html(result: SynapSIResult) -> str:
    """Self-contained HTML. Every model-generated string is escaped."""
    e = html.escape
    j, s, m = result.judgment, result.synthesis, result.metadata
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>SynapSI: {e(result.problem.question[:80])}</title><style>{_CSS}</style></head><body>",
        f"<h1>{e(result.problem.question)}</h1>",
        f"<p><span class='badge'>{e(m.strategy)}</span> <span class='badge'>{e(m.mode)}</span> ",
    ]
    if j:
        parts.append(
            f"<span class='badge'>verdict: {e(j.verdict.value)}</span> "
            f"<span class='badge'>answer: {e(j.answer or '—')}</span> "
            f"<span class='badge'>confidence {j.confidence:.2f}</span></p>"
        )
    if s:
        parts.append(f"<h2>Synthesis</h2><p>{e(s.summary)}</p><div class='grid'>")
        for title, findings in (
            ("Established", s.established),
            ("Probable", s.probable),
            ("Disputed", s.disputed),
            ("Unknown", s.unknown),
        ):
            items = "".join(
                f"<li>{e(f.statement)} <small>{e(', '.join(f.claim_ids + f.evidence_ids))}</small></li>"
                for f in findings
            )
            parts.append(
                f"<div class='card'><b>{title}</b><ul>{items or '<li><i>none</i></li>'}</ul></div>"
            )
        parts.append("</div>")
        if s.recommendations:
            parts.append(
                "<h3>Next steps</h3><ul>"
                + "".join(f"<li>{e(r)}</li>" for r in s.recommendations)
                + "</ul>"
            )
    if j:
        parts.append(
            f"<h2>Judgment</h2><p><b>{e(j.decision)}</b></p><p>{e(j.reasoning_summary)}</p>"
            f"<p><small>method {e(j.method)} · evidence {e(j.evidence_strength.value)}</small></p>"
        )
    parts.append("<h2>Perspectives</h2><div class='grid'>")
    for name, p in result.final_perspectives.items():
        parts.append(
            f"<div class='card'><b>{e(name)}</b> <small>{e(p.role)} · {e(p.model)}</small>"
            f"<p>answer: <b>{e(p.answer or '—')}</b> ({p.confidence:.2f})</p><p>{e(p.position)}</p></div>"
        )
    parts.append(
        "</div><h2>Claims</h2><table><tr><th>ID</th><th>Agent</th><th>Status</th><th>Statement</th><th>Evidence</th></tr>"
    )
    for c in result.claims:
        if c.withdrawn:
            continue
        parts.append(
            f"<tr><td>{e(c.id)}</td><td>{e(c.agent)}</td>"
            f"<td><span class='badge {e(c.status.value)}'>{e(c.status.value)}</span></td>"
            f"<td>{e(c.statement)}</td><td>{e(', '.join(c.evidence_ids))}</td></tr>"
        )
    parts.append("</table><h2>Evidence</h2><ul>")
    for ev in result.evidence:
        src = ev.provenance.source_id or ""
        link = (
            f" <a href='{e(src)}' rel='noopener noreferrer nofollow'>{e(src)}</a>"
            if src.startswith(("http://", "https://"))
            else f" <small>{e(src)}</small>"
        )
        parts.append(
            f"<li><b>{e(ev.id)}</b> <span class='badge'>{e(ev.provenance.source_kind.value)}</span> "
            f"{e(ev.content[:400])}{link}</li>"
        )
    parts.append("</ul>")
    if result.disagreements:
        parts.append("<h2>Disagreements</h2><ul>")
        for d in result.disagreements:
            parts.append(
                f"<li><b>{e(d.id)}</b> [{e(d.kind)}, {e(d.status)}] {e(d.topic)} <small>{e(d.evidence_balance)}</small></li>"
            )
        parts.append("</ul>")
    u = m.usage.total
    parts.append(
        f"<h2>Metadata</h2><p><small>run {e(result.run_id)} · {u.calls} model calls · "
        f"{u.total_tokens} tokens · cost {e(_fmt_cost(u.cost_usd))} · {m.latency_s:.2f}s</small></p>"
        "</body></html>"
    )
    return "".join(parts)
