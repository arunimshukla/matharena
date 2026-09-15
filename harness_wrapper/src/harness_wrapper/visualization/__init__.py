"""Dependency-free HTML rendering for harness traces."""

from __future__ import annotations

import json
from importlib import resources

from ..traces import Trace

_TRACE_DATA_MARKER = '<script id="initial-trace" type="application/json">{}</script>'


def render_trace_html(trace: Trace) -> str:
    """Return a self-contained interactive HTML visualization of ``trace``.

    The trace is embedded in the returned page; the viewer does not make any
    network requests. Content is escaped before insertion so trace text cannot
    terminate the JSON script element or inject markup into the page.
    """

    records: list[dict[str, object]] = []
    for event in trace.events:
        record: dict[str, object] = {
            "schema_version": 1,
            "session_id": trace.session_id,
            "event": event.to_dict(),
        }
        if trace.harness is not None:
            record["harness"] = trace.harness
        if trace.model is not None:
            record["model"] = trace.model
        if trace.metadata:
            record["trace_metadata"] = trace.metadata
        records.append(record)

    # JSON inside a script raw-text element must not contain a literal '<':
    # otherwise trace content such as '</script>' could close the element.
    payload = {
        "session": {
            "session_id": trace.session_id,
            "harness": trace.harness,
            "model": trace.model,
            "trace_metadata": trace.metadata,
        },
        "records": records,
    }
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    template = resources.files(__package__).joinpath("index.html").read_text(encoding="utf-8")
    replacement = f'<script id="initial-trace" type="application/json">{data}</script>'
    if _TRACE_DATA_MARKER not in template:
        raise RuntimeError("trace visualization template is missing its data marker")
    return template.replace(_TRACE_DATA_MARKER, replacement, 1)


__all__ = ["render_trace_html"]
