"""HTML for the run dashboard (see :mod:`synthdata.dashboard`).

One self-contained page: inline CSS and JavaScript, no network requests, so it
works opened straight from disk on an offline machine. Tables, the model
timeline and the HPO sparklines are drawn here; the resource charts and the
log filter are drawn in the browser from the JSON embedded in the page.
"""

import html
import json
from datetime import UTC, datetime

from synthdata.dashboard import _duration, base_model

STATE_LABEL = {
    "running": ("Running", "run"),
    "finished": ("Finished", "good"),
    "failed": ("Failed", "bad"),
    "stopped": ("Stopped", "bad"),
    "waiting": ("Waiting", "muted"),
}
STATUS_LABEL = {
    "running": ("running", "run"),
    "done": ("done", "good"),
    "failed": ("failed", "bad"),
    "stopped": ("stopped", "bad"),
    "pending": ("pending", "muted"),
    "queued": ("queued", "muted"),
    "paused": ("paused", "warn"),
    "not run": ("no trials", "muted"),
    "no pass": ("none passed, defaults used", "warn"),
}
#: Categorical colors for the steps on the model timeline (--k1 ...).
STEP_COLORS = 7
ICON = {"run": "▶", "good": "✓", "bad": "✕", "warn": "!", "muted": "•"}


def _e(text) -> str:
    return html.escape("" if text is None else str(text))


def _badge(status: str, table=STATUS_LABEL) -> str:
    label, kind = table.get(status, (status, "muted"))
    return f'<span class="badge {kind}"><i>{ICON[kind]}</i>{_e(label)}</span>'


def _time(iso: str | None, fmt: str = "datetime") -> str:
    """A ``<time>`` the page's script rewrites in the viewer's local time."""
    if not iso:
        return "-"
    return f'<time datetime="{iso}" data-fmt="{fmt}">{_e(iso.replace("T", " ").replace("Z", " UTC"))}</time>'


def _num(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


# ---------------------------------------------------------------- sections


def _header(d: dict, refresh: int) -> str:
    run = d["run"]
    label, kind = STATE_LABEL.get(run["state"], (run["state"], "muted"))
    title = f"Run dashboard{' · ' + _e(d['config_name']) if d['config_name'] else ''}"
    facts = [
        ("Experiment", _e(d["experiment"] or "-")),
        ("Host", _e(d["host"] or "-")),
        ("Started", _time(run["started"])),
        ("Elapsed", _e(_duration(run["elapsed_s"]))),
        ("Last log output", _time(run["last_output"], "ago")),
    ]
    note = f'<p class="note {kind}">{_e(run["note"])}</p>' if run["note"] else ""
    live = run["state"] in ("running", "waiting")
    toggle = (
        f'<button id="pause" type="button" aria-pressed="false">Pause refresh</button>'
        f'<span class="muted small" id="refresh-info">refreshes every {refresh} s</span>'
        if live
        else '<span class="muted small">run over: page no longer refreshes</span>'
    )
    dl = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    return f"""
<header class="top">
  <div class="title-row">
    <h1>{title}</h1>
    <span class="state {kind}"><i>{ICON[kind]}</i>{label}</span>
    <div class="spacer"></div>
    <div class="controls">{toggle}<span class="muted small">updated {_time(d["generated"], "time")}</span></div>
  </div>
  <dl class="facts">{dl}</dl>
  {note}
</header>"""


def _tiles(d: dict) -> str:
    stages = d["stages"]
    current = next((s for s in stages if s["status"] in ("running", "failed", "stopped")), None)
    done = sum(s["status"] == "done" for s in stages)
    if current:
        stage_value, stage_sub = (
            current["name"],
            f"stage {stages.index(current) + 1} of {len(stages)}",
        )
    elif stages and done == len(stages):
        stage_value, stage_sub = "all done", f"{len(stages)} stages"
    else:
        stage_value, stage_sub = "-", f"{done} of {len(stages)} stages done"

    step = next((s for s in reversed(d["steps"]) if s["running"]), None) or (
        d["steps"][-1] if d["steps"] else None
    )
    if step:
        models_value = f"{step['done']} / {step['total']}"
        models_sub = (
            f"{step['label']}: {step['running']} running"
            + (f", {step['failed']} failed" if step["failed"] else "")
            + (
                f", {step['with_metric_failures']} with failed metrics"
                if step.get("with_metric_failures")
                else ""
            )
        )
    else:
        models_value, models_sub = "-", "no per-model workers yet"

    hpo = d["hpo"]
    if hpo:
        trials_done = sum(s["complete"] + s["pruned"] for s in hpo)
        target = sum(s["target"] or 0 for s in hpo)
        hpo_value = f"{trials_done}" + (f" / {target}" if target else "")
        hpo_sub = (
            f"{sum(s['running'] for s in hpo)} running, "
            f"{sum(s.get('infeasible', 0) for s in hpo)} failed a screen, "
            f"{sum(s['failed'] for s in hpo)} errored"
        )
    else:
        hpo_value, hpo_sub = "-", "no HPO study yet"

    series = d["resources"]["series"]
    ram = [v for v in series["pipeline_ram_gib"] if v is not None]
    total = next((v for v in reversed(series["system_ram_total_gib"]) if v), None)
    if ram:
        ram_value = f"{max(ram):.1f} GiB"
        ram_sub = f"peak of this run{f', machine has {total:.0f} GiB' if total else ''}"
    else:
        ram_value, ram_sub = "-", "no resource samples yet"

    n_err, n_warn = d["n_errors"], d["n_warnings"]
    prob_kind = "bad" if n_err else ("warn" if n_warn else "good")
    tiles = [
        ("Stage", stage_value, stage_sub, ""),
        ("Models done", models_value, models_sub, ""),
        ("HPO trials", hpo_value, hpo_sub, ""),
        ("Peak RAM", ram_value, ram_sub, ""),
        (
            "Problems",
            f'<a href="#problems">{n_err} error{"s" * (n_err != 1)}</a>',
            f"{n_warn} warning{'s' * (n_warn != 1)}",
            prob_kind,
        ),
    ]
    return (
        '<section class="tiles">'
        + "".join(
            f'<div class="tile {kind}"><div class="tile-label">{label}</div>'
            f'<div class="tile-value">{value}</div><div class="tile-sub">{_e(sub)}</div></div>'
            for label, value, sub, kind in tiles
        )
        + "</section>"
    )


def _stage_strip(d: dict) -> str:
    items = []
    for s in d["stages"]:
        dur = _duration(s["duration_s"]) if s["duration_s"] is not None else ""
        peak = (
            f'<div class="stage-peak">{_e(s["peak"].split(": ", 1)[-1])}</div>' if s["peak"] else ""
        )
        items.append(
            f'<li class="stage {s["status"]}"><div class="stage-name">{_e(s["name"])}</div>'
            f'{_badge(s["status"])}<div class="stage-time">{_e(dur)}</div>{peak}</li>'
        )
    return (
        f'<section class="card"><h2>Stages</h2><ol class="stages">{"".join(items)}</ol></section>'
    )


def _at(iso: str | None, fallback: float) -> float:
    """Epoch seconds of ``iso``, or ``fallback`` (e.g. now, for a task still running)."""
    moment = _ts(iso)
    return fallback if moment is None else moment


def _ticks(t0: float, t1: float, max_ticks: int = 8) -> list[float]:
    """Round clock times between ``t0`` and ``t1`` (epoch seconds) for axis ticks."""
    span = max(t1 - t0, 1)
    steps = [60 * m for m in (1, 2, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1440)]
    step = next((s for s in steps if span / s <= max_ticks), steps[-1])
    first = (int(t0) // step + 1) * step
    return [t for t in range(first, int(t1), step) if 0.05 < (t - t0) / span < 0.93]


def _segments(tasks: list[dict], t1: float) -> list[dict]:
    """One segment per step for a model: its replicates' intervals merged where they overlap."""
    out = []
    for step in dict.fromkeys(t["step"] for t in tasks):
        mine = sorted(
            (t for t in tasks if t["step"] == step and t["start"]),
            key=lambda t: t["start"],
        )
        spans: list[list] = []
        for t in mine:
            start, end = _at(t["start"], t1), _at(t["end"], t1)
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
                spans[-1][2].append(t)
            else:
                spans.append([start, end, [t]])
        for start, end, group in spans:
            statuses = {t["status"] for t in group}
            status = next((s for s in ("failed", "stopped", "running") if s in statuses), "done")
            out.append({"step": step, "start": start, "end": end, "tasks": group, "status": status})
    return out


PHASE_LABEL = {"fit": "untuned fits", "HPO": "HPO search", "tuned fit": "tuned fits"}


def _phase_spans(tasks: list[dict], t1: float) -> list[list]:
    """[kind, start, end, count] per run of phases; a model's replicates merge into one span."""
    spans: list[list] = []
    for t in tasks:
        for phase in t.get("phases") or []:
            start, end = _at(phase["start"], t1), _at(phase["end"], t1)
            if spans and spans[-1][0] == phase["kind"]:
                spans[-1][2] = end
                spans[-1][3] += 1
            else:
                spans.append([phase["kind"], start, end, 1])
    return spans


def _phase_text(kind: str, start: float, end: float, n: int) -> str:
    return PHASE_LABEL[kind] + (f" ({n})" if kind != "HPO" else "") + f" {_duration(end - start)}"


def _phases(tasks: list[dict], t1: float, pct, model: str) -> list[str]:
    """Shading over a generation worker's bar: its HPO search and its tuned fits."""
    out = []
    for kind, start, end, n in _phase_spans(tasks, t1):
        if kind == "fit":
            continue  # the bar itself is untuned fitting
        tip = f"{model} · generation · {_phase_text(kind, start, end, n)}"
        cls = "hpo" if kind == "HPO" else "tuned"
        out.append(
            f'<div class="phase {cls}" style="left:{pct(start):.2f}%;'
            f'width:{max(pct(end) - pct(start), 0.3):.2f}%" title="{_e(tip)}"></div>'
        )
    return out


def _timeline(d: dict) -> str:
    tasks = d["tasks"]
    if not d["steps"] and not tasks:
        return (
            '<section class="card"><h2>Models</h2><p class="empty">No per-model workers have started. '
            "Each model's start and finish appear here when <code>compute.workers</code> is more than 1."
            "</p></section>"
        )
    t1 = _ts(d["run"]["ended"]) or _ts(d["generated"]) or datetime.now(UTC).timestamp()
    t0 = _ts(d["run"]["started"]) or min(
        (_at(t["start"], t1) for t in tasks if t["start"]), default=t1
    )
    span = max(t1 - t0, 1)

    def pct(t: float) -> float:
        return (t - t0) / span * 100

    steps = list(dict.fromkeys([s["label"] for s in d["steps"]] + [t["step"] for t in tasks]))
    color = {step: f"var(--k{i % STEP_COLORS + 1})" for i, step in enumerate(steps)}
    ticks = _ticks(t0, t1)
    grid = "".join(f'<i class="tick" style="left:{pct(t):.2f}%"></i>' for t in ticks)
    hpo = {s["model"]: s for s in d["hpo"]}

    by_model: dict[str, list] = {}
    for t in sorted(tasks, key=lambda t: (t["start"] or "~", t["order"])):
        by_model.setdefault(base_model(t["model"]), []).append(t)
    # A tuned model (ctgan_hpo) sits under its untuned one, whose generation
    # worker also trained it; families keep the order they started in.
    family = {m: i for i, m in enumerate(dict.fromkeys(m.removesuffix("_hpo") for m in by_model))}
    by_model = dict(
        sorted(by_model.items(), key=lambda kv: (family[kv[0].removesuffix("_hpo")], kv[0]))
    )
    rows = []
    for model, mine in by_model.items():
        segs = []
        for s in _segments(mine, t1):
            n = len(s["tasks"])
            # What the segment did: its phases (untuned fits, search, tuned
            # fits) for a generation worker, else its replicates.
            spans = _phase_spans(s["tasks"], t1)
            if spans:
                what = ", ".join(_phase_text(*span) for span in spans)
            elif n > 1:
                what = f"{n} replicates"
            else:
                what = "" if s["tasks"][0]["model"] == model else s["tasks"][0]["model"]
            parts = [model, s["step"], what, s["status"], _duration(s["end"] - s["start"])]
            tip = " · ".join(p for p in parts if p)
            n_failed = sum(t.get("metric_failures", 0) for t in s["tasks"])
            if n_failed:
                tip += f" · {n_failed} metric failure{'s' * (n_failed != 1)}"
            partial = " partial" if n_failed and s["status"] == "done" else ""
            segs.append(
                f'<div class="seg {s["status"]}{partial}" style="left:{pct(s["start"]):.2f}%;'
                f'width:{max(pct(s["end"]) - pct(s["start"]), 0.4):.2f}%;background:{color[s["step"]]}" '
                f'title="{_e(tip)}"></div>'
            )
        segs += _phases(mine, t1, pct, model)
        statuses = {t["status"] for t in mine}
        status = next((s for s in ("failed", "running", "stopped") if s in statuses), "done")
        starts = [_at(t["start"], t1) for t in mine if t["start"]]
        ends = [_at(t["end"], t1) for t in mine if t["start"]]
        busy = sum(t["duration_s"] or 0 for t in mine)
        wall = _duration(max(ends) - min(starts)) if starts else "-"
        dur = f'<span class="g-dur" title="first start to last finish; {_e(_duration(busy))} of worker time">{_e(wall)}</span>'
        study = hpo.get(model)
        extra = ""
        if study and study["target"]:
            extra = f'<span class="muted small">HPO {study["complete"] + study["pruned"]}/{study["target"]}</span>'
        errors = [f"{t['step']}: {t['error']}" for t in mine if t.get("error")]
        err = f'<div class="row-error">{_e(errors[-1])}</div>' if errors else ""
        metric_steps = {}
        for t in mine:
            if t.get("metric_failures"):
                metric_steps[t["step"]] = metric_steps.get(t["step"], 0) + t["metric_failures"]
        if metric_steps:
            listed = ", ".join(f"{step} {count}" for step, count in metric_steps.items())
            err += f'<div class="row-warn">failed metrics: {_e(listed)}</div>'
        rows.append(
            f'<div class="g-row"><div class="g-name"><span class="model">{_e(model)}</span>'
            f"{_badge(status)}{dur}{extra}{err}</div>"
            f'<div class="g-track">{grid}{"".join(segs)}</div></div>'
        )

    stage_bands = "".join(
        f'<div class="band" style="left:{pct(_at(s["start"], t1)):.2f}%;'
        f'width:{max(pct(_at(s["end"], t1)) - pct(_at(s["start"], t1)), 0.4):.2f}%" '
        f'title="{_e(s["name"])} · {_e(_duration(s["duration_s"]))}"><span>{_e(s["name"])}</span></div>'
        for s in d["stages"]
        if s["start"]
    )
    stage_row = (
        f'<div class="g-row g-stages"><div class="g-name muted small">pipeline stage</div>'
        f'<div class="g-track">{grid}{stage_bands}</div></div>'
    )
    tick_labels = "".join(
        f'<span style="left:{pct(t):.2f}%">{_time(datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "time")}</span>'
        for t in ticks
    )
    axis = (
        f'<div class="g-row g-axis"><div class="g-name"></div><div class="g-track">'
        f'<span style="left:0">{_time(d["run"]["started"], "time")}</span>{tick_labels}'
        f'<span style="right:0">{_time(d["run"]["ended"] or d["generated"], "time")}</span></div></div>'
    )

    summary = []
    for step in d["steps"]:
        queued = step["total"] - sum(t["step"] == step["label"] for t in tasks)
        text = f"{step['done']} of {step['total']} done"
        if step["failed"]:
            text += f", {step['failed']} failed"
        if step["running"]:
            text += f", {step['running']} running"
        if step.get("with_metric_failures"):
            text += f", {step['with_metric_failures']} with failed metrics"
        if queued > 0:
            text += f", {queued} waiting"
        if step.get("workers"):
            text += f" · {step['workers']} at a time, {step['cores']} CPU threads each"
        summary.append(
            f'<li><i class="sw" style="background:{color[step["label"]]}"></i>'
            f'<b>{_e(step["label"])}</b> <span class="muted">{_e(text)}</span></li>'
        )
    legend = (
        f'<ul class="steps-legend">{"".join(summary)}</ul>'
        '<p class="muted small">One bar per model; each colored segment is one step, from its '
        "first replicate's start to its last one's finish. Hover a segment for details. "
        "Striped: running; red outline: failed; amber outline: finished, but some metrics failed "
        "(listed under Problems). On a generation bar, the plain part is the untuned fits, "
        "the light part the HPO search and the dark part the tuned fits.</p>"
    )
    return (
        f'<section class="card"><h2>Models</h2>{legend}'
        f'<div class="gantt">{stage_row}{"".join(rows)}{axis}</div></section>'
    )


def _screen_text(screen: dict) -> str:
    """One failed screen for a tooltip: value against its limit when recorded."""
    if screen.get("value") is not None and screen.get("limit") is not None:
        bound = "at most" if screen.get("bound") == "max" else "at least"
        return f"{screen['name']}: {screen['value']:.3g} (needs {bound} {screen['limit']:.3g})"
    return f"{screen['name']}: off by {screen['violation']:.3g}"


def _trial_tip(c: list, objective: str | None = None) -> str:
    lines = [f"trial {c[0]}: {objective + ' ' if objective else ''}{c[1]:.4g}"]
    failed = c[4] if len(c) > 4 else []
    if failed:
        lines += ["failed " + _screen_text(f) for f in failed]
    elif not c[3]:
        lines.append("failed a sanity screen")
    else:
        lines.append("passed every sanity screen")
    return "\n".join(lines)


def _sparkline(curve: list, maximize: bool, objective: str | None = None) -> str:
    if len(curve) < 2:
        return '<span class="muted small">-</span>'
    w, h, pad = 220, 56, 4
    xs = [c[0] for c in curve]
    values = [c[1] for c in curve if c[1] is not None]
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1e-9
    x0, x1 = min(xs), max(xs) or 1

    def px(x):
        return pad + (x - x0) / max(x1 - x0, 1) * (w - 2 * pad)

    def py(y):
        return h - pad - (y - lo) / (hi - lo) * (h - 2 * pad)

    # Each dot's <title> is the browser's own hover tooltip: no script needed.
    dots = "".join(
        f'<circle cx="{px(c[0]):.1f}" cy="{py(c[1]):.1f}" r="2.6" class="{"dot" if c[3] else "dot off"}">'
        f"<title>{_e(_trial_tip(c, objective))}</title></circle>"
        for c in curve
    )
    best = [c for c in curve if c[2] is not None]
    path = ""
    for i, c in enumerate(best):
        x, y = px(c[0]), py(c[2])
        path += f"M{x:.1f},{y:.1f}" if i == 0 else f"H{x:.1f}V{y:.1f}"
    if best:
        path += f"H{px(x1):.1f}"
    better = "higher" if maximize else "lower"
    label = f"{objective or 'objective'} by trial, best so far ({better} is better)"
    # The objective and its range sit next to the plot, so the y axis reads without the table.
    axis = (
        f'<div class="spark-axis small"><b>{_e(objective or "objective")}</b> '
        f'<span class="muted">{_num(lo)} to {_num(hi)}, {better} is better</span></div>'
    )
    return (
        f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(label)}">'
        f'<title>{_e(label)}</title><path d="{path}" class="best"/>{dots}</svg>{axis}'
    )


def _hpo(d: dict) -> str:
    if d["hpo_error"]:
        body = f'<p class="empty">{_e(d["hpo_error"])}</p>'
    elif not d["hpo"]:
        body = '<p class="empty">No HPO study has started for this experiment.</p>'
    else:
        rows = []
        for s in d["hpo"]:
            n = s["complete"] + s["pruned"]
            target = s["target"]
            status = s.get("status") or "done"
            pct = min(n / target * 100, 100) if target else 0
            prog = f'<div class="prog"><div style="width:{pct:.1f}%"></div></div>' if target else ""
            counts = f"{n}{f' / {target}' if target else ''}"
            detail = (
                f"{s['complete'] - s.get('infeasible', 0)} passed the screens, "
                f"{s.get('infeasible', 0)} failed a screen, {s['failed']} errored"
            )
            fails = s.get("screen_fails") or {}
            screens = (
                "<br>".join(f"{_e(name)} <b>{count}</b>" for name, count in fails.items())
                if fails
                else '<span class="muted small">none failed</span>'
            )
            stop = s.get("stop") if status == "stopped" else None
            why = f'<div class="muted small">{_e(stop)}</div>' if stop else ""
            err = (
                f'<details><summary class="small">last failure</summary><pre>{_e(s["last_error"])}</pre></details>'
                if s["last_error"]
                else ""
            )
            best = (
                '<span class="muted small">none passed</span>'
                if s["best"] is None and s["complete"] and s.get("infeasible", 0) >= s["complete"]
                else _num(s["best"])
            )
            eta = _duration(s["eta_s"]) if s["eta_s"] is not None and status == "running" else "-"
            rows.append(
                f'<tr><td class="model">{_e(s["model"])}</td>'
                f"<td>{_badge(status)}{why}</td>"
                f'<td><div class="tnum">{counts}</div>{prog}<div class="muted small">{detail}</div></td>'
                f'<td class="small">{screens}</td>'
                f'<td class="num">{best}</td><td>{_sparkline(s["curve"], s["maximize"], s.get("objective"))}</td>'
                f'<td class="num">{_e(_duration(s["median_trial_s"]))}</td><td class="num">{_e(eta)}</td>'
                f"<td>{err}</td></tr>"
            )
        body = (
            '<div class="scroll"><table><thead><tr><th>Model</th><th>Status</th>'
            "<th>Trials</th><th>Failed screens (trials)</th>"
            '<th class="num">Best</th><th>Trials and best so far</th><th class="num">Median trial</th>'
            '<th class="num">Time left</th><th></th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
            '<p class="legend-row small">'
            '<span class="legend"><svg class="spark" width="10" height="10"><circle cx="5" cy="5" r="2.6" class="dot"/></svg>'
            "trial score</span>"
            '<span class="legend"><svg class="spark" width="10" height="10"><circle cx="5" cy="5" r="2.6" class="dot off"/></svg>'
            "trial that failed a sanity screen (never chosen)</span>"
            '<span class="legend"><i style="border-color:var(--s1)"></i>best passing score so far</span>'
            '<span class="muted">x: trial number, y: the HPO objective; hover a dot for its '
            "screens</span></p>"
            '<p class="muted small">Best counts only trials that pass the sanity screens. '
            "Time left is remaining trials times the median of the last ten, capped by the study's timeout. "
            "When no trial passes the screens, the model is fitted with synthcity's defaults.</p>"
        )
    return f'<section class="card"><h2>Hyperparameter search</h2>{body}</section>'


def _resource_rows(tasks: list[dict], by: str) -> str:
    """``tbody`` rows of use by task, under a header row per step (``by="step"``) or model."""

    def split(task: str) -> tuple[str, str]:
        step, _, model = task.partition(": ")
        return (step, model) if model else ("pipeline", task)

    # Replicates of a model in a step fold into one row: peaks are the
    # largest, "now" values the sum over those running.
    merged: dict[tuple[str, str], dict] = {}
    for t in tasks:
        step, model = split(t["task"])
        m = merged.setdefault(
            (step, base_model(model)),
            {"n": 0, "ram_peak": 0.0, "cpu_peak": 0.0, "vram_peak": 0.0, "active": False},
        )
        m["n"] += 1
        for k in ("ram_peak", "cpu_peak", "vram_peak"):
            m[k] = max(m[k], t[k] or 0)
        if t.get("active"):
            m["active"] = True
            for k in ("ram_now", "cpu_now"):
                m[k] = (m.get(k) or 0) + (t.get(k) or 0)
    groups: dict[str, list] = {}
    for (step, model), t in merged.items():
        key, label = (step, model) if by == "step" else (model, step)
        if t["n"] > 1:
            label += f" ×{t['n']}"
        groups.setdefault(key, []).append((label, t))
    order = sorted(groups, key=lambda k: (k in ("pipeline", "main"), k))
    out = []
    for key in order:
        members = sorted(groups[key], key=lambda m: m[0])
        active = sum(bool(t.get("active")) for _, t in members)
        peak = max(t["ram_peak"] for _, t in members)
        note = f"{len(members)} row{'s' * (len(members) != 1)}" + (
            f", {active} running" if active else ""
        )
        out.append(
            f'<tr class="group"><th colspan="7">{_e(key)} <span class="muted small">{note}, '
            f"RAM peak {peak:.1f} GiB</span></th></tr>"
        )
        out += [
            f'<tr><td class="indent">{_e(label)}</td><td>{_badge("running") if t.get("active") else ""}</td>'
            f'<td class="num">{_num(t.get("ram_now"), 1)}</td><td class="num">{_num(t["ram_peak"], 1)}</td>'
            f'<td class="num">{_num(t.get("cpu_now"), 1)}</td><td class="num">{_num(t["cpu_peak"], 1)}</td>'
            f'<td class="num">{_num((t["vram_peak"] or 0) / 1024, 1)}</td></tr>'
            for label, t in members
        ]
    return "".join(out)


def _resource_table(tasks: list[dict]) -> str:
    head = (
        '<thead><tr><th>Task</th><th></th><th class="num">RAM now (GiB)</th><th class="num">RAM peak</th>'
        '<th class="num">CPU now (cores)</th><th class="num">CPU peak</th>'
        '<th class="num">GPU memory peak (GiB)</th></tr></thead>'
    )
    return (
        '<h3>By task <span class="filters inline"><label><input type="radio" name="res-by" value="step" checked> '
        'group by step</label><label><input type="radio" name="res-by" value="model"> group by model</label></span></h3>'
        f'<div class="scroll"><table class="res" data-by="step">{head}<tbody>{_resource_rows(tasks, "step")}</tbody></table>'
        f'<table class="res" data-by="model" hidden>{head}<tbody>{_resource_rows(tasks, "model")}</tbody></table></div>'
    )


def _resources(d: dict) -> str:
    res = d["resources"]
    if not res["n_rows"]:
        return (
            '<section class="card"><h2>Resources</h2><p class="empty">No resource samples yet '
            "(the first row lands one monitor interval after the start).</p></section>"
        )
    table = _resource_table(res["tasks"]) if res["tasks"] else ""
    return (
        '<section class="card"><h2>Resources</h2><div class="charts">'
        '<figure id="c-ram"></figure><figure id="c-cpu"></figure>'
        '<figure id="c-gpu"></figure><figure id="c-vram"></figure></div>'
        f"{table}</section>"
    )


def _problems(d: dict) -> str:
    if not d["problems"]:
        body = '<p class="empty good-text">✓ No errors or warnings in the log.</p>'
    else:
        items = []
        for p in d["problems"]:
            kind = "bad" if p["level"] == "error" else "warn"
            count = f'<span class="count">×{p["count"]}</span>' if p["count"] > 1 else ""
            ctx = f"<pre>{_e(chr(10).join(p['context']))}</pre>" if p["context"] else ""
            source = f'<span class="source">{_e(p["source"])}</span>' if p["source"] else ""
            head = (
                f'<span class="badge {kind}"><i>{ICON[kind]}</i>{p["level"]}</span>{source}'
                f'<span class="ptext">{_e(p["text"])}</span>{count}<span class="muted small">{_time(p["last"], "time")}</span>'
            )
            items.append(
                f"<li><details><summary>{head}</summary>{ctx}</details></li>"
                if ctx
                else f'<li><div class="psum">{head}</div></li>'
            )
        body = f'<ul class="problems">{"".join(items)}</ul>'
    return f'<section class="card" id="problems"><h2>Problems</h2>{body}</section>'


def _ci_bar(x: dict, lo: float, hi: float, best: float | None) -> str:
    """The overall score and its 95% interval on a shared scale, as a small SVG."""
    w, h, pad = 260, 18, 4

    def px(v: float) -> float:
        return pad + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - 2 * pad)

    marks = ""
    if best is not None:
        marks += f'<line x1="{px(best):.1f}" x2="{px(best):.1f}" y1="0" y2="{h}" class="ref"/>'
    if x["low"] is not None and x["high"] is not None:
        marks += (
            f'<line x1="{px(x["low"]):.1f}" x2="{px(x["high"]):.1f}" y1="9" y2="9" class="ci"/>'
            f'<line x1="{px(x["low"]):.1f}" x2="{px(x["low"]):.1f}" y1="5" y2="13" class="ci"/>'
            f'<line x1="{px(x["high"]):.1f}" x2="{px(x["high"]):.1f}" y1="5" y2="13" class="ci"/>'
        )
    if x["overall"] is not None:
        kind = "base" if x["baseline"] else "mean"
        marks += f'<circle cx="{px(x["overall"]):.1f}" cy="9" r="3.5" class="{kind}"/>'
    label = f"overall {_num(x['overall'])}, 95% CI {_num(x['low'])} to {_num(x['high'])}"
    return (
        f'<svg class="cibar" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{label}"><title>{label}</title>{marks}</svg>'
    )


def _ranking(ranking: list[dict]) -> str:
    values = [v for x in ranking for v in (x["overall"], x["low"], x["high"]) if v is not None]
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    # Rank scores live in [0, 1]; a t-interval from three replicates can
    # reach past that, and stretching the axis for it squeezes every bar.
    lo, hi = max(lo, 0.0), min(hi, 1.0)
    margin = max((hi - lo) * 0.05, 1e-6)
    lo, hi = lo - margin, hi + margin
    best_row = next((x for x in ranking if not x["baseline"]), None)
    best = best_row["overall"] if best_row else None
    dims = [
        d for d in ("utility", "privacy", "fairness") if any(x.get(d) is not None for x in ranking)
    ]
    rows = []
    for x in ranking:
        if x["baseline"]:
            vs = '<span class="muted small">baseline</span>'
        elif x is best_row:
            vs = '<span class="pill good">best</span>'
        elif x["tied"] is None:
            vs = '<span class="muted small" title="needs 2+ replicates on both sides">n/a</span>'
        else:
            vs = (
                '<span class="pill good">tied</span>'
                if x["tied"]
                else '<span class="muted">below</span>'
            )
        untuned = (
            ' <span class="pill warn" title="No HPO trial passed the screens: fitted with the '
            'untuned settings, so it repeats the untuned model.">untuned</span>'
            if x.get("untuned")
            else ""
        )
        rows.append(
            f'<tr class="{"baseline" if x["baseline"] else ""}"><td class="num">{_e(x["rank"])}</td>'
            f'<td class="model">{_e(x["model"])}{untuned}</td>'
            f'<td class="num">{_num(x["overall"])}</td><td>{_ci_bar(x, lo, hi, best)}</td>'
            f'<td class="num muted small">{_num(x["low"])} to {_num(x["high"])}</td><td>{vs}</td>'
            + "".join(f'<td class="num">{_num(x.get(dim))}</td>' for dim in dims)
            + f'<td class="num">{_e(x["n"])}</td></tr>'
        )
    head = "".join(f'<th class="num">{dim.title()}</th>' for dim in dims)
    return (
        '<h3>Ranking</h3><div class="scroll"><table><thead><tr><th class="num">Rank</th><th>Model</th>'
        f'<th class="num">Overall</th><th>95% CI <span class="muted small">{_num(lo, 2)} to {_num(hi, 2)}</span></th>'
        f'<th></th><th>vs best</th>{head}<th class="num">Replicates</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="muted small">Scores are rank scores, higher is better; each is the mean over seed '
        "replicates. The bar is the 95% interval of the overall score and the dashed line the best model's "
        "mean. <b>tied</b>: a one-sided Welch t-test at the 5% level cannot place the model below the best "
        "one; <b>below</b>: it can; <b>n/a</b>: one side has a single replicate. Baselines are reference rows, "
        "never ranked against the models. <b>untuned</b>: no HPO trial passed the screens, so the "
        "tuned model was fitted with the untuned settings and repeats the untuned one.</p>"
    )


def _results(d: dict) -> str:
    r = d["results"]
    if not r:
        return (
            '<section class="card"><h2>Results</h2><p class="empty">Evaluation has not written its '
            "ranking yet.</p></section>"
        )
    parts = []
    if "ranking" in r:
        parts.append(_ranking(r["ranking"]))
    if "checks" in r:
        c = r["checks"]
        kind = "warn" if c["warn"] else "good"
        line = f"<p>{_badge('done' if not c['warn'] else 'paused')} {c['total'] - c['warn']} of {c['total']} output checks pass.</p>"
        warned = "".join(
            f"<li><b>{_e(w['check'])}</b> · {_e(w['model'])}: {_e(w['value'])} (limit {_e(w['limit'])}) {_e(w['detail'])}</li>"
            for w in c["warned"]
        )
        parts.append(f'<h3>Checks</h3><div class="{kind}-box">{line}<ul>{warned}</ul></div>')
    if "report" in r:
        href = _e(r.get("report_rel", r["report"]))
        parts.append(
            f'<p>Full results: <a href="{href}">report.md</a> <span class="muted small">{_e(r["report"])}</span></p>'
        )
    return f'<section class="card"><h2>Results</h2>{"".join(parts)}</section>'


def _log(d: dict) -> str:
    options = "".join(f'<option value="{_e(t)}">{_e(t)}</option>' for t in d["log_tasks"])
    dropped = d["log_progress_dropped"]
    note = f", {dropped} progress-bar lines hidden" if dropped else ""
    return f"""
<section class="card"><h2>Log <span class="muted small">{_e(d["log"])}</span></h2>
  <div class="filters">
    <label>Model <select id="log-task"><option value="">all</option>{options}</select></label>
    <label>Contains <input id="log-text" type="search" placeholder="text"></label>
    <label><input id="log-problems" type="checkbox"> errors and warnings only</label>
    <span class="muted small" id="log-count">{len(d["log_lines"])} lines{_e(note)}</span>
  </div>
  <pre id="log" class="tail"></pre>
</section>"""


# ---------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--good-ink:#006300;--warn:#fab219;
--warn-ink:#8a5a00;--bad:#d03b3b;--run:#2a78d6;--track:#f0efec;--shadow:0 1px 2px rgba(0,0,0,.06);
--k1:#2a78d6;--k2:#eb6834;--k3:#1a9e8f;--k4:#8a5cd1;--k5:#c4458f;--k6:#8f9a1b;--k7:#a0703c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--good-ink:#0ca30c;
--warn-ink:#fab219;--run:#3987e5;--track:#262624;--shadow:none}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--good-ink:#0ca30c;--warn-ink:#fab219;--run:#3987e5;--track:#262624;--shadow:none}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:20px;margin:0;font-weight:650}h2{font-size:15px;margin:0 0 12px;font-weight:650}
h3{font-size:13px;margin:16px 0 8px;font-weight:600;color:var(--ink2)}
a{color:var(--s1)}code{font-size:12px}
.muted{color:var(--muted)}.small{font-size:12px;font-weight:400}.spacer{flex:1}
.tnum{font-variant-numeric:tabular-nums;font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.top{margin-bottom:16px}.title-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
button{font:inherit;font-size:12px;padding:4px 10px;border-radius:6px;border:1px solid var(--axis);background:var(--surface);color:var(--ink);cursor:pointer}
.state{display:inline-flex;align-items:center;gap:6px;font-weight:600;padding:3px 10px;border-radius:999px;border:1px solid currentColor}
.state i,.badge i{font-style:normal;font-size:10px}
.state.run{color:var(--run)}.state.good{color:var(--good-ink)}.state.bad{color:var(--bad)}.state.muted{color:var(--muted)}
.state.run i{animation:pulse 1.6s ease-in-out infinite}@keyframes pulse{50%{opacity:.25}}
.facts{display:flex;flex-wrap:wrap;gap:6px 28px;margin:10px 0 0}.facts div{display:flex;gap:6px}
.facts dt{color:var(--muted)}.facts dd{margin:0;font-variant-numeric:tabular-nums}
.note{margin:10px 0 0;padding:8px 12px;border-radius:8px;border-left:3px solid var(--warn);background:var(--surface)}
.note.bad{border-color:var(--bad)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:12px 14px;box-shadow:var(--shadow)}
.tile.bad{border-left:3px solid var(--bad)}.tile.warn{border-left:3px solid var(--warn)}.tile.good{border-left:3px solid var(--good)}
.tile-label{font-size:12px;color:var(--muted)}.tile-value{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;margin:2px 0}
.tile-value a{color:inherit;text-decoration:none}.tile-sub{font-size:12px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:16px;margin-bottom:16px;box-shadow:var(--shadow);min-width:0}
.grid2{display:grid;grid-template-columns:1fr;gap:0 16px}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;padding:1px 7px;border-radius:999px;white-space:nowrap;border:1px solid}
.badge.run{color:var(--run)}.badge.good{color:var(--good-ink)}.badge.bad{color:var(--bad)}.badge.warn{color:var(--warn-ink)}.badge.muted{color:var(--muted)}
.stages{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.stage{border:1px solid var(--grid);border-radius:8px;padding:10px 12px;position:relative;border-top:3px solid var(--grid)}
.stage.running{border-top-color:var(--run)}.stage.done{border-top-color:var(--good)}.stage.failed,.stage.stopped{border-top-color:var(--bad)}
.stage-name{font-weight:600;margin-bottom:4px}.stage-time{font-variant-numeric:tabular-nums;margin-top:4px}
.stage-peak{font-size:11px;color:var(--muted);margin-top:2px}
.g-row{display:grid;grid-template-columns:minmax(280px,30%) 1fr;gap:12px;align-items:center;padding:3px 0;border-bottom:1px solid var(--grid)}
.g-name{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;min-width:0}.model{font-weight:600;overflow-wrap:anywhere}
.g-dur{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:12px}
.g-track{position:relative;height:18px;background:var(--track);border-radius:4px}
.g-track .tick{position:absolute;top:0;bottom:0;border-left:1px solid var(--axis)}
.seg{position:absolute;top:3px;bottom:3px;border-radius:3px;min-width:2px}
.seg.running{background-image:repeating-linear-gradient(-45deg,transparent 0 5px,rgba(255,255,255,.4) 5px 10px)!important;background-size:28px 100%;animation:slide 1.2s linear infinite}
.seg.failed,.seg.stopped{outline:2px solid var(--bad);outline-offset:1px}
.seg.partial{outline:2px solid var(--warn);outline-offset:1px}
.seg.stopped{opacity:.55}
.phase{position:absolute;top:3px;bottom:3px;pointer-events:auto}
.phase.hpo{background:rgba(255,255,255,.55)}
.phase.tuned{background:rgba(0,0,0,.35)}
@keyframes slide{to{background-position:28px 0}}
.g-stages .g-track{background:none}
.band{position:absolute;top:1px;bottom:1px;border:1px solid var(--axis);border-radius:3px;background:var(--page);font-size:11px;color:var(--ink2);overflow:hidden;white-space:nowrap}
.band span{padding:0 4px;line-height:14px}
.g-axis{border:0}.g-axis .g-track{background:none;height:16px;font-size:11px;color:var(--muted)}.g-axis span{position:absolute}
.g-axis span[style^="left:"]:not([style="left:0"]){transform:translateX(-50%)}
.steps-legend{list-style:none;margin:0 0 6px;padding:0;display:flex;flex-wrap:wrap;gap:4px 18px;font-size:13px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend-row{display:flex;flex-wrap:wrap;gap:4px 18px;align-items:center;margin:8px 0 2px}
.cibar .ci{stroke:var(--ink2);stroke-width:1.5}.cibar .mean{fill:var(--s1)}.cibar .base{fill:none;stroke:var(--muted);stroke-width:1.5}
.cibar .ref{stroke:var(--s1);stroke-dasharray:3 3;opacity:.6}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:0 7px;border-radius:999px;border:1px solid}.pill.good{color:var(--good-ink)}.pill.warn{color:var(--warn-ink);font-weight:500;margin-left:6px}
tr.group th{background:var(--track);color:var(--ink);font-size:13px}td.indent{padding-left:20px}
.filters.inline{display:inline-flex;margin:0 0 0 12px;font-weight:400;color:var(--ink2)}
.row-error{flex-basis:100%;font-size:12px;color:var(--bad);overflow-wrap:anywhere}
.row-warn{flex-basis:100%;font-size:12px;color:var(--warn-ink);overflow-wrap:anywhere}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:middle}
th{font-size:12px;font-weight:600;color:var(--muted)}tr.baseline td{color:var(--ink2)}td.model{white-space:nowrap}
.prog{height:4px;background:var(--track);border-radius:2px;margin:3px 0;min-width:90px}.prog div{height:100%;background:var(--s1);border-radius:2px}
.spark .dot{fill:var(--muted);opacity:.55}.spark .dot.off{fill:none;stroke:var(--muted);stroke-width:.8;pointer-events:all}.spark-axis{margin-top:2px;white-space:nowrap}.spark .best{fill:none;stroke:var(--s1);stroke-width:2}
details summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:6px 0 0}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}
figure{margin:0;position:relative;min-width:0}figcaption{font-size:13px;font-weight:600;margin-bottom:4px;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center}
.legend{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:400;color:var(--ink2)}
.legend i{display:inline-block;width:14px;height:0;border-top:2px solid}
.chart svg{display:block;width:100%;height:190px;overflow:visible}
.chart .gridline{stroke:var(--grid);stroke-width:1}.chart .tick{fill:var(--muted);font-size:11px}
.chart .stageline{stroke:var(--axis);stroke-dasharray:3 3}.chart .stagelabel{fill:var(--muted);font-size:10px}
.chart .hover{stroke:var(--muted);stroke-width:1}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);white-space:nowrap;z-index:2}
.tip b{font-weight:600}.tip .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px}
.problems{list-style:none;margin:0;padding:0}.problems li{border-bottom:1px solid var(--grid);padding:6px 0}
.problems summary,.psum{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:baseline}
.source{font-size:12px;color:var(--ink2);font-weight:600}.ptext{overflow-wrap:anywhere;flex:1;min-width:200px}
.count{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.empty{color:var(--muted);margin:0}.good-text{color:var(--good-ink)}
.warn-box,.good-box{border-left:3px solid var(--warn);padding:2px 12px}.good-box{border-color:var(--good)}
.filters{display:flex;flex-wrap:wrap;gap:8px 16px;margin-bottom:8px;font-size:13px;align-items:center}
select,input[type=search]{font:inherit;font-size:13px;padding:3px 6px;border:1px solid var(--axis);border-radius:6px;background:var(--surface);color:var(--ink)}
.tail{max-height:480px;overflow:auto;background:var(--page);border:1px solid var(--grid);border-radius:8px;padding:10px;margin:0}
.tail .e{color:var(--bad)}.tail .w{color:var(--warn-ink)}
@media (max-width:640px){.g-row{grid-template-columns:1fr}.tile-value{font-size:20px}}
"""

JS = r"""
(function(){
const D=JSON.parse(document.getElementById('data').textContent);
const store={get(k){try{return sessionStorage.getItem(k)}catch(e){return null}},set(k,v){try{sessionStorage.setItem(k,v)}catch(e){}}};
const pad=n=>String(n).padStart(2,'0');
function fmt(d,f){const t=pad(d.getHours())+':'+pad(d.getMinutes());
  if(f==='time')return t;if(f==='hm')return t;if(f==='hms')return t+':'+pad(d.getSeconds());
  return d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+' '+t;}
function ago(d){const s=Math.max(0,(Date.now()-d)/1000);if(s<90)return Math.round(s)+' s ago';if(s<5400)return Math.round(s/60)+' min ago';return Math.round(s/3600)+' h ago';}
document.querySelectorAll('time[datetime]').forEach(el=>{const d=new Date(el.getAttribute('datetime'));if(isNaN(d))return;
  el.title=el.getAttribute('datetime').replace('T',' ').replace('Z',' UTC');
  el.textContent=el.dataset.fmt==='ago'?ago(d):fmt(d,el.dataset.fmt);});

// ---- refresh, keeping scroll position and filters
const live=D.run.state==='running'||D.run.state==='waiting';
let paused=store.get('dash-paused')==='1';
const btn=document.getElementById('pause');
function paint(){if(btn){btn.textContent=paused?'Resume refresh':'Pause refresh';btn.setAttribute('aria-pressed',paused);}}
if(btn){btn.onclick=()=>{paused=!paused;store.set('dash-paused',paused?'1':'0');paint();};}
paint();
const sy=store.get('dash-scroll');if(sy)window.scrollTo(0,+sy);
window.addEventListener('scroll',()=>store.set('dash-scroll',String(window.scrollY)),{passive:true});
if(live&&D.refresh>0)setInterval(()=>{if(!paused)location.reload();},D.refresh*1000);

// ---- resource table grouping
const resBy=document.querySelectorAll('input[name=res-by]');
function showRes(by){document.querySelectorAll('table.res').forEach(t=>t.hidden=t.dataset.by!==by);store.set('dash-res-by',by);}
resBy.forEach(r=>{r.addEventListener('change',()=>showRes(r.value));if(r.value===store.get('dash-res-by')){r.checked=true;showRes(r.value);}});

// ---- log
const tail=document.getElementById('log'),selTask=document.getElementById('log-task'),txt=document.getElementById('log-text'),onlyP=document.getElementById('log-problems'),cnt=document.getElementById('log-count'),cnt0=cnt.textContent;
const isErr=l=>/ (ERROR|CRITICAL): |Traceback|failed/.test(l),isWarn=l=>/ WARNING: |Warning: /.test(l);
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderTail(){const task=selTask.value,q=txt.value.toLowerCase(),p=onlyP.checked;
  store.set('dash-log',JSON.stringify([task,txt.value,p]));
  const lines=D.log.filter(l=>(!task||l.startsWith('['+task+']'))&&(!q||l.toLowerCase().includes(q))&&(!p||isErr(l)||isWarn(l)));
  tail.innerHTML=lines.map(l=>isErr(l)?'<span class="e">'+esc(l)+'</span>':isWarn(l)?'<span class="w">'+esc(l)+'</span>':esc(l)).join('\n')||'<span class="muted">no matching lines</span>';
  cnt.textContent=(task||q||p)?lines.length+' of '+cnt0:cnt0;
  tail.scrollTop=tail.scrollHeight;}
try{const s=JSON.parse(store.get('dash-log')||'null');if(s){selTask.value=s[0];txt.value=s[1];onlyP.checked=s[2];}}catch(e){}
[selTask,txt,onlyP].forEach(el=>el.addEventListener('input',renderTail));renderTail();

// ---- resource charts
const R=D.resources.series;if(!R||!R.t.length)return;
const times=R.t.map(s=>new Date(s).getTime());
function niceMax(v){if(!(v>0))return 1;const raw=v/4,p=Math.pow(10,Math.floor(Math.log10(raw)));for(const m of [1,1.5,2,2.5,3,4,5,6,8,10])if(m*p>=raw)return 4*m*p;return 40*p;}
const css=getComputedStyle(document.documentElement);
function chart(id,title,unit,series,opt){const fig=document.getElementById(id);if(!fig)return;
  series=series.filter(s=>s.values.some(v=>v!=null));
  if(!series.length){fig.innerHTML='<figcaption>'+title+'</figcaption><p class="empty">No data (no GPU, or not sampled).</p>';return;}
  const legend=series.map(s=>'<span class="legend"><i style="border-color:'+s.color+(s.dash?';border-top-style:dashed':'')+'"></i>'+s.name+'</span>').join('');
  fig.innerHTML='<figcaption>'+title+legend+'</figcaption><div class="chart"></div><div class="tip" hidden></div>';
  const box=fig.querySelector('.chart'),tip=fig.querySelector('.tip');
  const W=Math.max(box.clientWidth,280),H=190,L=44,Rr=12,T=14,B=22;
  let max=opt&&opt.max||niceMax(Math.max(...series.flatMap(s=>s.values.filter(v=>v!=null))));
  const t0=times[0],t1=Math.max(times[times.length-1],t0+60000);
  const x=t=>L+(t-t0)/(t1-t0)*(W-L-Rr),y=v=>T+(1-v/max)*(H-T-B);
  let g='';for(let i=0;i<=4;i++){const v=max*i/4,yy=y(v);g+='<line class="gridline" x1="'+L+'" x2="'+(W-Rr)+'" y1="'+yy+'" y2="'+yy+'"/><text class="tick" x="'+(L-6)+'" y="'+(yy+4)+'" text-anchor="end">'+(+v.toFixed(v<10?1:0))+'</text>';}
  const nt=Math.max(2,Math.min(6,Math.floor((W-L)/110)));
  for(let i=0;i<=nt;i++){const t=t0+(t1-t0)*i/nt;g+='<text class="tick" x="'+x(t)+'" y="'+(H-6)+'" text-anchor="'+(i===0?'start':i===nt?'end':'middle')+'">'+fmt(new Date(t),t1-t0<1200000?'hms':'hm')+'</text>';}
  const labels=[];if(R.stage[0])labels.push([L,R.stage[0]]);
  for(let i=1;i<R.stage.length;i++)if(R.stage[i]!==R.stage[i-1]&&R.stage[i]){const xx=x(times[i]);labels.push([xx,R.stage[i]]);g+='<line class="stageline" x1="'+xx+'" x2="'+xx+'" y1="'+T+'" y2="'+(H-B)+'"/>';}
  let next=1e9;for(let i=labels.length-1;i>=0;i--){const [xx,n]=labels[i];if(next-xx>60){g+='<text class="stagelabel" x="'+(xx+3)+'" y="'+(T+8)+'">'+n+'</text>';next=xx;}}
  for(const s of series){let d='',pen=false;s.values.forEach((v,i)=>{if(v==null){pen=false;return;}d+=(pen?'L':'M')+x(times[i]).toFixed(1)+','+y(v).toFixed(1);pen=true;});
    g+='<path d="'+d+'" fill="none" stroke="'+s.color+'" stroke-width="2" stroke-linejoin="round"'+(s.dash?' stroke-dasharray="5 4"':'')+'/>';}
  g+='<line class="hover" y1="'+T+'" y2="'+(H-B)+'" visibility="hidden"/><rect x="'+L+'" y="0" width="'+(W-L-Rr)+'" height="'+H+'" fill="transparent"/>';
  box.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+title+'">'+g+'</svg>';
  const svg=box.querySelector('svg'),hl=svg.querySelector('.hover');
  svg.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;
    const t=t0+(px-L)/(W-L-Rr)*(t1-t0);let i=0,best=1e18;times.forEach((tt,j)=>{const dd=Math.abs(tt-t);if(dd<best){best=dd;i=j;}});
    const xx=x(times[i]);hl.setAttribute('x1',xx);hl.setAttribute('x2',xx);hl.setAttribute('visibility','visible');
    tip.innerHTML='<b>'+fmt(new Date(times[i]))+'</b>'+(R.stage[i]?' · '+R.stage[i]:'')+series.map(s=>'<div><span class="sw" style="background:'+s.color+'"></span>'+s.name+': <b>'+(s.values[i]==null?'-':(+s.values[i].toFixed(2)))+'</b> '+unit+'</div>').join('');
    tip.hidden=false;const left=xx*r.width/W;tip.style.top='28px';tip.style.left=(left>r.width*0.6?left-tip.offsetWidth-10:left+10)+'px';});
  svg.addEventListener('mouseleave',()=>{hl.setAttribute('visibility','hidden');tip.hidden=true;});}
const s1=css.getPropertyValue('--s1').trim(),s2=css.getPropertyValue('--s2').trim(),mu=css.getPropertyValue('--muted').trim();
const konst=(v,n)=>R.t.map(()=>v??null);
const gib=a=>a.map(v=>v==null?null:v/1024);
function draw(){
chart('c-ram','Memory','GiB',[{name:'This run',values:R.pipeline_ram_gib,color:s1},{name:'Whole machine',values:R.system_ram_used_gib,color:s2},{name:'Machine total',values:R.system_ram_total_gib,color:mu,dash:true}]);
chart('c-cpu','CPU','cores',[{name:'This run',values:R.pipeline_cpu_cores,color:s1},{name:'Cores available',values:konst(D.machine.cpu_cores),color:mu,dash:true}]);
chart('c-gpu','GPU utilization','%',[{name:'All GPUs',values:R.gpu_util_pct,color:s1}],{max:100});
chart('c-vram','GPU memory','GiB',[{name:'This run',values:gib(R.pipeline_vram_mib),color:s1},{name:'All processes',values:gib(R.gpu_mem_used_mib),color:s2},{name:'GPU limit',values:gib(R.gpu_mem_total_mib||[]),color:mu,dash:true}]);}
draw();let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(draw,150);});
})();
"""


def render(d: dict, refresh_seconds: int = 30) -> str:
    """The full page for the data from :func:`synthdata.dashboard.collect`."""
    payload = {
        "run": d["run"],
        "refresh": refresh_seconds,
        "log": d["log_lines"],
        "resources": {"series": d["resources"]["series"]},
        "machine": d["machine"],
    }
    # "</" inside the JSON would end the script element early.
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    title = (
        f"{STATE_LABEL.get(d['run']['state'], ('Run',))[0]} · {d['config_name'] or 'synthdata'} run"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{CSS}</style></head>
<body><main>
{_header(d, refresh_seconds)}
{_tiles(d)}
{_stage_strip(d)}
{_timeline(d)}
{_hpo(d)}
{_resources(d)}
{_problems(d)}
{_results(d)}
{_log(d)}
<p class="muted small">Built from {_e(d["log_name"])} and the files next to it by synthdata-dashboard. Times are shown in your local time zone; hover a time for UTC.</p>
</main>
<script type="application/json" id="data">{data}</script>
<script>{JS}</script>
</body></html>"""
