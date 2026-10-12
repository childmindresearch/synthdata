"""HTML for the run dashboard (see :mod:`synthdata.dashboard`).

One self-contained page: inline CSS and JavaScript, no network requests, so it
works opened straight from disk on an offline machine. Tables, the model
timeline and the HPO sparklines are drawn here; the resource charts and the
log filter are drawn in the browser from the JSON embedded in the page.
"""

import html
import json
from datetime import UTC, datetime

from synthdata.dashboard import _duration

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
}
BASELINE_TAG = ' <span class="muted small">baseline</span>'
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
        models_sub = f"{step['label']}: {step['running']} running" + (
            f", {step['failed']} failed" if step["failed"] else ""
        )
    else:
        models_value, models_sub = "-", "no per-model workers yet"

    hpo = d["hpo"]
    if hpo:
        trials_done = sum(s["complete"] + s["pruned"] for s in hpo)
        target = sum(s["target"] or 0 for s in hpo)
        hpo_value = f"{trials_done}" + (f" / {target}" if target else "")
        hpo_sub = (
            f"{sum(s['running'] for s in hpo)} running, {sum(s['failed'] for s in hpo)} failed"
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


def _timeline(d: dict) -> str:
    tasks = d["tasks"]
    if not d["steps"] and not tasks:
        return (
            '<section class="card"><h2>Models</h2><p class="empty">No per-model workers have started. '
            "Each model's start and finish appear here when <code>compute.workers</code> is more than 1."
            "</p></section>"
        )
    t1 = _ts(d["run"]["ended"]) or _ts(d["generated"])
    t0 = _ts(d["run"]["started"]) or min((_ts(t["start"]) for t in tasks if t["start"]), default=t1)
    span = max(t1 - t0, 1)
    hpo = {s["model"]: s for s in d["hpo"]}
    blocks = []
    for step in d["steps"] or [{"label": t["step"]} for t in tasks][:1]:
        label = step["label"]
        mine = sorted((t for t in tasks if t["step"] == label), key=lambda t: t["order"])
        rows = []
        for t in mine:
            start = _ts(t["start"])
            end = _ts(t["end"]) or t1
            if start is None:
                bar = ""
            else:
                left = (start - t0) / span * 100
                width = max((end - start) / span * 100, 0.4)
                tip = f"{t['model']} · {t['status']} · {_duration(t['duration_s'])}"
                bar = (
                    f'<div class="bar {t["status"]}" style="left:{left:.2f}%;width:{width:.2f}%" '
                    f'title="{_e(tip)}"></div>'
                )
            extra = ""
            study = hpo.get(t["model"]) if label == "generation" else None
            if study and study["target"]:
                n = study["complete"] + study["pruned"]
                extra = f'<span class="muted small">HPO {n}/{study["target"]}</span>'
            err = f'<div class="row-error">{_e(t["error"])}</div>' if t.get("error") else ""
            rows.append(
                f'<div class="g-row"><div class="g-name"><span class="model">{_e(t["model"])}</span>'
                f'{_badge(t["status"])}<span class="g-dur">{_e(_duration(t["duration_s"]))}</span>{extra}{err}</div>'
                f'<div class="g-track">{bar}</div></div>'
            )
        queued = step.get("total", len(mine)) - len(mine)
        if queued > 0:
            rows.append(
                f'<div class="g-row"><div class="g-name">{_badge("queued")}'
                f'<span class="muted small">{queued} more model{"s" * (queued != 1)} waiting for a worker</span>'
                f'</div><div class="g-track"></div></div>'
            )
        pool = ""
        if step.get("workers"):
            pool = f" · {step['workers']} at a time, {step['cores']} CPU threads each"
        summary = f"{step.get('done', 0)} of {step.get('total', len(mine))} done{pool}"
        blocks.append(
            f'<div class="g-step"><h3>{_e(label)} <span class="muted small">{_e(summary)}</span></h3>'
            f"{''.join(rows)}</div>"
        )
    axis = (
        f'<div class="g-row g-axis"><div class="g-name"></div><div class="g-track">'
        f'<span style="left:0">{_time(d["run"]["started"], "time")}</span>'
        f'<span style="right:0">{_time(d["run"]["ended"] or d["generated"], "time")}</span></div></div>'
    )
    return f'<section class="card"><h2>Models</h2>{"".join(blocks)}{axis}</section>'


def _sparkline(curve: list, maximize: bool) -> str:
    if len(curve) < 2:
        return '<span class="muted small">-</span>'
    w, h, pad = 150, 32, 3
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

    dots = "".join(
        f'<circle cx="{px(c[0]):.1f}" cy="{py(c[1]):.1f}" r="1.6" class="{"dot" if c[3] else "dot off"}"/>'
        for c in curve
    )
    best = [c for c in curve if c[2] is not None]
    path = ""
    for i, c in enumerate(best):
        x, y = px(c[0]), py(c[2])
        path += f"M{x:.1f},{y:.1f}" if i == 0 else f"H{x:.1f}V{y:.1f}"
    if best:
        path += f"H{px(x1):.1f}"
    label = f"best so far by trial ({'higher' if maximize else 'lower'} is better)"
    return (
        f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{label}">'
        f'<title>{label}</title>{dots}<path d="{path}" class="best"/></svg>'
    )


def _hpo(d: dict) -> str:
    if d["hpo_error"]:
        body = f'<p class="empty">{_e(d["hpo_error"])}</p>'
    elif not d["hpo"]:
        body = '<p class="empty">No HPO study has started for this experiment.</p>'
    else:
        running_models = {t["model"] for t in d["tasks"] if t["status"] == "running"}
        rows = []
        for s in d["hpo"]:
            n = s["complete"] + s["pruned"]
            target = s["target"]
            between_trials = s["model"] in running_models and not (target and n >= target)
            if s["running"] or (between_trials and d["run"]["state"] == "running"):
                status = "running"
            elif target and n >= target:
                status = "done"
            elif d["run"]["state"] == "running" and s["model"] not in running_models:
                status = "paused" if n else "queued"
            else:
                status = "done" if not target else "stopped"
            pct = min(n / target * 100, 100) if target else 0
            prog = f'<div class="prog"><div style="width:{pct:.1f}%"></div></div>' if target else ""
            counts = f"{n}{f' / {target}' if target else ''}"
            detail = f"{s['complete']} complete, {s['pruned']} pruned, {s['failed']} failed"
            err = (
                f'<details><summary class="small">last failure</summary><pre>{_e(s["last_error"])}</pre></details>'
                if s["last_error"]
                else ""
            )
            eta = _duration(s["eta_s"]) if s["eta_s"] is not None and status == "running" else "-"
            rows.append(
                f'<tr><td class="model">{_e(s["model"])}</td><td>{_badge(status)}</td>'
                f'<td><div class="num">{counts}</div>{prog}<div class="muted small">{detail}</div></td>'
                f'<td class="num">{_num(s["best"])}</td><td>{_sparkline(s["curve"], s["maximize"])}</td>'
                f'<td class="num">{_e(_duration(s["median_trial_s"]))}</td><td class="num">{_e(eta)}</td>'
                f"<td>{err}</td></tr>"
            )
        body = (
            '<div class="scroll"><table><thead><tr><th>Model</th><th>Status</th><th>Trials</th>'
            '<th class="num">Best</th><th>Best so far</th><th class="num">Median trial</th>'
            '<th class="num">Time left</th><th></th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div>"
            '<p class="muted small">Best counts only trials that pass the sanity screens. '
            "Time left is remaining trials times the median of the last ten, capped by the study's timeout.</p>"
        )
    return f'<section class="card"><h2>Hyperparameter search</h2>{body}</section>'


def _resources(d: dict) -> str:
    res = d["resources"]
    if not res["n_rows"]:
        return (
            '<section class="card"><h2>Resources</h2><p class="empty">No resource samples yet '
            "(the first row lands one monitor interval after the start).</p></section>"
        )
    tasks = res["tasks"]
    rows = "".join(
        f"<tr><td>{_e(t['task'])}</td><td>{_badge('running') if t.get('active') else ''}</td>"
        f'<td class="num">{_num(t.get("ram_now"), 1)}</td><td class="num">{_num(t["ram_peak"], 1)}</td>'
        f'<td class="num">{_num(t.get("cpu_now"), 1)}</td><td class="num">{_num(t["cpu_peak"], 1)}</td>'
        f'<td class="num">{_num((t["vram_peak"] or 0) / 1024, 1)}</td></tr>'
        for t in tasks[:30]
    )
    table = (
        '<h3>By model</h3><div class="scroll"><table><thead><tr><th>Task</th><th></th>'
        '<th class="num">RAM now (GiB)</th><th class="num">RAM peak</th><th class="num">CPU now (cores)</th>'
        '<th class="num">CPU peak</th><th class="num">GPU memory peak (GiB)</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        if tasks
        else ""
    )
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


def _results(d: dict) -> str:
    r = d["results"]
    if not r:
        return (
            '<section class="card"><h2>Results</h2><p class="empty">Evaluation has not written its '
            "ranking yet.</p></section>"
        )
    parts = []
    if "ranking" in r:
        rows = "".join(
            f'<tr class="{"baseline" if x["baseline"] else ""}"><td class="num">{_e(x["rank"])}</td>'
            f'<td class="model">{_e(x["model"])}{BASELINE_TAG if x["baseline"] else ""}</td>'
            f'<td class="num">{_num(x["overall"])}</td>'
            f'<td class="num muted">{_num(x["low"])} to {_num(x["high"])}</td>'
            f'<td>{_badge("done") if x["tied"] else ""}</td><td class="num">{_e(x["n"])}</td></tr>'
            for x in r["ranking"][:12]
        )
        parts.append(
            '<h3>Ranking</h3><div class="scroll"><table><thead><tr><th class="num">Rank</th><th>Model</th>'
            '<th class="num">Overall</th><th class="num">95% CI</th><th>Tied with best</th>'
            f'<th class="num">Replicates</th></tr></thead><tbody>{rows}</tbody></table></div>'
        )
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


def _tail(d: dict) -> str:
    options = "".join(f'<option value="{_e(t)}">{_e(t)}</option>' for t in d["tasks_in_tail"])
    return f"""
<section class="card"><h2>Log <span class="muted small">{_e(d["log"])}</span></h2>
  <div class="filters">
    <label>Model <select id="tail-task"><option value="">all</option>{options}</select></label>
    <label>Contains <input id="tail-text" type="search" placeholder="text"></label>
    <label><input id="tail-problems" type="checkbox"> errors and warnings only</label>
  </div>
  <pre id="tail" class="tail"></pre>
</section>"""


# ---------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--good-ink:#006300;--warn:#fab219;
--warn-ink:#8a5a00;--bad:#d03b3b;--run:#2a78d6;--track:#f0efec;--shadow:0 1px 2px rgba(0,0,0,.06)}
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
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
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
.g-step h3{margin-top:4px}.g-row{display:grid;grid-template-columns:minmax(220px,34%) 1fr;gap:12px;align-items:center;padding:3px 0;border-bottom:1px solid var(--grid)}
.g-name{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;min-width:0}.model{font-weight:600;overflow-wrap:anywhere}
.g-dur{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:12px}
.g-track{position:relative;height:14px;background:var(--track);border-radius:4px}
.bar{position:absolute;top:2px;bottom:2px;border-radius:4px;background:var(--axis)}
.bar.running{background:var(--run)}.bar.done{background:var(--good)}.bar.failed,.bar.stopped{background:var(--bad)}
.bar.running{background-image:linear-gradient(90deg,transparent 60%,rgba(255,255,255,.35));background-size:40px 100%;animation:slide 1.2s linear infinite}
@keyframes slide{to{background-position:40px 0}}
.g-axis{border:0}.g-axis .g-track{background:none;height:16px;font-size:11px;color:var(--muted)}.g-axis span{position:absolute}
.row-error{flex-basis:100%;font-size:12px;color:var(--bad);overflow-wrap:anywhere}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:middle}
th{font-size:12px;font-weight:600;color:var(--muted)}tr.baseline td{color:var(--ink2)}td.model{white-space:nowrap}
.prog{height:4px;background:var(--track);border-radius:2px;margin:3px 0;min-width:90px}.prog div{height:100%;background:var(--s1);border-radius:2px}
.spark .dot{fill:var(--muted);opacity:.55}.spark .dot.off{fill:none;stroke:var(--muted);stroke-width:.8}.spark .best{fill:none;stroke:var(--s1);stroke-width:2}
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
  if(f==='time')return t;if(f==='hm')return t;
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

// ---- log tail
const tail=document.getElementById('tail'),selTask=document.getElementById('tail-task'),txt=document.getElementById('tail-text'),onlyP=document.getElementById('tail-problems');
const isErr=l=>/ (ERROR|CRITICAL): |Traceback|failed/.test(l),isWarn=l=>/ WARNING: |Warning: /.test(l);
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function renderTail(){const task=selTask.value,q=txt.value.toLowerCase(),p=onlyP.checked;
  store.set('dash-tail',JSON.stringify([task,txt.value,p]));
  const lines=D.tail.filter(l=>(!task||l.startsWith('['+task+']'))&&(!q||l.toLowerCase().includes(q))&&(!p||isErr(l)||isWarn(l)));
  tail.innerHTML=lines.map(l=>isErr(l)?'<span class="e">'+esc(l)+'</span>':isWarn(l)?'<span class="w">'+esc(l)+'</span>':esc(l)).join('\n')||'<span class="muted">no matching lines</span>';
  tail.scrollTop=tail.scrollHeight;}
try{const s=JSON.parse(store.get('dash-tail')||'null');if(s){selTask.value=s[0];txt.value=s[1];onlyP.checked=s[2];}}catch(e){}
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
  for(let i=0;i<=nt;i++){const t=t0+(t1-t0)*i/nt;g+='<text class="tick" x="'+x(t)+'" y="'+(H-6)+'" text-anchor="'+(i===0?'start':i===nt?'end':'middle')+'">'+fmt(new Date(t),'hm')+'</text>';}
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
        "tail": d["tail"],
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
{_tail(d)}
<p class="muted small">Built from {_e(d["log_name"])} and the files next to it by synthdata-dashboard. Times are shown in your local time zone; hover a time for UTC.</p>
</main>
<script type="application/json" id="data">{data}</script>
<script>{JS}</script>
</body></html>"""
