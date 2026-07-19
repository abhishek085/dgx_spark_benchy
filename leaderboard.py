#!/usr/bin/env python3
"""
leaderboard.py — build a leaderboard from results/spark_bench_plus.csv.

Every spark_bench_plus.py run appends long-format rows to one CSV, tagged with --label. This
groups all rows sharing a label (across separate tier2/tier3/eval/capacity/hermes invocations of
the same candidate) into one leaderboard entry, with an emphasis on numbers a non-expert can act
on directly: how big a document can this model actually handle, how fast is it, how many people
can use it at the same time, does tool-calling actually work -- not just an abstract score.

Renders:
  overall        general-purpose quality ranking (LocalScore + components)
  orchestrator   capacity ceiling on tool-chain-shaped traffic + tool-call correctness
  hermes         the composite Hermes Score view for personal-agent-harness use
  html           one row per model, every number in one flat table, plain-language headers

Usage:
  ./leaderboard.py                 # print markdown to stdout
  ./leaderboard.py --write         # also write results/LEADERBOARD.md and results/LEADERBOARD.html
  ./leaderboard.py --csv other.csv --out other.md --html other.html --write
"""

import argparse, csv, html, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "results", "spark_bench_plus.csv")
DEFAULT_OUT = os.path.join(HERE, "results", "LEADERBOARD.md")
DEFAULT_HTML_OUT = os.path.join(HERE, "results", "LEADERBOARD.html")


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _approx_size_b(model_name):
    """Best-effort parameter count in billions parsed from the model name (e.g. '...-70B...',
    '...-35B-A3B...' -> 35.0). Used only for the leaderboard's smallest/largest-fit callout."""
    m = re.search(r"(\d+(?:\.\d+)?)B", model_name or "", re.IGNORECASE)
    return float(m.group(1)) if m else None


def _precision(model_name):
    """Best-effort compression format parsed from the model name -- plain-language answer to
    'is this the small/fast version or the full-size one'."""
    n = (model_name or "").lower()
    if "nvfp4" in n:
        return "NVFP4 (4-bit)"
    if "awq" in n:
        return "AWQ (4-bit)"
    if "gptq" in n:
        return "GPTQ (4-bit)"
    if "mxfp4" in n:
        return "MXFP4 (4-bit)"
    if "fp8" in n:
        return "FP8 (8-bit)"
    if "int8" in n:
        return "INT8 (8-bit)"
    return "BF16 (full size)"


def _fmt_context(n):
    if not n:
        return "—"
    return f"{n // 1024}K tokens" if n >= 1024 else f"{n} tokens"


def build_entries(rows):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    entries = []
    for label, rs in by_label.items():
        model = next((r["model"] for r in rs if r.get("model")), "")
        timestamp = max(r["timestamp"] for r in rs if r.get("timestamp"))
        e = {"label": label, "model": model, "timestamp": timestamp, "size_b": _approx_size_b(model),
             "precision": _precision(model)}

        for k in ("quality", "reliability", "efficiency", "responsiveness", "localscore"):
            vals = [_num(r["value"]) for r in rs
                     if r["cmd"] == "eval" and r["profile_or_workload"] == "overall" and r["metric"] == k]
            vals = [v for v in vals if v is not None]
            e[k] = vals[-1] if vals else None

        ceilings = {}
        for r in rs:
            # capacity_ceiling rows come from both the `capacity` command (cmd="capacity") and
            # the `hermes` command (cmd="hermes", profile_or_workload="hermes") — capture both.
            if r["cmd"] in ("capacity", "hermes") and r["metric"] == "capacity_ceiling":
                val = _num(r["value"])
                ceilings[r["profile_or_workload"]] = int(val) if val and val > 0 else None
        e["ceilings"] = ceilings

        passes = {}
        for r in rs:
            if r["cmd"] in ("tier3", "checks") and r["metric"] == "pass":
                v = _num(r["value"])
                if v is not None:
                    passes[r["profile_or_workload"]] = v
        e["tier3_pass"] = passes

        tps_vals = [_num(r["value"]) for r in rs if r["metric"] == "decode_tps"]
        tps_vals = [v for v in tps_vals if v is not None]
        e["decode_tps"] = sum(tps_vals) / len(tps_vals) if tps_vals else None

        # Practical limits, from tier2's context probe (see ops/run_bench_loop.py's
        # context_probe_list) -- the real "how big a document can this model actually handle on
        # this box" and "how fast is it at its best" numbers, not averages across every run.
        tier2_rows = [r for r in rs if r["cmd"] in ("tier2", "speed")]
        ok_contexts = sorted({int(_num(r["context"])) for r in tier2_rows
                               if r["metric"] == "decode_tps" and _num(r["context"]) is not None})
        e["max_context_ok"] = ok_contexts[-1] if ok_contexts else None
        peak_tps_vals = [_num(r["value"]) for r in tier2_rows if r["metric"] == "decode_tps"]
        peak_tps_vals = [v for v in peak_tps_vals if v is not None]
        e["peak_decode_tps"] = max(peak_tps_vals) if peak_tps_vals else None

        max_concurrent = [v for v in e["ceilings"].values() if v]
        e["max_concurrent"] = max(max_concurrent) if max_concurrent else None

        for k in ("quality", "capacity_norm", "responsiveness_norm", "hermes_score"):
            vals = [_num(r["value"]) for r in rs
                     if r["cmd"] == "hermes" and r["profile_or_workload"] == "hermes" and r["metric"] == k]
            vals = [v for v in vals if v is not None]
            e[f"hermes_{k}"] = vals[-1] if vals else None

        entries.append(e)
    return entries


def fmt(v, digits=1):
    return "—" if v is None else f"{v:.{digits}f}"


def render_overall(entries):
    lines = [
        "## Overall quality ranking\n",
        "*General-purpose accuracy across 22 graded tasks (tool use, coding, safety, "
        "instruction-following, and more). `LocalScore` is one 0-100 number combining how often "
        "the model got the task right (Quality), how consistent it was across repeats "
        "(Reliability), how good its accuracy-per-token was (Efficiency), and how fast it started "
        "responding (Responsiveness).*\n",
        "| model | label | LocalScore | Quality | Reliability | Efficiency | Responsiveness | tokens/sec | last run |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(entries, key=lambda e: (e["localscore"] is None, -(e["localscore"] or 0)))
    for e in ranked:
        lines.append(
            f"| {e['model'] or '—'} | {e['label']} | {fmt(e['localscore'])} | {fmt(e['quality'])} | "
            f"{fmt(e['reliability'])} | {fmt(e['efficiency'])} | {fmt(e['responsiveness'])} | "
            f"{fmt(e['decode_tps'])} | {e['timestamp']} |"
        )
    return "\n".join(lines) + "\n"


def render_orchestrator(entries):
    lines = [
        "## How many people can use it at once\n",
        "*Each \"ceiling\" is the largest number of simultaneous conversations this box sustained "
        "on that kind of traffic before answers got measurably worse or slower — the real answer "
        "to \"how many users can share this model.\" `orchestrator` is multi-step tool-chain "
        "traffic (the shape an autonomous agent sends), `coding_agent` is code-generation "
        "requests, `chat_agent` is casual back-and-forth conversation.*\n",
        "| model | label | tool-chain agents | coding agents | chat sessions | tool-calling works? |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    ranked = sorted(entries, key=lambda e: -(e["ceilings"].get("orchestrator") or -1))
    for e in ranked:
        c = e["ceilings"]
        tc = e["tier3_pass"].get("tool_call")
        tc_s = "✅" if tc == 1.0 else ("❌" if tc == 0.0 else "—")
        lines.append(
            f"| {e['model'] or '—'} | {e['label']} | {c.get('orchestrator') or '—'} | "
            f"{c.get('coding_agent') or '—'} | {c.get('chat_agent') or '—'} | {tc_s} |"
        )
    return "\n".join(lines) + "\n"


def render_hermes(entries):
    hermes_entries = [e for e in entries if e.get("hermes_hermes_score") is not None]
    lines = [
        "## Hermes Benchmark — best model for a personal-agent harness\n",
        "*One 0-100 number for \"how good would this model be behind a Hermes-style personal "
        "agent\": 50% how well it actually completes real agent tasks (tool use, web search, "
        "remembering things across a conversation), 30% how many concurrent sessions it "
        "sustains, 20% how quickly it starts responding.*\n",
    ]
    if not hermes_entries:
        lines.append("_No `hermes` benchmark runs recorded yet._\n")
        return "\n".join(lines) + "\n"

    lines += [
        "| model | label | Hermes Score | Quality | Capacity | Responsiveness | approx size |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(hermes_entries, key=lambda e: -(e["hermes_hermes_score"] or 0))
    for e in ranked:
        size = f"{e['size_b']:.0f}B" if e.get("size_b") else "—"
        lines.append(
            f"| {e['model'] or '—'} | {e['label']} | {fmt(e['hermes_hermes_score'])} | "
            f"{fmt(e['hermes_quality'])} | {fmt(e['hermes_capacity_norm'])} | "
            f"{fmt(e['hermes_responsiveness_norm'])} | {size} |"
        )
    lines.append("")

    passing = [e for e in hermes_entries if (e["hermes_hermes_score"] or 0) >= 70 and e.get("size_b")]
    if passing:
        smallest = min(passing, key=lambda e: e["size_b"])
        largest = max(passing, key=lambda e: e["size_b"])
        lines.append(
            f"**Smallest model that's actually good enough (70+/100):** {smallest['model']} "
            f"(~{smallest['size_b']:.0f}B, score {smallest['hermes_hermes_score']:.1f}) — the one "
            f"to reach for if you're tight on memory.  "
        )
        lines.append(
            f"**Largest model tested:** {largest['model']} "
            f"(~{largest['size_b']:.0f}B, score {largest['hermes_hermes_score']:.1f}) — the "
            f"highest-quality option this box can run.\n"
        )
    return "\n".join(lines) + "\n"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>dgx_spark_benchy leaderboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1150px; margin: 2rem auto; padding: 0 1rem;
    background: light-dark(#fafafa, #14161a); color: light-dark(#1a1a1a, #e8e8e8);
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: light-dark(#555, #aaa); font-size: 0.95rem; margin: 0 0 1rem; max-width: 70ch; }}
  .meta {{ color: light-dark(#888, #777); font-size: 0.8rem; margin-bottom: 1rem; }}
  .legend {{
    background: light-dark(#f0f0f0, #1c1e23); border-radius: 8px; padding: 0.8rem 1.1rem;
    font-size: 0.82rem; color: light-dark(#444, #bbb); margin-bottom: 1.5rem; line-height: 1.7;
  }}
  .legend b {{ color: light-dark(#1a1a1a, #e8e8e8); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.87rem; }}
  th, td {{ text-align: right; padding: 0.55rem 0.7rem; border-bottom: 1px solid light-dark(#ddd, #333); }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ font-weight: 600; color: light-dark(#444, #bbb); border-bottom: 2px solid light-dark(#ccc, #444);
       position: sticky; top: 0; background: light-dark(#fafafa, #14161a); }}
  tr:hover {{ background: light-dark(#f0f0f0, #1f2228); }}
  td.model {{ font-family: ui-monospace, monospace; font-size: 0.8rem; }}
  .rank1 {{ font-weight: 700; }}
  .rank1 td.model {{ color: light-dark(#0a6, #3d8); }}
  .na {{ color: light-dark(#bbb, #555); }}
  .score {{ font-variant-numeric: tabular-nums; }}
  table {{ overflow-x: auto; display: block; }}
  footer {{ margin-top: 2rem; font-size: 0.8rem; color: light-dark(#888, #777); }}
  footer a {{ color: inherit; }}
  .byline {{ color: light-dark(#999, #666); font-size: 0.78rem; margin: -0.5rem 0 1rem; }}
</style>
</head>
<body>
<h1>dgx_spark_benchy leaderboard</h1>
<p class="byline">by the Nokast community</p>
<p class="subtitle">Real numbers from actually running each model on one NVIDIA DGX Spark
(128GB unified memory) — not vendor benchmarks. Every row below sent real requests to a real
server and measured what happened.</p>
<p class="meta">generated from {csv_name} · {n} model(s) benchmarked · sorted by Overall Score</p>
<div class="legend">
  <b>Format</b> — how compressed the model is (smaller = less memory, usually a little less accurate).
  <b>Overall Score</b> — 0-100, how good this model is for a personal-agent use case (Hermes Score if tested that way, otherwise general quality score).
  <b>Max Document Size</b> — the largest amount of text we could hand it in one go and still get a real answer back.
  <b>Speed</b> — tokens per second at its best, i.e. how fast the words come out once it starts answering.
  <b>Max Concurrent Users</b> — how many people/sessions could use it at the same time on this one box before it slowed down or started erroring.
  <b>Tool Calling</b> — whether it can reliably call functions/tools (needed for anything agentic, like web search or running code).
</div>
<table>
<thead><tr>
  <th>model</th><th>label</th><th>size</th><th>format</th><th>Overall&nbsp;Score</th>
  <th>Max&nbsp;Document&nbsp;Size</th><th>Speed&nbsp;(tok/s)</th><th>Max&nbsp;Concurrent&nbsp;Users</th>
  <th>Tool&nbsp;Calling</th><th>last tested</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
<footer>See the <a href="MODEL_WIKI.md">Model Wiki</a> for which models are officially recommended
for DGX Spark / GB10 and what each one is actually good at, and the repo README for methodology.</footer>
</body>
</html>
"""


def _hfmt(v, digits=1):
    return '<span class="na">—</span>' if v is None else f'<span class="score">{v:.{digits}f}</span>'


def render_html(entries, csv_name):
    def sort_key(e):
        primary = e["hermes_hermes_score"] if e["hermes_hermes_score"] is not None else e["localscore"]
        return (primary is None, -(primary or 0))

    ranked = sorted(entries, key=sort_key)
    rows = []
    for i, e in enumerate(ranked):
        size = f"{e['size_b']:.0f}B" if e.get("size_b") else '<span class="na">—</span>'
        row_cls = ' class="rank1"' if i == 0 else ""
        overall = e["hermes_hermes_score"] if e["hermes_hermes_score"] is not None else e["localscore"]
        tc = e["tier3_pass"].get("tool_call")
        tc_s = "✅" if tc == 1.0 else ("❌" if tc == 0.0 else '<span class="na">—</span>')
        max_users = e.get("max_concurrent")
        rows.append(
            f"<tr{row_cls}>"
            f"<td class=\"model\">{html.escape(e['model'] or '—')}</td>"
            f"<td>{html.escape(e['label'])}</td>"
            f"<td>{size}</td>"
            f"<td>{html.escape(e.get('precision') or '—')}</td>"
            f"<td>{_hfmt(overall)}</td>"
            f"<td>{html.escape(_fmt_context(e.get('max_context_ok')))}</td>"
            f"<td>{_hfmt(e['peak_decode_tps'], 0)}</td>"
            f"<td>{max_users if max_users else '<span class=\"na\">—</span>'}</td>"
            f"<td>{tc_s}</td>"
            f"<td>{html.escape(e['timestamp'][:10])}</td>"
            f"</tr>"
        )
    return HTML_TEMPLATE.format(csv_name=html.escape(csv_name), n=len(entries), rows="\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description="Build a markdown + HTML leaderboard from spark_bench_plus results")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT, help="markdown output path")
    ap.add_argument("--html", default=DEFAULT_HTML_OUT, help="HTML output path")
    ap.add_argument("--write", action="store_true", help="write both --out and --html")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"no results yet at {args.csv} — run a benchmark first (see quickbench.sh).")
        return

    rows = load_rows(args.csv)
    entries = build_entries(rows)
    if not entries:
        print("no labeled runs found in CSV.")
        return

    csv_name = os.path.relpath(args.csv, HERE)
    text = "\n".join([
        "# dgx_spark_benchy leaderboard\n",
        f"_generated from `{csv_name}`, {len(entries)} labeled run(s)_\n",
        render_overall(entries),
        render_orchestrator(entries),
        render_hermes(entries),
    ])
    print(text)

    if args.write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\n-> wrote {args.out}")

        os.makedirs(os.path.dirname(args.html), exist_ok=True)
        with open(args.html, "w") as f:
            f.write(render_html(entries, csv_name))
        print(f"-> wrote {args.html}")


if __name__ == "__main__":
    main()
