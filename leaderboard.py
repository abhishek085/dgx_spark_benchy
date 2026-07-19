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

        # Peak concurrency actually tested per profile -- straight off the capacity_ceiling row's
        # own `concurrency` column, which spark_bench_plus.py already sets to "how far this sweep
        # got" (the break point if it broke, otherwise the last level in the list) before we ever
        # touch it. No extra derivation: just the largest number of parallel requests this profile
        # was actually pushed to during testing.
        ceilings = {}
        for r in rs:
            if r["cmd"] in ("capacity", "hermes") and r["metric"] == "capacity_ceiling":
                c = _num(r["concurrency"])
                if c is not None and c > 0:
                    ceilings[r["profile_or_workload"]] = int(c)
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

        # Detailed single-stream numbers, previously computed but never surfaced past the
        # per-run markdown file -- TTFT/TPOT at the smallest tested context (best case), peak
        # prefill throughput (fastest prompt-processing seen).
        def _tier2_metric_at_min_context(metric):
            candidates = [(int(_num(r["context"])), _num(r["value"])) for r in tier2_rows
                          if r["metric"] == metric and _num(r["context"]) is not None
                          and _num(r["value"]) is not None]
            if not candidates:
                return None
            return min(candidates, key=lambda c: c[0])[1]

        e["ttft_ms"] = _tier2_metric_at_min_context("ttft_ms")
        e["tpot_ms"] = _tier2_metric_at_min_context("tpot_ms")
        prefill_vals = [_num(r["value"]) for r in tier2_rows if r["metric"] == "prefill_tps"]
        prefill_vals = [v for v in prefill_vals if v is not None]
        e["peak_prefill_tps"] = max(prefill_vals) if prefill_vals else None

        # The actual --gpu-memory-utilization this model was served at (from capacity/hermes
        # rows, which record it directly) -- distinct from any other model's value, since
        # different runs can use different levels.
        gpu_utils = [_num(r["gpu_util"]) for r in rs if r["cmd"] in ("capacity", "hermes")]
        gpu_utils = [v for v in gpu_utils if v is not None]
        e["gpu_util"] = gpu_utils[-1] if gpu_utils else None

        # Real machine telemetry sampled during this model's run (see ops/run_bench_loop.py's
        # GpuTelemetry) -- not present for runs from before that was added.
        telemetry_rows = [r for r in rs if r["cmd"] == "telemetry"]
        for metric in ("peak_gpu_util_pct", "avg_gpu_util_pct", "peak_mem_used_gb",
                       "peak_mem_pct", "peak_temp_c", "peak_power_w"):
            vals = [_num(r["value"]) for r in telemetry_rows if r["metric"] == metric]
            vals = [v for v in vals if v is not None]
            e[metric] = vals[-1] if vals else None

        # Headline "how many concurrent users" number: the peak concurrency tested, from the
        # `capacity` command's three profiles (orchestrator/coding_agent/chat_agent) only --
        # they all share the same fixed 1..32 sweep, so they're directly comparable to each
        # other. `hermes` uses its own much wider auto-escalating range (up to 256) and is
        # reported separately, in the Hermes Score section.
        capacity_ceilings = [v for p, v in e["ceilings"].items() if p in ("orchestrator", "coding_agent", "chat_agent")]
        e["max_concurrent"] = max(capacity_ceilings) if capacity_ceilings else None

        for k in ("quality", "capacity_norm", "responsiveness_norm", "hermes_score"):
            vals = [_num(r["value"]) for r in rs
                     if r["cmd"] == "hermes" and r["profile_or_workload"] == "hermes" and r["metric"] == k]
            vals = [v for v in vals if v is not None]
            e[f"hermes_{k}"] = vals[-1] if vals else None

        # A row exists whenever ANY command ran for this label -- but early/legacy runs (before
        # eval+capacity were added to the automated loop) may only have a `speed` pass, with no
        # real quality or capacity data. Surface that plainly instead of showing a ranked-looking
        # row with mostly blanks.
        e["status"] = ("complete" if (e["localscore"] is not None or e["hermes_hermes_score"] is not None)
                        else "partial")

        # Per-domain eval quality (tool_use, structured, safety, etc. -- see eval_scenarios.py's
        # SCENARIOS list for the full domain set) -- raw material for "best model AT X" views
        # that don't fold everything into one composite number. Excludes profile_or_workload==
        # "overall", which is the aggregate LocalScore row, not a domain. Recorded by run_eval's
        # "Domain breakdown" section as metric="domain_quality" (not "quality" -- that name is
        # reserved for the "overall" row's LocalScore component).
        domain_quality = {}
        for r in rs:
            if r["cmd"] == "eval" and r["metric"] == "domain_quality":
                v = _num(r["value"])
                if v is not None:
                    domain_quality[r["profile_or_workload"]] = v
        e["domain_quality"] = domain_quality
        e["tool_use_quality"] = domain_quality.get("tool_use")

        # Coding accuracy at the LOWEST concurrency tested on the coding_agent profile -- as
        # close to a single-user "can it actually write correct code" reading as capacity data
        # gets, deliberately excluding what happens to accuracy under load (that's what the
        # capacity ceiling numbers are for).
        coding_rows = [(int(r["concurrency"]), _num(r["value"])) for r in rs
                       if r["cmd"] == "capacity" and r["profile_or_workload"] == "coding_agent"
                       and r["metric"] == "accuracy" and _num(r["value"]) is not None
                       and str(r["concurrency"]).isdigit()]
        e["coding_accuracy"] = (min(coding_rows, key=lambda c: c[0])[1] * 100) if coding_rows else None

        # Load time (see ops/run_bench_loop.py's write_load_time) -- how long from `vllm serve`
        # to answering, the real cold-start cost of switching to this model.
        load_vals = [_num(r["value"]) for r in rs if r["cmd"] == "startup" and r["metric"] == "load_time_s"]
        load_vals = [v for v in load_vals if v is not None]
        e["load_time_s"] = load_vals[-1] if load_vals else None

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
        "*The peak number of simultaneous requests of that traffic type this box was actually "
        "tested against — `orchestrator` is multi-step tool-chain traffic (the shape an "
        "autonomous agent sends), `coding_agent` is code-generation requests, `chat_agent` is "
        "casual back-and-forth conversation. All three are swept over the same concurrency "
        "levels (1, 2, 4, 8, 16, 32), so the numbers are directly comparable to each other.*\n",
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


# "Best model for..." categories -- deliberately separate from the composite LocalScore/Hermes
# Score views above. Same idea as picking a car for towing vs. a car for fuel economy: one
# "best overall" number can't answer "which model should I use if I specifically care about X",
# so these pick a winner per named use case instead, straight from the same raw per-model data
# already recorded -- no new benchmarking needed to add, remove, or redefine a category here.
# (title, description, eligibility filter, sort field, sort direction, metric label+formatter)
CATEGORIES = [
    ("Fastest (with real answers, not just fast garbage)",
     "Highest peak decode speed among models that still cleared a basic quality bar (Quality "
     "≥ 50/100) -- the pick when speed is what you care about most, as long as it's not "
     "getting things wrong to get there.",
     lambda e: (e.get("quality") or 0) >= 50 and e.get("peak_decode_tps") is not None,
     lambda e: e["peak_decode_tps"], "peak_decode_tps", "{:.0f} tok/s"),
    ("Most accurate (that isn't painfully slow)",
     "Highest Quality score among models still doing at least 20 tokens/sec -- the pick when "
     "correctness matters most and you just need it to not crawl.",
     lambda e: (e.get("peak_decode_tps") or 0) >= 20 and e.get("quality") is not None,
     lambda e: e["quality"], "quality", "{:.0f}/100 quality"),
    ("Best at coding",
     "Highest code-generation accuracy (exec-verified against test cases), speed not "
     "considered at all -- the pick for a coding assistant specifically.",
     lambda e: e.get("coding_accuracy") is not None,
     lambda e: e["coding_accuracy"], "coding_accuracy", "{:.0f}% coding accuracy"),
    ("Best at tool calling",
     "Highest tool-use accuracy from the graded eval suite -- the pick for anything agentic "
     "that lives or dies on correctly calling functions.",
     lambda e: e.get("tool_use_quality") is not None,
     lambda e: e["tool_use_quality"], "tool_use_quality", "{:.0f}/100 tool-use quality"),
    ("Best for a Hermes-style personal agent",
     "Highest Hermes Score -- see the dedicated section below for the full breakdown.",
     lambda e: e.get("hermes_hermes_score") is not None,
     lambda e: e["hermes_hermes_score"], "hermes_hermes_score", "{:.0f}/100 Hermes Score"),
]


def best_for_category(entries, spec):
    _, _, eligible, sort_key, _, _ = spec
    candidates = [e for e in entries if eligible(e)]
    if not candidates:
        return None
    return max(candidates, key=sort_key)


def render_best_for(entries):
    lines = [
        "## Best model for...\n",
        "*Different from the composite scores above on purpose -- these pick a winner for one "
        "specific thing you might care about, straight from the same raw measurements, no "
        "single number trying to represent everything at once. A model can win here and rank "
        "modestly on LocalScore/Hermes Score, or vice versa -- that's the point.*\n",
        "| best for... | winner | why | size |",
        "|---|---|---|---:|",
    ]
    any_winner = False
    for title, desc, eligible, sort_key, metric_key, fmt_str in CATEGORIES:
        winner = best_for_category(entries, (title, desc, eligible, sort_key, metric_key, fmt_str))
        if not winner:
            lines.append(f"| **{title}** | _no qualifying model yet_ | {desc} | — |")
            continue
        any_winner = True
        size = f"{winner['size_b']:.0f}B" if winner.get("size_b") else "—"
        metric_val = fmt_str.format(sort_key(winner))
        lines.append(f"| **{title}** | {winner['model']} ({metric_val}) | {desc} | {size} |")
    if not any_winner:
        lines.append("\n_No models have qualifying data for any category yet._")
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
  table {{ overflow-x: auto; display: block; margin-bottom: 2.2rem; }}
  h2 {{ font-size: 1.1rem; margin: 0 0 0.4rem; }}
  .section-note {{ color: light-dark(#666, #999); font-size: 0.82rem; margin: 0 0 0.8rem; max-width: 80ch; }}
  footer {{ margin-top: 1rem; font-size: 0.8rem; color: light-dark(#888, #777); }}
  footer a {{ color: inherit; }}
  .byline {{ color: light-dark(#999, #666); font-size: 0.78rem; margin: -0.5rem 0 1rem; }}
  .status-complete {{ color: light-dark(#0a6, #3d8); }}
  .status-partial {{ color: light-dark(#b80, #db8); }}
</style>
</head>
<body>
<h1>dgx_spark_benchy leaderboard</h1>
<p class="byline">by the Nokast community</p>
<p class="subtitle">Real numbers from actually running each model on one NVIDIA DGX Spark
(128GB unified memory) — not vendor benchmarks. Every row below sent real requests to a real
server and measured what happened.</p>
<p class="meta">generated from {csv_name} · {n} model(s) benchmarked · sorted by completeness, then score, then most-recently tested</p>
<div class="legend">
  <b>Status</b> — Complete = ran the full eval+capacity+hermes+speed+checks pipeline. Partial = an older/legacy run, only has a subset of numbers; full re-test is queued (see the Model Wiki).
  <b>Format</b> — how compressed the model is (smaller = less memory, usually a little less accurate).
  <b>LocalScore</b> — 0-100 general quality score (accuracy + reliability + efficiency + responsiveness across 22 graded tasks). Independent of Hermes Score — see below.
  <b>Hermes Score</b> — a <i>different</i> 0-100 number specifically for "how good behind a Hermes-style personal agent": 50% real agent-task completion, 30% concurrent-session capacity, 20% responsiveness. A model can score well on one and poorly on the other (e.g. good general quality, but no tool-call parser wired up yet — see Model Wiki).
  <b>Max Document Size</b> — the largest amount of text we actually tested it with and got a real answer back (see the note above the table — this is the tested range, not necessarily the model's true breaking point).
  <b>Peak Speed</b> — decode tokens/sec at its best (smallest-context single-stream request).
  <b>Max Concurrent Users</b> — how many people/sessions could use it at the same time on this one box before it slowed down or started erroring.
  <b>Tool Calling</b> — whether it can reliably call functions/tools (needed for anything agentic, like web search or running code).
</div>

<h2>Best model for...</h2>
<p class="section-note">Different from the table below on purpose — each row here picks a winner
for one specific thing you might care about, straight from the same raw measurements, instead of
one composite number trying to represent everything at once. A model can win a category here and
rank modestly below, or vice versa.</p>
<table>
<thead><tr><th>best for...</th><th>winner</th><th>why</th><th>size</th></tr></thead>
<tbody>
{best_for_rows}
</tbody>
</table>

<h2>Full leaderboard</h2>
<p class="section-note">"Max Concurrent Users" is the peak number of simultaneous
orchestrator/coding/chat requests this model was tested against (same 1→32 sweep for every
model). See "How many people can use it at once" below for the per-traffic-type breakdown, and
the Hermes section for that workload's own (much wider-ranged) capacity sweep.</p>
<table>
<thead><tr>
  <th>model</th><th>label</th><th>status</th><th>size</th><th>format</th>
  <th>LocalScore</th><th>Hermes&nbsp;Score</th>
  <th>Max&nbsp;Document&nbsp;Size</th><th>Peak&nbsp;Speed&nbsp;(tok/s)</th><th>Max&nbsp;Concurrent&nbsp;Users</th>
  <th>Tool&nbsp;Calling</th><th>last tested</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>

<h2>Detailed inference performance</h2>
<p class="section-note">The numbers behind "Peak Speed" — best-case single-stream timing at the
smallest context tested, plus the GPU memory budget each model was actually served at. TTFT =
time to first token, TPOT = time per output token (decode latency), prefill = how fast it reads
the prompt before answering.</p>
<table>
<thead><tr>
  <th>model</th><th>label</th><th>GPU&nbsp;util&nbsp;used</th><th>Load&nbsp;time</th><th>TTFT&nbsp;(ms)</th>
  <th>Prefill&nbsp;(tok/s)</th><th>Decode&nbsp;peak&nbsp;(tok/s)</th><th>TPOT&nbsp;(ms)</th>
</tr></thead>
<tbody>
{perf_rows}
</tbody>
</table>

<h2>Machine telemetry</h2>
<p class="section-note">Real GPU behavior sampled every 5s for the duration of each model's
benchmark run (not the box's own hermes-vllm production traffic) — how hot, how loaded, and how
much memory it actually used, versus the --gpu-memory-utilization flag we asked vLLM to target.
Only present for runs after telemetry capture was added; older runs show —.</p>
<table>
<thead><tr>
  <th>model</th><th>label</th><th>Peak&nbsp;GPU&nbsp;util</th><th>Avg&nbsp;GPU&nbsp;util</th>
  <th>Peak&nbsp;memory&nbsp;used</th><th>Peak&nbsp;temp</th><th>Peak&nbsp;power</th>
</tr></thead>
<tbody>
{telemetry_rows}
</tbody>
</table>

<footer>See the <a href="MODEL_WIKI.md">Model Wiki</a> for which models are officially recommended
for DGX Spark / GB10 and what each one is actually good at, and the repo README for methodology.</footer>
</body>
</html>
"""


def _hfmt(v, digits=1):
    return '<span class="na">—</span>' if v is None else f'<span class="score">{v:.{digits}f}</span>'


def _hfmt_unit(v, unit, digits=0):
    return '<span class="na">—</span>' if v is None else f'<span class="score">{v:.{digits}f}{unit}</span>'


def _rank_entries(entries):
    """Complete runs before partial ones, best score first within each group, most-recently
    tested first as the final tie-break (stable sort applied least-significant key first)."""
    def overall_score(e):
        return e["hermes_hermes_score"] if e["hermes_hermes_score"] is not None else e["localscore"]

    ranked = sorted(entries, key=lambda e: e["timestamp"], reverse=True)
    ranked = sorted(ranked, key=lambda e: (e["status"] != "complete",
                                            overall_score(e) is None,
                                            -(overall_score(e) or 0)))
    return ranked


def render_html(entries, csv_name):
    ranked = _rank_entries(entries)

    best_for_rows = []
    for title, desc, eligible, sort_key, metric_key, fmt_str in CATEGORIES:
        winner = best_for_category(entries, (title, desc, eligible, sort_key, metric_key, fmt_str))
        if not winner:
            best_for_rows.append(
                f"<tr><td><b>{html.escape(title)}</b></td>"
                f"<td><span class=\"na\">no qualifying model yet</span></td>"
                f"<td>{html.escape(desc)}</td><td><span class=\"na\">—</span></td></tr>"
            )
            continue
        size = f"{winner['size_b']:.0f}B" if winner.get("size_b") else '<span class="na">—</span>'
        metric_val = fmt_str.format(sort_key(winner))
        best_for_rows.append(
            f"<tr><td><b>{html.escape(title)}</b></td>"
            f"<td class=\"model\">{html.escape(winner['model'])} "
            f"<span class=\"score\">({html.escape(metric_val)})</span></td>"
            f"<td>{html.escape(desc)}</td><td>{size}</td></tr>"
        )

    rows = []
    for i, e in enumerate(ranked):
        size = f"{e['size_b']:.0f}B" if e.get("size_b") else '<span class="na">—</span>'
        row_cls = ' class="rank1"' if i == 0 and e["status"] == "complete" else ""
        tc = e["tier3_pass"].get("tool_call")
        tc_s = "✅" if tc == 1.0 else ("❌" if tc == 0.0 else '<span class="na">—</span>')
        max_users = e.get("max_concurrent")
        max_users_s = str(max_users) if max_users else '<span class="na">—</span>'
        status_s = (f'<span class="status-complete">complete</span>' if e["status"] == "complete"
                    else '<span class="status-partial">partial</span>')
        rows.append(
            f"<tr{row_cls}>"
            f"<td class=\"model\">{html.escape(e['model'] or '—')}</td>"
            f"<td>{html.escape(e['label'])}</td>"
            f"<td>{status_s}</td>"
            f"<td>{size}</td>"
            f"<td>{html.escape(e.get('precision') or '—')}</td>"
            f"<td>{_hfmt(e['localscore'])}</td>"
            f"<td>{_hfmt(e['hermes_hermes_score'])}</td>"
            f"<td>{html.escape(_fmt_context(e.get('max_context_ok')))}</td>"
            f"<td>{_hfmt(e['peak_decode_tps'], 0)}</td>"
            f"<td>{max_users_s}</td>"
            f"<td>{tc_s}</td>"
            f"<td>{html.escape(e['timestamp'][:10])}</td>"
            f"</tr>"
        )

    perf_rows = []
    for e in ranked:
        gpu_util_s = f"{e['gpu_util']:.2f}" if e.get("gpu_util") is not None else '<span class="na">—</span>'
        perf_rows.append(
            f"<tr>"
            f"<td class=\"model\">{html.escape(e['model'] or '—')}</td>"
            f"<td>{html.escape(e['label'])}</td>"
            f"<td>{gpu_util_s}</td>"
            f"<td>{_hfmt_unit(e.get('load_time_s'), 's')}</td>"
            f"<td>{_hfmt(e['ttft_ms'], 0)}</td>"
            f"<td>{_hfmt(e['peak_prefill_tps'], 0)}</td>"
            f"<td>{_hfmt(e['peak_decode_tps'], 0)}</td>"
            f"<td>{_hfmt(e['tpot_ms'], 0)}</td>"
            f"</tr>"
        )

    telemetry_rows = []
    for e in ranked:
        mem_s = (f"{e['peak_mem_used_gb']:.0f}GB ({e['peak_mem_pct']:.0f}%)"
                 if e.get("peak_mem_used_gb") is not None else None)
        telemetry_rows.append(
            f"<tr>"
            f"<td class=\"model\">{html.escape(e['model'] or '—')}</td>"
            f"<td>{html.escape(e['label'])}</td>"
            f"<td>{_hfmt_unit(e.get('peak_gpu_util_pct'), '%')}</td>"
            f"<td>{_hfmt_unit(e.get('avg_gpu_util_pct'), '%')}</td>"
            f"<td>{html.escape(mem_s) if mem_s else '<span class=\"na\">—</span>'}</td>"
            f"<td>{_hfmt_unit(e.get('peak_temp_c'), '°C')}</td>"
            f"<td>{_hfmt_unit(e.get('peak_power_w'), 'W')}</td>"
            f"</tr>"
        )

    return HTML_TEMPLATE.format(csv_name=html.escape(csv_name), n=len(entries),
                                 best_for_rows="\n".join(best_for_rows),
                                 rows="\n".join(rows), perf_rows="\n".join(perf_rows),
                                 telemetry_rows="\n".join(telemetry_rows))


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
        render_best_for(entries),
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
