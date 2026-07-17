#!/usr/bin/env python3
"""
leaderboard.py — build a markdown leaderboard from results/spark_bench_plus.csv.

Every spark_bench_plus.py run appends long-format rows to one CSV, tagged
with --label. This groups all rows sharing a label (across separate tier3 /
eval / capacity invocations of the same candidate) into one leaderboard
entry, and renders two views:

  overall        LocalScore + component scores, sorted best first — the
                 general-purpose ranking.
  orchestrator   Capacity ceiling on the `orchestrator` workload profile
                 (the ReAct/tool-chain shape Hermes-style harnesses send),
                 plus tool-call correctness — the view that matters if
                 you're screening candidates specifically for Hermes.

Usage:
  ./leaderboard.py                 # print to stdout
  ./leaderboard.py --write         # also write results/LEADERBOARD.md
  ./leaderboard.py --csv other.csv --out other.md --write
"""

import argparse, csv, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "results", "spark_bench_plus.csv")
DEFAULT_OUT = os.path.join(HERE, "results", "LEADERBOARD.md")


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


def build_entries(rows):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    entries = []
    for label, rs in by_label.items():
        model = next((r["model"] for r in rs if r.get("model")), "")
        timestamp = max(r["timestamp"] for r in rs if r.get("timestamp"))
        e = {"label": label, "model": model, "timestamp": timestamp, "size_b": _approx_size_b(model)}

        for k in ("quality", "reliability", "efficiency", "responsiveness", "localscore"):
            vals = [_num(r["value"]) for r in rs
                     if r["cmd"] == "eval" and r["profile_or_workload"] == "overall" and r["metric"] == k]
            vals = [v for v in vals if v is not None]
            e[k] = vals[-1] if vals else None

        ceilings = {}
        for r in rs:
            if r["cmd"] == "capacity" and r["metric"] == "capacity_ceiling":
                val = _num(r["value"])
                ceilings[r["profile_or_workload"]] = int(val) if val and val > 0 else None
        e["ceilings"] = ceilings

        passes = {}
        for r in rs:
            if r["cmd"] == "tier3" and r["metric"] == "pass":
                v = _num(r["value"])
                if v is not None:
                    passes[r["profile_or_workload"]] = v
        e["tier3_pass"] = passes

        tps_vals = [_num(r["value"]) for r in rs if r["metric"] == "decode_tps"]
        tps_vals = [v for v in tps_vals if v is not None]
        e["decode_tps"] = sum(tps_vals) / len(tps_vals) if tps_vals else None

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
        "## Overall leaderboard\n",
        "| model | label | LocalScore | Quality | Reliability | Efficiency | Responsiveness | decode tok/s | last run |",
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
        "## Orchestrator / Hermes-style view\n",
        "*Capacity ceiling = max concurrent agent sessions sustained on that workload profile "
        "before accuracy/error thresholds break. `orchestrator` is the ReAct/tool-chain shape "
        "Hermes-style harnesses send — sort by that column when screening a candidate "
        "specifically for Hermes.*\n",
        "| model | label | orchestrator ceiling | coding_agent ceiling | chat_agent ceiling | tool_call pass |",
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
        "## Hermes Benchmark\n",
        "*Composite score for running a Hermes-style agent harness: 50% task quality (tool use, "
        "web search, long-context recall, multi-turn state), 30% capacity ceiling on the `hermes` "
        "profile (normalized to a target concurrent-session count), 20% responsiveness (TTFT at "
        "that ceiling). Run via `spark_bench_plus.py hermes ...`.*\n",
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
            f"**Smallest model clearing a 70/100 Hermes Score:** {smallest['model']} "
            f"(~{smallest['size_b']:.0f}B, score {smallest['hermes_hermes_score']:.1f})  "
        )
        lines.append(
            f"**Largest model tested:** {largest['model']} "
            f"(~{largest['size_b']:.0f}B, score {largest['hermes_hermes_score']:.1f})\n"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Build a markdown leaderboard from spark_bench_plus results")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true", help="also write the leaderboard to --out")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"no results yet at {args.csv} — run a benchmark first (see quickbench.sh).")
        return

    rows = load_rows(args.csv)
    entries = build_entries(rows)
    if not entries:
        print("no labeled runs found in CSV.")
        return

    text = "\n".join([
        "# dgx_spark_benchy leaderboard\n",
        f"_generated from `{os.path.relpath(args.csv, HERE)}`, {len(entries)} labeled run(s)_\n",
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


if __name__ == "__main__":
    main()
