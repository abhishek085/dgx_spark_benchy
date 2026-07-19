#!/usr/bin/env python3
"""
model_wiki.py — build results/MODEL_WIKI.md: a plain-language guide to which models make sense
to run on an NVIDIA DGX Spark (GB10, 128GB unified memory) — what each one is actually good at,
not just a score table, and what's still to come.

Three kinds of model entries:
  tested   models this box has actually benchmarked (results/spark_bench_plus.csv) — real
           numbers pulled in via leaderboard.py, a curated note on whether it's one of the
           handful of models NVIDIA/Hugging Face/Unsloth specifically call out for GB10, and a
           "what it's best at" writeup derived from the same raw per-domain/category data
           leaderboard.py's "Best model for..." section uses.
  queued   models worth trying that haven't been benchmarked here yet — auto-drops off this list
           the moment a benchmark run actually covers it (matched against the CSV, not hardcoded
           by hand), so this file doesn't go stale as batches complete.
  skipped  checked and deliberately not queued (too big, wrong format, pre-2026, etc.)

The curated facts below (official-recommendation status, family notes) were gathered by hand
from NVIDIA/Hugging Face/Unsloth sources on 2026-07-18/19 — re-check before trusting them stale.

Usage:
  ./model_wiki.py                # print markdown to stdout
  ./model_wiki.py --write        # also write results/MODEL_WIKI.md
"""
import argparse, os
import leaderboard as lb

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "MODEL_WIKI.md")
REPO_URL = "https://github.com/abhishek085/dgx_spark_benchy"

# Curated per-family notes. Matched against a benchmarked entry's model name by substring, so one
# entry covers every quant variant of a family (nvidia/unsloth/google/Qwen NVFP4 + BF16 etc).
FAMILY_NOTES = [
    ("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B",
     "Third-party/self-published NVFP4 quants of Qwen's own model — not on NVIDIA's short list "
     "of DGX Spark reference models, but a strong MoE agentic-coding model in its own right."),
    ("qwen3.6-27b", "Qwen3.6 27B",
     "Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official "
     "GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) "
     "run well here."),
    ("gemma-4-26b-a4b", "Gemma 4 26B-A4B",
     "Called out repeatedly as a popular community choice specifically for GB10 — a good "
     "efficiency/quality balance for a MoE model in this size class."),
    ("gemma-4-31b", "Gemma 4 31B",
     "Google's larger dense Gemma 4 — not MoE, so heavier per-token, but strong general quality."),
    ("nemotron-3-nano-30b-a3b", "Nemotron 3 Nano",
     "NVIDIA's own official small reference model for DGX Spark — the primary example in "
     "NVIDIA's own DGX Spark playbook."),
    ("ornith", "Ornith 1.0",
     "Not an NVIDIA/HF/Unsloth official GB10 pick — a dense agentic-coding model built on "
     "Qwen 3.5 by deepreinforce-ai, tuned for terminal-based coding agents. Added on request."),
]

# Queued: real repo IDs, confirmed to exist via web research, not yet run through bench-all on
# this box. Kept here (not just in ops/models.txt) because this file is public and ops/ is
# gitignored — the roadmap itself is fine to share even though the container internals aren't.
# Filtered against actually-tested models at render time, so this list self-updates as batches
# complete instead of needing to be hand-edited every run.
QUEUED_GROUPS = [
    ("Re-testing with the fixed pipeline",
     "The rest of the original 12-model batch, queued to re-run now that the max-num-seqs "
     "scaling, Gemma4 tool-calling, context-window detection, and speed-aware scoring fixes are in.", [
        ("nvidia/Qwen3.6-27B-NVFP4", ""), ("unsloth/Qwen3.6-27B-NVFP4", ""),
        ("unsloth/Qwen3.6-35B-A3B-NVFP4", ""), ("nvidia/Gemma-4-31B-IT-NVFP4", ""),
        ("unsloth/gemma-4-31B-it-NVFP4", ""), ("unsloth/gemma-4-26B-A4B-it-NVFP4", ""),
        ("Qwen/Qwen3.6-27B", "BF16"), ("Qwen/Qwen3.6-35B-A3B", "BF16"),
        ("google/gemma-4-31B-it", "BF16"), ("google/gemma-4-26B-A4B-it", "BF16"),
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", ""), ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", ""),
        ("nvidia/Gemma-4-26B-A4B-NVFP4", ""), ("deepreinforce-ai/Ornith-1.0-35B", ""),
     ]),
    ("Smallest meaningful",
     "Gemma 4's on-device tier (2026-04) — is there any real task this box can run "
     "near-instantly with tiny memory use?", [
        ("google/gemma-4-E2B-it", "~2.3B effective, BF16"),
        ("google/gemma-4-E4B-it", "~4.5B effective, BF16"),
     ]),
    ("NVIDIA's other official DGX Spark models",
     "The rest of NVIDIA's own Nemotron 3 lineup (2026-03).", [
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "120B total/12B active MoE, NVFP4"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "same model, uncompressed"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", ""),
     ]),
    ("Nemotron 3 Nano Omni",
     "Multimodal (video/audio/image/text) Mamba2-Transformer hybrid MoE, reasoning-tuned, 2026 "
     "— 3 precisions to compare.", [
        ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16", ""),
        ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8", ""),
        ("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", ""),
     ]),
    ("DiffusionGemma 26B-A4B-it",
     "Same Gemma 4 MoE backbone, but block-diffusion text generation instead of "
     "token-by-token autoregression — a genuinely different generation architecture, 2026.", [
        ("google/diffusiongemma-26B-A4B-it", "BF16"),
        ("nvidia/diffusiongemma-26B-A4B-it-NVFP4", "NVFP4"),
     ]),
    ("Biggest that fits",
     "A community 25%-REAP-pruned NVFP4 quant of MiniMax M2.7 (2026-03), purpose-built and "
     "calibrated for a single GB10 box (98.9GB on disk) — the full un-pruned model needs 2x Spark.", [
        ("saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10", "172B total/10B active MoE, NVFP4"),
     ]),
]

CHECKED_AND_SKIPPED = [
    ("thinkingmachines/Inkling", "975B total / 41B active (MoE)",
     "Too large even at NVFP4 (~490GB of weights alone) — needs multiple nodes, not a single GB10 box."),
    ("prism-ml/Ternary-Bonsai", "1.58-bit ternary weights, 1.7B-27B",
     "Only published as GGUF/MLX — no vLLM-loadable format exists yet."),
    ("DeepSeek V4-Pro / V4-Flash (2026)", "1.6T / 284B total",
     "Smallest 2026 DeepSeek release is still 284B total — community recipes confirm this needs "
     "2x DGX Spark nodes, not one."),
    ("MiniMax M2.7 (full, un-pruned)", "230B total / 10B active (MoE)",
     "~130GB at NVFP4 — confirmed 2x-Spark-node deployments only; use the REAP-pruned single-node "
     "variant queued above instead."),
    ("GPT-OSS-20B / GPT-OSS-120B", "20B / 120B, MXFP4",
     "Requested, but released 2025-08-05 — dropped per this project's 2026-or-later rule for new "
     "candidates."),
    ("Phi-4-multimodal-instruct / Phi-4-reasoning-plus", "—",
     "Requested, but released 2025-02 / 2025-04-30 — dropped per this project's 2026-or-later "
     "rule for new candidates."),
]

OFFICIAL_PICKS = [
    ("Nemotron 3 Nano (30B-A3B, NVFP4)", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
     "NVIDIA's primary reference model for DGX Spark — the official playbook's main example."),
    ("Nemotron 3 Super (120B-A12B, NVFP4)", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
     "NVIDIA's flagship open agentic model, trained natively at NVFP4; the largest model that "
     "realistically fits a single 128GB GB10 box."),
]

# Benchmark *types* on the roadmap, distinct from which models are queued -- new kinds of
# measurement this project doesn't do yet at all.
FUTURE_BENCHMARKS = [
    ("Fine-tuning benchmarks",
     "This box will also be used to benchmark fine-tuning workloads, not just inference: LoRA "
     "vs. full fine-tune training throughput, memory used during training at different model "
     "sizes/batch sizes, and how that trades off against the inference numbers already on this "
     "leaderboard. Not built yet — planned."),
    ("Context-handling / memory-management technique comparison",
     "The `speed` context probe reports the largest context a model handled, but not *how* "
     "different architectures manage that memory — KV-cache quantization, prefix-cache reuse "
     "across conversation turns, Mamba/hybrid-attention models that scale near-linearly with "
     "context vs. full-attention models that don't. Matters specifically for a multi-turn "
     "agentic harness like Hermes. Planned, not built yet."),
    ("Context × concurrency permutation testing",
     "Right now context size and concurrent-session capacity are measured separately (a "
     "single-stream context probe, and a fixed-small-context concurrency sweep). Testing the "
     "actual combination — e.g. how many concurrent 32K-context sessions this box sustains, not "
     "just how many concurrent short-prompt sessions — is on the list."),
]


def _match_family(model_name):
    n = (model_name or "").lower()
    for key, title, note in FAMILY_NOTES:
        if key in n:
            return title, note
    return None, None


def _model_specialities(entry, all_entries):
    wins = []
    for title, desc, eligible, sort_key, metric_key, fmt_str in lb.CATEGORIES:
        winner = lb.best_for_category(all_entries, (title, desc, eligible, sort_key, metric_key, fmt_str))
        if winner is entry:
            wins.append(title)
    strong_domains = sorted(
        ((d, v) for d, v in entry.get("domain_quality", {}).items() if v >= 60),
        key=lambda dv: -dv[1]
    )[:3]
    return wins, strong_domains


def render(entries):
    tested_models = {e["model"] for e in entries if e.get("model")}

    lines = [
        "# DGX Spark Model Wiki\n",
        "_by the Nokast community_\n",
        "A plain-language guide to which models are actually worth running on a single NVIDIA "
        "DGX Spark (GB10, 128GB unified memory) — what each one is good at, what's still being "
        "tested, and what's coming next. Not a score table on its own — see "
        "[`results/LEADERBOARD.html`](LEADERBOARD.html) for the full numbers this page's "
        "writeups are drawn from, and the repo [README](../README.md) / "
        "[METHODOLOGY.md](../METHODOLOGY.md) for how they're measured.\n",
        "## Want a model tested?\n",
        f"Open an issue on [the repo]({REPO_URL}/issues) with the Hugging Face repo ID and why "
        "you want it — or send a PR adding it to the `QUEUED_GROUPS` list in `model_wiki.py`, "
        "which is what actually drives the \"coming soon\" section below. A few things make a "
        "request more likely to get run soon: it should be loadable in vLLM (this project "
        "targets vLLM specifically — GGUF-only releases and SGLang-specific ones aren't "
        "supported yet), and it should actually fit a single box's 128GB unified memory at "
        "whatever quantization you're asking for. Older (pre-2026) models are lower priority "
        "but not refused outright — see the \"checked and skipped\" section for the current "
        "cutoff reasoning.\n",
        "## Benchmarked on this box\n",
    ]

    tested = [e for e in entries if e.get("model") and e.get("status") == "complete"]
    tested.sort(key=lambda e: -(e.get("size_b") or 0))
    if not tested:
        lines.append("_No models fully benchmarked yet._\n")
    for e in tested:
        title, note = _match_family(e["model"])
        size = f"{e['size_b']:.0f}B" if e.get("size_b") else "unknown size"
        wins, strong_domains = _model_specialities(e, entries)
        lines.append(f"### {e['model']}\n")
        lines.append(f"- **Size / format:** {size}, {e.get('precision', '—')}")
        lines.append(f"- **Official recommendation status:** {note or 'No specific official GB10 endorsement found for this exact repo — same architecture family as models that do get called out, but not confirmed itself.'}")
        if wins:
            lines.append(f"- **Best model for:** " + "; ".join(wins))
        if strong_domains:
            domain_s = ", ".join(f"{d} ({v:.0f}/100)" for d, v in strong_domains)
            lines.append(f"- **Strongest graded domains:** {domain_s}")
        lines.append(f"- **LocalScore / Hermes Score:** {lb.fmt(e['localscore'])} / {lb.fmt(e['hermes_hermes_score'])}")
        lines.append(f"- **Biggest document it handled:** {lb._fmt_context(e.get('max_context_ok'))}")
        lines.append(f"- **Best-case speed:** {lb.fmt(e.get('peak_decode_tps'), 0)} tokens/sec" if e.get('peak_decode_tps') else "- **Best-case speed:** not yet measured")
        max_conc = e.get("max_concurrent")
        lines.append(f"- **Max concurrent requests tested:** {max_conc if max_conc else '—'}")
        hermes_conc = e.get("ceilings", {}).get("hermes")
        if hermes_conc:
            lines.append(f"- **Hermes-shaped traffic (tool chains, web search, long-context recall) "
                         f"tested up to:** {hermes_conc} concurrent sessions")
        if e.get("load_time_s") is not None:
            lines.append(f"- **Load time:** {e['load_time_s']:.0f}s")
        if e.get("peak_temp_c") is not None:
            lines.append(f"- **Under load:** peak {e['peak_gpu_util_pct']:.0f}% GPU utilization, "
                         f"{e['peak_temp_c']:.0f}°C, {e['peak_power_w']:.0f}W")
        tc = e["tier3_pass"].get("tool_call")
        tc_s = "works" if tc == 1.0 else ("does not work reliably" if tc == 0.0 else "not yet tested")
        lines.append(f"- **Tool calling:** {tc_s}")
        lines.append("")

    in_progress = [e for e in entries if e.get("model") and e.get("status") != "complete"]
    if in_progress:
        lines.append("### Partial / legacy results (full re-test queued)\n")
        for e in in_progress:
            lines.append(f"- `{e['model']}` (label `{e['label']}`) — only has a subset of numbers "
                         f"from before the current benchmark pipeline; treat as stale.")
        lines.append("")

    lines.append("## Coming soon — queued, not yet benchmarked\n")
    lines.append("These are on the list for a future run but don't have real numbers yet — no "
                 "scores are shown for these on purpose. This list updates itself as batches "
                 "complete (filtered against what's actually been tested), not hand-maintained.\n")
    any_queued = False
    for title, blurb, repos in QUEUED_GROUPS:
        remaining = [(repo, note) for repo, note in repos if repo not in tested_models]
        if not remaining:
            continue
        any_queued = True
        lines.append(f"**{title}** — {blurb}\n")
        for repo, note in remaining:
            note_s = f" ({note})" if note else ""
            lines.append(f"- `{repo}`{note_s}")
        lines.append("")
    if not any_queued:
        lines.append("_Everything queued has been tested — see above, or open an issue to add more._\n")

    lines.append("## Coming soon — new kinds of benchmarks, not just new models\n")
    lines.append("Beyond just testing more models with the existing pipeline, these are new "
                 "*kinds* of measurement planned for this project:\n")
    for title, desc in FUTURE_BENCHMARKS:
        lines.append(f"- **{title}** — {desc}")
    lines.append("")

    lines.append("## NVIDIA's own official picks for DGX Spark / GB10\n")
    lines.append("For context — NVIDIA's own shortlist, independent of what this project has "
                 "gotten around to testing:\n")
    for name, repo, note in OFFICIAL_PICKS:
        repo_s = f" (`{repo}`)" if repo != "—" else ""
        lines.append(f"- **{name}**{repo_s} — {note}")
    lines.append("")

    lines.append("## Checked and skipped\n")
    lines.append("Trending or requested models that turned out not to fit this hardware, or "
                 "aren't loadable in vLLM (this project targets vLLM for now — SGLang support is "
                 "future work, and GGUF-only releases are out of scope):\n")
    for name, size, note in CHECKED_AND_SKIPPED:
        lines.append(f"- **{name}** ({size}) — {note}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Build a plain-language model wiki from benchmark results")
    ap.add_argument("--csv", default=lb.DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    entries = []
    if os.path.exists(args.csv):
        entries = lb.build_entries(lb.load_rows(args.csv))

    text = render(entries)
    print(text)

    if args.write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\n-> wrote {args.out}")


if __name__ == "__main__":
    main()
