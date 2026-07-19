#!/usr/bin/env python3
"""
model_wiki.py — build results/MODEL_WIKI.md: a plain-language guide to which models make sense
to run on an NVIDIA DGX Spark (GB10, 128GB unified memory), not just a score table.

Two kinds of entries:
  tested   models this box has actually benchmarked (results/spark_bench_plus.csv) — real
           numbers pulled in via leaderboard.py, plus a curated note on whether it's one of the
           handful of models NVIDIA/Hugging Face/Unsloth specifically calls out for GB10, or
           just a community/self-published quant of a good family.
  queued   models worth trying that haven't been benchmarked here yet — "coming soon", no
           fabricated numbers, just why they're on the list.

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

# Queued: real repo IDs, confirmed to exist via web research on 2026-07-18/19, not yet run
# through bench-all on this box. Kept here (not just in ops/models.txt) because this file is
# public and ops/ is gitignored — the roadmap itself is fine to share even though the container
# internals aren't. Grouped to match ops/models.txt.
QUEUED_GROUPS = [
    ("Re-testing with the fixed pipeline",
     "The rest of the original 12-model batch, queued to re-run now that the max-num-seqs "
     "scaling, Gemma4 tool-calling, and reasoning-model token-budget fixes are in.", [
        ("nvidia/Qwen3.6-27B-NVFP4", ""), ("unsloth/Qwen3.6-27B-NVFP4", ""),
        ("unsloth/Qwen3.6-35B-A3B-NVFP4", ""), ("nvidia/Gemma-4-31B-IT-NVFP4", ""),
        ("unsloth/gemma-4-31B-it-NVFP4", ""), ("unsloth/gemma-4-26B-A4B-it-NVFP4", ""),
        ("Qwen/Qwen3.6-27B", "BF16"), ("Qwen/Qwen3.6-35B-A3B", "BF16"),
        ("google/gemma-4-31B-it", "BF16"), ("google/gemma-4-26B-A4B-it", "BF16"),
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
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "same model as the active NVFP4 one, uncompressed"),
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


def _match_family(model_name):
    n = (model_name or "").lower()
    for key, title, note in FAMILY_NOTES:
        if key in n:
            return title, note
    return None, None


def render(entries):
    lines = [
        "# DGX Spark Model Wiki\n",
        "_by the Nokast community_\n",
        "Not a score table — a plain-language guide to which models are actually worth running "
        "on a single NVIDIA DGX Spark (GB10, 128GB unified memory), what each one is good at, "
        "and whether it's one of the models NVIDIA/Hugging Face/Unsloth specifically call out "
        "for this hardware versus just a good model that happens to fit.\n",
        "See [`results/LEADERBOARD.html`](LEADERBOARD.html) for the full score table this data "
        "is drawn from, and the repo [README](../README.md) for how the numbers are measured.\n",
        "## Benchmarked on this box\n",
    ]

    tested = [e for e in entries if e.get("model")]
    tested.sort(key=lambda e: -(e.get("size_b") or 0))
    if not tested:
        lines.append("_No models benchmarked yet._\n")
    for e in tested:
        title, note = _match_family(e["model"])
        size = f"{e['size_b']:.0f}B" if e.get("size_b") else "unknown size"
        overall = e["hermes_hermes_score"] if e["hermes_hermes_score"] is not None else e["localscore"]
        lines.append(f"### {e['model']}\n")
        lines.append(f"- **Size / format:** {size}, {e.get('precision', '—')}")
        lines.append(f"- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`")
        lines.append(f"- **Official recommendation status:** {note or 'No specific official GB10 endorsement found for this exact repo — same architecture family as models that do get called out, but not confirmed itself.'}")
        lines.append(f"- **Overall score:** {lb.fmt(overall)}/100" if overall is not None else "- **Overall score:** not yet scored")
        lines.append(f"- **Biggest document it handled:** {lb._fmt_context(e.get('max_context_ok'))}")
        lines.append(f"- **Best-case speed:** {lb.fmt(e.get('peak_decode_tps'), 0)} tokens/sec" if e.get('peak_decode_tps') else "- **Best-case speed:** not yet measured")
        lines.append(f"- **Max concurrent sessions sustained:** {e.get('max_concurrent') or '—'}")
        tc = e["tier3_pass"].get("tool_call")
        tc_s = "works" if tc == 1.0 else ("does not work reliably" if tc == 0.0 else "not yet tested")
        lines.append(f"- **Tool calling:** {tc_s}")
        lines.append("")

    lines.append("## Coming soon — queued, not yet benchmarked\n")
    lines.append("These are on the list for a future run but don't have real numbers yet — no "
                 "scores are shown for these on purpose.\n")
    for title, blurb, repos in QUEUED_GROUPS:
        lines.append(f"**{title}** — {blurb}\n")
        for repo, note in repos:
            note_s = f" ({note})" if note else ""
            lines.append(f"- `{repo}`{note_s}")
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
