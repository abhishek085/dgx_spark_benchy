# DGX Spark Model Wiki

_by the Nokast community_

Not a score table — a plain-language guide to which models are actually worth running on a single NVIDIA DGX Spark (GB10, 128GB unified memory), what each one is good at, and whether it's one of the models NVIDIA/Hugging Face/Unsloth specifically call out for this hardware versus just a good model that happens to fit.

See [`results/LEADERBOARD.html`](LEADERBOARD.html) for the full score table this data is drawn from, and the repo [README](../README.md) for how the numbers are measured.

## Benchmarked on this box

### nvidia/Qwen3.6-35B-A3B-NVFP4

- **Size / format:** 35B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Third-party/self-published NVFP4 quants of Qwen's own model — not on NVIDIA's short list of DGX Spark reference models, but a strong MoE agentic-coding model in its own right.
- **Overall score:** 69.0/100
- **Biggest document it handled:** 15K tokens
- **Best-case speed:** 62 tokens/sec
- **Max concurrent sessions sustained:** 2
- **Tool calling:** works

### unsloth/Qwen3.6-35B-A3B-NVFP4

- **Size / format:** 35B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Third-party/self-published NVFP4 quants of Qwen's own model — not on NVIDIA's short list of DGX Spark reference models, but a strong MoE agentic-coding model in its own right.
- **Overall score:** 31.2/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 8
- **Tool calling:** works

### deepreinforce-ai/Ornith-1.0-35B

- **Size / format:** 35B, BF16 (full size)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Not an NVIDIA/HF/Unsloth official GB10 pick — a dense agentic-coding model built on Qwen 3.5 by deepreinforce-ai, tuned for terminal-based coding agents. Added on request.
- **Overall score:** 29.0/100
- **Biggest document it handled:** 15K tokens
- **Best-case speed:** 30 tokens/sec
- **Max concurrent sessions sustained:** 4
- **Tool calling:** does not work reliably

### nvidia/Gemma-4-31B-IT-NVFP4

- **Size / format:** 31B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Google's larger dense Gemma 4 — not MoE, so heavier per-token, but strong general quality.
- **Overall score:** 48.0/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 2
- **Tool calling:** does not work reliably

### unsloth/gemma-4-31B-it-NVFP4

- **Size / format:** 31B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Google's larger dense Gemma 4 — not MoE, so heavier per-token, but strong general quality.
- **Overall score:** 53.3/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 4
- **Tool calling:** does not work reliably

### google/gemma-4-31B-it

- **Size / format:** 31B, BF16 (full size)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Google's larger dense Gemma 4 — not MoE, so heavier per-token, but strong general quality.
- **Overall score:** 53.3/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 4
- **Tool calling:** does not work reliably

### nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4

- **Size / format:** 30B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** NVIDIA's own official small reference model for DGX Spark — the primary example in NVIDIA's own DGX Spark playbook.
- **Overall score:** 29.0/100
- **Biggest document it handled:** 66K tokens
- **Best-case speed:** 61 tokens/sec
- **Max concurrent sessions sustained:** 4
- **Tool calling:** does not work reliably

### nvidia/Qwen3.6-27B-NVFP4

- **Size / format:** 27B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) run well here.
- **Overall score:** 46.8/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 128
- **Tool calling:** works

### unsloth/Qwen3.6-27B-NVFP4

- **Size / format:** 27B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) run well here.
- **Overall score:** 28.3/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 4
- **Tool calling:** works

### nvidia/Qwen3.6-27B-NVFP4

- **Size / format:** 27B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) run well here.
- **Overall score:** 66.7/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 32
- **Tool calling:** works

### unsloth/Qwen3.6-27B-NVFP4

- **Size / format:** 27B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) run well here.
- **Overall score:** 44.5/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** —
- **Tool calling:** works

### Qwen/Qwen3.6-27B

- **Size / format:** 27B, BF16 (full size)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) run well here.
- **Overall score:** 22.9/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 4
- **Tool calling:** works

### nvidia/Gemma-4-26B-A4B-NVFP4

- **Size / format:** 26B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Called out repeatedly as a popular community choice specifically for GB10 — a good efficiency/quality balance for a MoE model in this size class.
- **Overall score:** 75.2/100
- **Biggest document it handled:** 16K tokens
- **Best-case speed:** 30 tokens/sec
- **Max concurrent sessions sustained:** 2
- **Tool calling:** works

### unsloth/gemma-4-26B-A4B-it-NVFP4

- **Size / format:** 26B, NVFP4 (4-bit)
- **Fits on this box:** yes — benchmarked live at `--gpu-memory-utilization 0.85`
- **Official recommendation status:** Called out repeatedly as a popular community choice specifically for GB10 — a good efficiency/quality balance for a MoE model in this size class.
- **Overall score:** 53.3/100
- **Biggest document it handled:** —
- **Best-case speed:** not yet measured
- **Max concurrent sessions sustained:** 4
- **Tool calling:** does not work reliably

## Coming soon — queued, not yet benchmarked

These are on the list for a future run but don't have real numbers yet — no scores are shown for these on purpose.

**Re-testing with the fixed pipeline** — The rest of the original 12-model batch, queued to re-run now that the max-num-seqs scaling, Gemma4 tool-calling, and reasoning-model token-budget fixes are in.

- `nvidia/Qwen3.6-27B-NVFP4`
- `unsloth/Qwen3.6-27B-NVFP4`
- `unsloth/Qwen3.6-35B-A3B-NVFP4`
- `nvidia/Gemma-4-31B-IT-NVFP4`
- `unsloth/gemma-4-31B-it-NVFP4`
- `unsloth/gemma-4-26B-A4B-it-NVFP4`
- `Qwen/Qwen3.6-27B` (BF16)
- `Qwen/Qwen3.6-35B-A3B` (BF16)
- `google/gemma-4-31B-it` (BF16)
- `google/gemma-4-26B-A4B-it` (BF16)

**Smallest meaningful** — Gemma 4's on-device tier (2026-04) — is there any real task this box can run near-instantly with tiny memory use?

- `google/gemma-4-E2B-it` (~2.3B effective, BF16)
- `google/gemma-4-E4B-it` (~4.5B effective, BF16)

**NVIDIA's other official DGX Spark models** — The rest of NVIDIA's own Nemotron 3 lineup (2026-03).

- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (120B total/12B active MoE, NVFP4)
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (same model as the active NVFP4 one, uncompressed)
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`

**Nemotron 3 Nano Omni** — Multimodal (video/audio/image/text) Mamba2-Transformer hybrid MoE, reasoning-tuned, 2026 — 3 precisions to compare.

- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`

**DiffusionGemma 26B-A4B-it** — Same Gemma 4 MoE backbone, but block-diffusion text generation instead of token-by-token autoregression — a genuinely different generation architecture, 2026.

- `google/diffusiongemma-26B-A4B-it` (BF16)
- `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (NVFP4)

**Biggest that fits** — A community 25%-REAP-pruned NVFP4 quant of MiniMax M2.7 (2026-03), purpose-built and calibrated for a single GB10 box (98.9GB on disk) — the full un-pruned model needs 2x Spark.

- `saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10` (172B total/10B active MoE, NVFP4)

## NVIDIA's own official picks for DGX Spark / GB10

For context — NVIDIA's own shortlist, independent of what this project has gotten around to testing:

- **Nemotron 3 Nano (30B-A3B, NVFP4)** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`) — NVIDIA's primary reference model for DGX Spark — the official playbook's main example.
- **Nemotron 3 Super (120B-A12B, NVFP4)** (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) — NVIDIA's flagship open agentic model, trained natively at NVFP4; the largest model that realistically fits a single 128GB GB10 box.

## Checked and skipped

Trending or requested models that turned out not to fit this hardware, or aren't loadable in vLLM (this project targets vLLM for now — SGLang support is future work, and GGUF-only releases are out of scope):

- **thinkingmachines/Inkling** (975B total / 41B active (MoE)) — Too large even at NVFP4 (~490GB of weights alone) — needs multiple nodes, not a single GB10 box.
- **prism-ml/Ternary-Bonsai** (1.58-bit ternary weights, 1.7B-27B) — Only published as GGUF/MLX — no vLLM-loadable format exists yet.
- **DeepSeek V4-Pro / V4-Flash (2026)** (1.6T / 284B total) — Smallest 2026 DeepSeek release is still 284B total — community recipes confirm this needs 2x DGX Spark nodes, not one.
- **MiniMax M2.7 (full, un-pruned)** (230B total / 10B active (MoE)) — ~130GB at NVFP4 — confirmed 2x-Spark-node deployments only; use the REAP-pruned single-node variant queued above instead.
- **GPT-OSS-20B / GPT-OSS-120B** (20B / 120B, MXFP4) — Requested, but released 2025-08-05 — dropped per this project's 2026-or-later rule for new candidates.
- **Phi-4-multimodal-instruct / Phi-4-reasoning-plus** (—) — Requested, but released 2025-02 / 2025-04-30 — dropped per this project's 2026-or-later rule for new candidates.

