# DGX Spark Model Wiki

_by the Nokast community_

A plain-language guide to which models are actually worth running on a single NVIDIA DGX Spark (GB10, 128GB unified memory) — what each one is good at, what's still being tested, and what's coming next. Not a score table on its own — see [`results/LEADERBOARD.html`](LEADERBOARD.html) for the full numbers this page's writeups are drawn from, and the repo [README](../README.md) / [METHODOLOGY.md](../METHODOLOGY.md) for how they're measured.

## Want a model tested?

Open an issue on [the repo](https://github.com/abhishek085/dgx_spark_benchy/issues) with the Hugging Face repo ID and why you want it — or send a PR adding it to the `QUEUED_GROUPS` list in `model_wiki.py`, which is what actually drives the "coming soon" section below. A few things make a request more likely to get run soon: it should be loadable in vLLM (this project targets vLLM specifically — GGUF-only releases and SGLang-specific ones aren't supported yet), and it should actually fit a single box's 128GB unified memory at whatever quantization you're asking for. Older (pre-2026) models are lower priority but not refused outright — see the "checked and skipped" section for the current cutoff reasoning.

## Benchmarked on this box

### nvidia/Qwen3.6-35B-A3B-NVFP4

- **Size / format:** 35B, NVFP4 (4-bit)
- **Official recommendation status:** Third-party/self-published NVFP4 quants of Qwen's own model — not on NVIDIA's short list of DGX Spark reference models, but a strong MoE agentic-coding model in its own right.
- **Best model for:** Best at coding; Best at tool calling
- **Strongest graded domains:** restraint (100/100), tool_use (83/100), long_context (75/100)
- **LocalScore / Hermes Score:** 61.7 / 69.0
- **Biggest document it handled:** 124K tokens
- **Best-case speed:** 59 tokens/sec
- **Max concurrent requests tested:** 32
- **Hermes-shaped traffic (tool chains, web search, long-context recall) tested up to:** 256 concurrent sessions
- **Tool calling:** works

### unsloth/Qwen3.6-35B-A3B-NVFP4

- **Size / format:** 35B, NVFP4 (4-bit)
- **Official recommendation status:** Third-party/self-published NVFP4 quants of Qwen's own model — not on NVIDIA's short list of DGX Spark reference models, but a strong MoE agentic-coding model in its own right.
- **Strongest graded domains:** restraint (100/100), long_context (75/100)
- **LocalScore / Hermes Score:** 60.4 / 69.0
- **Biggest document it handled:** 124K tokens
- **Best-case speed:** 58 tokens/sec
- **Max concurrent requests tested:** 32
- **Hermes-shaped traffic (tool chains, web search, long-context recall) tested up to:** 256 concurrent sessions
- **Load time:** 202s
- **Under load:** peak 96% GPU utilization, 74°C, 78W
- **Tool calling:** works

### nvidia/Gemma-4-31B-IT-NVFP4

- **Size / format:** 31B, NVFP4 (4-bit)
- **Official recommendation status:** Google's larger dense Gemma 4 — not MoE, so heavier per-token, but strong general quality.
- **Strongest graded domains:** long_context (100/100), restraint (100/100), robustness (100/100)
- **LocalScore / Hermes Score:** 61.3 / 71.7
- **Biggest document it handled:** 64K tokens
- **Best-case speed:** 7 tokens/sec
- **Max concurrent requests tested:** 32
- **Hermes-shaped traffic (tool chains, web search, long-context recall) tested up to:** 256 concurrent sessions
- **Load time:** 715s
- **Under load:** peak 96% GPU utilization, 77°C, 79W
- **Tool calling:** works

### nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4

- **Size / format:** 30B, NVFP4 (4-bit)
- **Official recommendation status:** NVIDIA's own official small reference model for DGX Spark — the primary example in NVIDIA's own DGX Spark playbook.
- **Strongest graded domains:** long_context (100/100), restraint (100/100), safety (75/100)
- **LocalScore / Hermes Score:** 64.7 / 29.0
- **Biggest document it handled:** 133K tokens
- **Best-case speed:** 61 tokens/sec
- **Max concurrent requests tested:** 32
- **Hermes-shaped traffic (tool chains, web search, long-context recall) tested up to:** 1 concurrent sessions
- **Load time:** 262s
- **Under load:** peak 96% GPU utilization, 81°C, 88W
- **Tool calling:** does not work reliably

### nvidia/Qwen3.6-27B-NVFP4

- **Size / format:** 27B, NVFP4 (4-bit)
- **Official recommendation status:** Same story as the 35B-A3B: a good dense agentic-coding model, not one of NVIDIA's official GB10 picks specifically, but published NVFP4 quants (including one under the nvidia/ org) run well here.
- **Strongest graded domains:** long_context (100/100), restraint (100/100), robustness (75/100)
- **LocalScore / Hermes Score:** 47.6 / 68.9
- **Biggest document it handled:** 124K tokens
- **Best-case speed:** 10 tokens/sec
- **Max concurrent requests tested:** 32
- **Hermes-shaped traffic (tool chains, web search, long-context recall) tested up to:** 256 concurrent sessions
- **Load time:** 272s
- **Under load:** peak 96% GPU utilization, 82°C, 91W
- **Tool calling:** works

### nvidia/Gemma-4-26B-A4B-NVFP4

- **Size / format:** 26B, NVFP4 (4-bit)
- **Official recommendation status:** Called out repeatedly as a popular community choice specifically for GB10 — a good efficiency/quality balance for a MoE model in this size class.
- **Best model for:** Fastest (with real answers, not just fast garbage); Most accurate (that isn't painfully slow); Best for a Hermes-style personal agent
- **Strongest graded domains:** instruction (100/100), long_context (100/100), restraint (100/100)
- **LocalScore / Hermes Score:** 73.9 / 75.2
- **Biggest document it handled:** 129K tokens
- **Best-case speed:** 30 tokens/sec
- **Max concurrent requests tested:** 32
- **Hermes-shaped traffic (tool chains, web search, long-context recall) tested up to:** 256 concurrent sessions
- **Load time:** 342s
- **Under load:** peak 96% GPU utilization, 74°C, 64W
- **Tool calling:** works

## Coming soon — queued, not yet benchmarked

These are on the list for a future run but don't have real numbers yet — no scores are shown for these on purpose. This list updates itself as batches complete (filtered against what's actually been tested), not hand-maintained.

**Re-testing with the fixed pipeline** — The rest of the original 12-model batch, queued to re-run now that the max-num-seqs scaling, Gemma4 tool-calling, context-window detection, and speed-aware scoring fixes are in.

- `unsloth/Qwen3.6-27B-NVFP4`
- `unsloth/gemma-4-31B-it-NVFP4`
- `unsloth/gemma-4-26B-A4B-it-NVFP4`
- `Qwen/Qwen3.6-27B` (BF16)
- `Qwen/Qwen3.6-35B-A3B` (BF16)
- `google/gemma-4-31B-it` (BF16)
- `google/gemma-4-26B-A4B-it` (BF16)
- `deepreinforce-ai/Ornith-1.0-35B`

**Smallest meaningful** — Gemma 4's on-device tier (2026-04) — is there any real task this box can run near-instantly with tiny memory use?

- `google/gemma-4-E2B-it` (~2.3B effective, BF16)
- `google/gemma-4-E4B-it` (~4.5B effective, BF16)

**NVIDIA's other official DGX Spark models** — The rest of NVIDIA's own Nemotron 3 lineup (2026-03).

- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (120B total/12B active MoE, NVFP4)
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (same model, uncompressed)
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

## Coming soon — new kinds of benchmarks, not just new models

Beyond just testing more models with the existing pipeline, these are new *kinds* of measurement planned for this project:

- **Fine-tuning benchmarks** — This box will also be used to benchmark fine-tuning workloads, not just inference: LoRA vs. full fine-tune training throughput, memory used during training at different model sizes/batch sizes, and how that trades off against the inference numbers already on this leaderboard. Not built yet — planned.
- **Context-handling / memory-management technique comparison** — The `speed` context probe reports the largest context a model handled, but not *how* different architectures manage that memory — KV-cache quantization, prefix-cache reuse across conversation turns, Mamba/hybrid-attention models that scale near-linearly with context vs. full-attention models that don't. Matters specifically for a multi-turn agentic harness like Hermes. Planned, not built yet.
- **Context × concurrency permutation testing** — Right now context size and concurrent-session capacity are measured separately (a single-stream context probe, and a fixed-small-context concurrency sweep). Testing the actual combination — e.g. how many concurrent 32K-context sessions this box sustains, not just how many concurrent short-prompt sessions — is on the list.

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

