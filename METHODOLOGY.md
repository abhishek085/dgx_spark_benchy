# How this benchmark actually works

_by the Nokast community_

This page exists so you don't have to trust the leaderboard numbers on faith — every one of them
comes from a real HTTP request sent to a real, running model server on one physical machine (an
NVIDIA DGX Spark, GB10 chip, 128GB unified memory), timed and graded by the code in this repo.
Nothing here is a vendor-reported number or a simulation. This page walks through each benchmark
stage, what it actually sends, and what a real run looks like.

**Scope:** this whole project benchmarks a *single* DGX Spark box. If you're clustering multiple
Sparks together, the `interconnect` command below is the one piece that's relevant to you —
everything else here is single-machine, single-GPU.

## The six things we measure

Each one is a `spark_bench_plus.py` subcommand. They can be run standalone against any
OpenAI-compatible server (vLLM, llama.cpp, etc.), or chained automatically per-model (see
"How the leaderboard gets built" below).

### 1. `speed` — raw inference speed

**What it answers:** "If I send this model one request, how fast does it respond, and does that
change as the prompt gets longer?"

**What it actually does:** builds a synthetic prompt of a target length (e.g. ~4,096 tokens,
~16,384 tokens...), sends it, and streams the response token-by-token, measuring:
- **TTFT** (time to first token) — how long you wait before anything appears
- **decode tok/s** — how fast words come out once it starts
- **TPOT** (time per output token) — the inverse of decode speed, in milliseconds
- **prefill tok/s** — how fast it *reads* the prompt before answering

```bash
./spark_bench_plus.py speed --label my-model \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \
  --contexts 4096,16384,32768
```

```
| context | TTFT (ms) | decode tok/s | TPOT (ms) | prefill tok/s | out toks |
|--------:|----------:|-------------:|----------:|--------------:|---------:|
|    4096 |       210 |         48.2 |      20.7 |         19500 |      256 |
|   16384 |       740 |         44.1 |      22.7 |         22100 |      256 |
|   32768 |      1510 |         41.8 |      23.9 |         21700 |      256 |
```
*(illustrative shape — your numbers depend on the model, quantization, and box)*

In the automated per-model loop this also drives the **"biggest document it handled"** number on
the leaderboard: we probe increasing context sizes (4K, 16K, 32K, 64K, 128K, capped by what the
model was actually served with) and record the largest one that came back without an error —
that's a real, measured ceiling, not the model's advertised max.

### 2. `checks` — does it actually do the basics

**What it answers:** "Before I spend an hour benchmarking this model, does it even work?"

Three quick real checks: write a Python function and *execute* the code it returns against test
cases (not just "does it look like code"), make it call a tool and verify the tool call actually
parses, and hide a passcode in a long block of text and see if it can find it. Fast, single-shot,
meant as a sanity gate.

```bash
./spark_bench_plus.py checks --label my-model \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1
```

### 3. `eval` — general quality, graded

**What it answers:** "How good are its answers, really — and is that consistent, or does it get
lucky sometimes?"

22 scenarios across 10 domains (tool use, structured output, robustness to bad input, safety,
knowing when *not* to act, multi-step tasks, following exact instructions, autonomous planning,
long-context, visual) — each graded with **partial credit** (a model that gets 2 of 3 required
tool calls right scores 0.66, not 0), and each scenario repeated `--repeats` times so we can tell
a genuinely reliable model from one that happened to nail it once.

```bash
./spark_bench_plus.py eval --label my-model \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 --repeats 3
```

Rolls up into **LocalScore** (0-100): a blend of Quality (did it get it right), Reliability
(consistent across repeats), Efficiency (accuracy per token spent), and Responsiveness (how fast
it started answering).

### 4. `capacity` — how many people can use it at once

**What it answers:** "Not 'how fast is one request' — how many *simultaneous* conversations can
this box actually sustain before things degrade?"

Sends increasing numbers of concurrent conversations (1, 2, 4, 8, 16, 32...) using realistic
multi-turn workload shapes (see `agent_profiles.py`: `orchestrator` = tool-chain agent traffic,
`coding_agent` = code-gen requests, `chat_agent` = casual conversation), and watches for the point
where error rate climbs past 10% or accuracy drops below 70% of the single-user baseline. That
point is the **ceiling** — the real, load-tested answer, not a theoretical one.

```bash
./spark_bench_plus.py capacity --label my-model \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \
  --skip-restart --profiles orchestrator,coding_agent,chat_agent \
  --concurrency 1,2,4,8,16,32
```

`--concurrency auto` instead escalates (doubling each round) until it actually finds the ceiling,
rather than requiring you to guess a wide-enough list — a small model and a huge one don't break
at the same point, and a fixed list either wastes rounds on a weak model or stops too early on a
strong one.

### 5. `hermes` — the composite score for a personal-agent harness

**What it answers:** "If I put this model behind a Hermes-style personal agent — the kind that
uses tools, searches the web, remembers things across a conversation, and needs to serve more
than one session — how good is it, overall?"

Runs the `hermes` workload profile (tool chains with `web_search`/`read_file`, long-context recall
*inside* a multi-turn session, preference tracking across turns) through both a quality pass and
an auto-escalating capacity sweep, then combines them:

**Hermes Score = 50% task quality + 30% capacity ceiling + 20% responsiveness**

```bash
./spark_bench_plus.py hermes --label my-model \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \
  --concurrency auto --target-ceiling 16 --target-ttft-ms 1500
```

### 6. `interconnect` — multi-node bandwidth (skip this if you have one box)

Measures RoCE bandwidth (`ib_write_bw`/`ib_read_bw`) between Spark nodes. Only matters if you're
clustering multiple Sparks together; it's a no-op on a single-box setup like this one.

## How the leaderboard gets built

For every model on the leaderboard, all six of the above (minus `interconnect`) ran in sequence
against a live server on this exact box: `eval` → `capacity` → `hermes` → `speed` (context probe)
→ `checks`, then the results get folded into one row on
[`results/LEADERBOARD.html`](results/LEADERBOARD.html). See
[`results/MODEL_WIKI.md`](results/MODEL_WIKI.md) for what each benchmarked model is actually good
at and whether it's one of the models NVIDIA/Hugging Face/Unsloth specifically recommend for this
hardware.

## What's not measured yet

- **Context-handling and memory-management technique comparison.** The `speed` context probe
  above tells you the *largest* context a model handled, but not *how* — different architectures
  manage that memory very differently (KV-cache quantization, prefix-cache reuse across turns,
  Mamba/hybrid-attention models that scale near-linearly with context vs. full-attention models
  that don't), and that matters specifically for a multi-turn agentic harness like Hermes, where
  the same long conversation gets re-processed turn after turn. A dedicated benchmark comparing
  how much that helps or hurts different architectures under real multi-turn agentic load is
  planned but not built yet.
