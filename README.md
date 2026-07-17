<p align="center">
  <img src="assets/logo.svg" width="120" alt="dgx_spark_benchy logo" />
</p>

<h1 align="center">dgx_spark_benchy</h1>

<p align="center">
  A benchmark harness for local LLM deployments that answers two questions spark-bench doesn't:
  <b>how does this model behave under an agent harness</b>, and
  <b>how many concurrent agent sessions can this box actually sustain</b> before it falls over?
</p>

<p align="center">
  <a href="#"><img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="#"><img alt="deps" src="https://img.shields.io/badge/dependencies-stdlib%20only-brightgreen"></a>
  <a href="#"><img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
  <a href="#"><img alt="status" src="https://img.shields.io/badge/status-active-success"></a>
</p>

---

## Why this exists

[spark-bench](https://github.com/Weschera/spark-bench) is a great mixed-capability benchmark for local hardware (built for the NVIDIA DGX Spark / GB10) — raw inference speed, a graded scenario suite, partial-credit grading, trial statistics. What it doesn't answer is the question anyone actually running an agent harness locally cares about:

> *If I run my agent harness against this model at `--gpu-memory-utilization 0.85`, how many concurrent sessions can I run, and what does accuracy look like at that load — before I hit a wall?*

`dgx_spark_benchy` covers the same ground as the original (tiers 1–3, a 10-domain graded eval suite, trial statistics) **plus a capacity dimension**: it restarts vLLM at different GPU memory budgets, sweeps concurrency per agent workload profile, and reports the actual ceiling — not just tokens/sec on a single stream.

It's deliberately not tied to one product. Whether you're running **Hermes**, **OpenClaw**, a custom LangGraph loop, or something you built yourself — you point it at your OpenAI-compatible endpoint and either use the built-in generic profiles or drop in your harness's real prompts via a JSON file. No code changes required for the common case.

## What's in the box

| Command | What it measures |
|---|---|
| `tier1` | RoCE interconnect bandwidth (multi-node Spark clusters; skips cleanly on single-box setups) |
| `tier2` | Raw inference: TTFT, decode tok/s, TPOT, prefill tok/s across context sizes, throughput under concurrency |
| `tier3` | Real single-shot workloads: exec-verified coding task, tool-call correctness, long-context needle retrieval |
| `eval` | Graded scenario suite — 22 scenarios across **10 domains** (tool_use, structured, robustness, safety, restraint, multi_step, instruction, autonomous_planning, visual, long_context), partial-credit grading, Pass@1 / Pass@K / reliability gap, a weighted **LocalScore** (Quality/Reliability/Efficiency/Responsiveness) |
| `capacity` | **The addition.** Auto-restarts vLLM per `--gpu-utils` level, sweeps concurrency per agent workload profile (fixed list or `--concurrency auto` to escalate until each model's real ceiling is found), detects the capacity ceiling (error rate > 10% or accuracy drops below 70% of baseline), reports msgs/min and accuracy *at that ceiling* |
| `hermes` | Composite **Hermes Score** for running a Hermes-style personal-agent harness specifically: task quality (tool use, web search, long-context recall, multi-turn state) + capacity ceiling + responsiveness, rolled into one 0–100 number — see below |

All commands write to one long-format CSV (`results/*.csv`) plus a per-run Markdown report, so runs stay comparable over time the same way spark-bench's do.

### The capacity twist, in one command

```bash
./spark_bench_plus.py capacity \
  --label qwen-capacity \
  --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --endpoint http://localhost:8000/v1 \
  --vllm-cmd "vllm serve {model} --gpu-memory-utilization {gpu_util} --port {port}" \
  --gpu-utils 0.6,0.75,0.85,0.95 \
  --profiles orchestrator,coding_agent,chat_agent \
  --concurrency 1,2,4,8,16,32
```

```
=== CAPACITY SUMMARY ===
gpu_util=0.6    profile=orchestrator   ceiling=8     acc=0.71  msgs/min=142.3  ttft_p50=891ms
gpu_util=0.6    profile=coding_agent   ceiling=4     acc=0.88  msgs/min=96.1   ttft_p50=1120ms
gpu_util=0.85   profile=orchestrator   ceiling=16    acc=0.68  msgs/min=201.4  ttft_p50=1340ms
gpu_util=0.85   profile=coding_agent   ceiling=8     acc=0.82  msgs/min=134.5  ttft_p50=1580ms
gpu_util=0.95   profile=orchestrator   ceiling=16    acc=0.55  msgs/min=189.0  ttft_p50=2210ms
```
*(illustrative — numbers depend entirely on your model, quantization, and hardware)*

That's the actual answer to "how many messages can I send and how accurate will it be" — per GPU-memory budget, per workload shape.

Pass `--concurrency auto` instead of a fixed list to have it escalate (double each round: 1, 2, 4, 8...) until it actually finds the ceiling or hits `--max-concurrency` (default 256), rather than requiring you to guess a wide-enough list per model — useful because a strong model and a weak model don't break at the same concurrency, and a fixed list either wastes rounds on a weak model or stops too early on a strong one.

## Workload profiles — built for any harness, not just one

`agent_profiles.py` ships four profiles that cover the common shapes of agent traffic:

- **`orchestrator`** — multi-step tool-chain planning (the ReAct-style shape: plan → call tool → call tool → summarize). This is what Hermes-style autonomous orchestration, OpenClaw's agent loop, or any tool-using agent sends.
- **`coding_agent`** — code generation graded by actually executing the output against test cases. What Aider/Cursor-style or OpenClaw's coding mode sends.
- **`chat_agent`** — casual multi-turn conversational load with light context recall. What a personal-assistant-style harness sends.
- **`hermes`** — broader personal-agent harness shape: tool chains that also include `web_search`/`read_file` (not just calendar/email/reminder), long-context recall *inside* a multi-turn agent session, and multi-turn preference/state tracking. Backs the dedicated `hermes` command below.

Don't want to touch Python? Point `--profiles-file` at a JSON file with your harness's real prompts:

```json
{
  "my_harness_profile": [
    {
      "id": "task_1",
      "messages": [{"role": "user", "content": "..."}],
      "tools": null,
      "grader_type": "keyword",
      "grader_args": {"keywords": ["expected", "terms"]}
    }
  ]
}
```

See `profiles/example_custom_profile.json` for a working example. Supported `grader_type`s: `keyword`, `tool_sequence`, `json_valid`, `code_exec` (executes the reply's code block and checks its return value against test cases — see `agent_profiles.py`'s `PROFILE_JSON_SCHEMA` comment for the exact shape). Full profiling against your **actual** prompts will always be more representative than the generic ones — copy a built-in profile and edit it to match your real traffic if you can.

## Hermes Benchmark

If you're specifically trying to answer "which model should I run behind my Hermes-style agent harness, and how big does it need to be" — the `hermes` command runs the `hermes` profile above through a quality pass (single-shot, no load) and a capacity sweep (auto-escalating by default), then combines them into one **Hermes Score**:

```bash
./spark_bench_plus.py hermes \
  --label nemotron-super-hermes \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --endpoint http://localhost:8000/v1 \
  --gpu-util 0.85 --concurrency auto \
  --target-ceiling 16 --target-ttft-ms 1500
```

Hermes Score = 50% task quality + 30% capacity ceiling (normalized against `--target-ceiling` concurrent sessions) + 20% responsiveness (normalized against `--target-ttft-ms`). This command assumes your endpoint is already serving — it doesn't manage vLLM lifecycle, same as `capacity --skip-restart`.

Run it against every candidate you're considering and `results/LEADERBOARD.md` (via `./leaderboard.py --write`) will show a ranked Hermes Score table plus a callout for the **smallest** and **largest** models that clear a 70/100 bar — the actual answer to "what's the smallest model that still works well enough for Hermes, and what's the biggest one I've actually validated."

### Generating a bigger, more diverse task pool first

The built-in profiles ship a handful of hand-written tasks each (3–7). That's fine for a quick read, but at high concurrency (`--concurrency auto` pushing toward 32+) cycling through only a few prompts repeatedly can make vLLM's prefix caching look faster than it would on real, varied traffic — and doesn't stress reasoning breadth the way genuinely different Hermes requests would. `generate_data.py` uses a big "generator" model to produce a much larger pool:

```bash
./generate_data.py \
  --endpoint http://localhost:8000/v1 \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --profiles orchestrator,coding_agent,chat_agent,hermes \
  --count 30 \
  --out profiles/generated.json

./spark_bench_plus.py hermes --label my-candidate --model ... --endpoint ... \
  --profiles-file profiles/generated.json
```

This doesn't manage vLLM lifecycle either — load the generator model yourself first, generate the data, then swap back to whatever you're actually benchmarking. See "Choosing a model for offline data generation" below for which model to use as the generator.

## Quickstart

```bash
git clone https://github.com/<you>/dgx_spark_benchy.git
cd dgx_spark_benchy
# stdlib only — no pip install needed, just Python 3.10+

# 1. what the original spark-bench covers
./spark_bench_plus.py tier2 --label baseline \
  --model your-model --endpoint http://localhost:8000/v1 \
  --contexts 4096,32768 --concurrency 1,4,16

./spark_bench_plus.py eval --label baseline \
  --model your-model --endpoint http://localhost:8000/v1 --repeats 3

# 2. the capacity twist
./spark_bench_plus.py capacity --label baseline \
  --model your-model --endpoint http://localhost:8000/v1 \
  --gpu-utils 0.85 --skip-restart \
  --profiles orchestrator,coding_agent,chat_agent \
  --concurrency 1,2,4,8,16,32
```

Any OpenAI-compatible `/v1/chat/completions` endpoint works — vLLM, llama.cpp, SGLang, OpenRouter. `--skip-restart` assumes your server is already running; omit it and pass `--vllm-cmd` if you want the `capacity` command to manage vLLM restarts across GPU-utilization levels itself (locally or over SSH via `--ssh-host`).

## Leaderboard + quick screening for new candidates

If you're trying a lot of NVFP4 quants or trimmed models (unsloth releases, your own fine-tunes,
someone else's PR) and just want a fast read before committing to a full multi-repeat eval:

```bash
./quickbench.sh <label> <model> <endpoint> [gpu_util]

# e.g.
./quickbench.sh qwen3-nvfp4-v2 unsloth/Qwen3-32B-NVFP4 http://localhost:8001/v1 0.85
```

This runs `tier3` (sanity), a 1-repeat `eval` (all 10 domains), and a single-`gpu_util` `capacity`
pass against `orchestrator`/`coding_agent`/`chat_agent`, then rebuilds `results/LEADERBOARD.md`.
The candidate's own OpenAI-compatible server needs to already be serving at `--endpoint` —
quickbench doesn't start vLLM for you, since serving flags vary too much across quant formats to
assume one invocation fits every candidate.

If you're also running another resident model on the same box that needs its memory freed up for
the candidate, stop/start that service yourself around the call (`docker stop <name>` / `docker
start <name>`, or however you manage it) — that's outside quickbench's scope on purpose, since it
depends entirely on your own setup.

Anyone can benchmark their own model the same way — point `quickbench.sh` (or the underlying
`spark_bench_plus.py` commands directly) at any OpenAI-compatible endpoint, no code changes
required. `agent_profiles.py`'s generic profiles work out of the box; a JSON `--profiles-file`
lets you swap in real prompts from your own harness without touching Python (see
`profiles/example_custom_profile.json`).

Rebuild the leaderboard anytime from accumulated results:

```bash
./leaderboard.py --write   # reads results/spark_bench_plus.csv -> results/LEADERBOARD.md
```

It renders two views: an overall ranking by LocalScore, and an orchestrator/Hermes-style view
sorted by capacity ceiling on the `orchestrator` workload profile (the ReAct/tool-chain traffic
shape Hermes-style harnesses send) plus tool-call correctness — useful when you're screening
candidates specifically for an agent-orchestration use case rather than general chat quality.

## Repository structure

```
dgx_spark_benchy/
├── spark_bench_plus.py    # CLI entrypoint — tier1/tier2/tier3/eval/capacity/hermes
├── eval_scenarios.py       # 22 graded scenarios across 10 domains
├── agent_profiles.py       # orchestrator/coding_agent/chat_agent/hermes workload profiles
├── generate_data.py        # generator-model -> larger diverse task pool JSON for any profile
├── leaderboard.py          # builds results/LEADERBOARD.md from accumulated CSV results
├── quickbench.sh           # fast tier3+eval+capacity screening pass for a new model candidate
├── profiles/
│   └── example_custom_profile.json
├── assets/
│   └── logo.svg
└── results/                 # CSV + per-run markdown + saved visual artifacts (gitignored)
    ├── runs/
    └── artifacts/
```

## Choosing a model for offline data generation

If you're using the Spark's spare cycles to generate synthetic/benchmark data (e.g. for
`generate_data.py`) rather than serve requests interactively, latency doesn't matter — quality
does, so pick the largest model that fits `128GB` of unified memory with real headroom left for
KV cache and OS/container overhead, not the absolute edge case:

- **NVIDIA Nemotron Super (120B-A12B, NVFP4)** is the strongest pick if it's already on your box:
  it's a mixture-of-experts model **trained natively at NVFP4** (not quantized after the fact
  post-training), so it holds quality much closer to full precision than a typical PTQ 4-bit
  quant, with benchmarks (MMLU-Pro, GPQA, HMMT) landing within a point of its BF16 numbers. Its
  weights run **~82GB**, leaving ~46GB of your 128GB for KV cache and OS overhead — comfortable
  room for long generation prompts.
- **`gpt-oss-120b` (NVFP4/MXFP4)** is a close alternative: also a strong reasoning/tool-use MoE,
  with an even smaller **~63–68GB** footprint (more KV-cache headroom still), OpenAI-trained
  specifically for agentic tasks.
- **Llama 3.1/3.3 70B (NVFP4)** is a solid dense option at a similar ~35–40GB footprint if you
  want the most headroom, or prefer Llama's output style for your dataset.
- Avoid reaching for the biggest MoE options like Qwen3-235B-class NVFP4 quants (~120GB+ for
  weights alone) — on a 128GB box that leaves no slack for KV cache or the OS, and
  `--gpu-memory-utilization` needs that slack to avoid OOM stalls mid-run.

Treat it as a one-off "generator" run: stop whatever's serving your day-to-day harness, load the
big generator model, run `generate_data.py` against it, then restart your regular service — same
manual stop/start pattern as screening a new candidate with `quickbench.sh` above.

## Design principles

Same ones spark-bench uses, because they're good ones:

- **stdlib only.** No dependency hell on a Spark box you're also using for training runs.
- **Partial credit, not pass/fail.** A model hitting 2 of 3 required tool calls scores 0.66, not 0.
- **Trial statistics matter.** Every eval scenario runs `--repeats` times; Pass@1 vs Pass@K exposes flakiness a single run hides.
- **Long-format CSV.** Every run appends comparable rows — build your own leaderboard however you like.

## What's not (yet) in here

Being upfront: the original spark-bench has production-hygiene gates this project doesn't replicate yet — a golden-gate grader self-check, an endpoint preflight probe, degenerate-score quarantine flags, box locking for shared endpoints, and an HTML report generator. Those matter more for a public multi-model leaderboard than for benchmarking your own box, but they're a reasonable place to contribute if you want to extend this further.

## Contributing

Scenarios, workload profiles, and graders are all just data — adding one is adding a dict to a list. PRs adding realistic profiles for other popular harnesses, or closing the gaps above, are welcome.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built as an extension of [spark-bench](https://github.com/Weschera/spark-bench) by Weschera — tier1/tier2/tier3 design and the partial-credit/trial-statistics grading philosophy originate there.
