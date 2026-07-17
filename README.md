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
| `capacity` | **The addition.** Auto-restarts vLLM per `--gpu-utils` level, sweeps concurrency per agent workload profile, detects the capacity ceiling (error rate > 10% or accuracy drops below 70% of baseline), reports msgs/min and accuracy *at that ceiling* |

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

## Workload profiles — built for any harness, not just one

`agent_profiles.py` ships three generic profiles that cover the common shapes of agent traffic:

- **`orchestrator`** — multi-step tool-chain planning (the ReAct-style shape: plan → call tool → call tool → summarize). This is what Hermes-style autonomous orchestration, OpenClaw's agent loop, or any tool-using agent sends.
- **`coding_agent`** — code generation graded by actually executing the output against test cases. What Aider/Cursor-style or OpenClaw's coding mode sends.
- **`chat_agent`** — casual multi-turn conversational load with light context recall. What a personal-assistant-style harness sends.

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

See `profiles/example_custom_profile.json` for a working example. Supported `grader_type`s: `keyword`, `tool_sequence`, `json_valid`. Full profiling against your **actual** prompts will always be more representative than the generic ones — copy a built-in profile and edit it to match your real traffic if you can.

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

## Repository structure

```
dgx_spark_benchy/
├── spark_bench_plus.py    # CLI entrypoint — tier1/tier2/tier3/eval/capacity
├── eval_scenarios.py       # 22 graded scenarios across 10 domains
├── agent_profiles.py       # orchestrator/coding_agent/chat_agent workload profiles
├── profiles/
│   └── example_custom_profile.json
├── assets/
│   └── logo.svg
└── results/                 # CSV + per-run markdown + saved visual artifacts (gitignored)
    ├── runs/
    └── artifacts/
```

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
