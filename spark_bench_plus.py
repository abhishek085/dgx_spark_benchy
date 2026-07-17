#!/usr/bin/env python3
"""
spark_bench_plus.py — DGX Spark benchmark harness.

Covers the same ground as github.com/Weschera/spark-bench, plus a capacity
dimension it doesn't have. Same design principles: stdlib only, one
OpenAI-compatible streaming client, long-format CSV output, comparable
across runs.

  tier1     interconnect (ib_write_bw/ib_read_bw over RoCE — multi-node only)
  tier2     raw inference: TTFT, decode tok/s, TPOT, prefill tok/s across
            context sizes, throughput under concurrency
  tier3     real single-shot workloads: coding (exec-verified), tool call,
            long-context needle retrieval
  eval      graded scenario suite (tool_use, structured, robustness,
            safety, restraint, multi_step, long_context) with partial
            credit + trial statistics (Pass@1, Pass@K, reliability gap)
  capacity  <- the twist. Not "how fast is one request" but "how many
            concurrent AGENT SESSIONS can this box sustain, at a given
            vLLM --gpu-memory-utilization, before accuracy or latency
            falls over" — using pluggable workload profiles (see
            agent_profiles.py) so it fits whatever harness you actually
            run: Hermes, OpenClaw, a custom LangGraph loop, anything.

Requirements: Python 3.10+, stdlib only. Any OpenAI-compatible endpoint
(vLLM, llama.cpp, OpenRouter, etc).

Examples:
  # what the original repo does
  ./spark_bench_plus.py tier2 --label qwen-baseline \\
      --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \\
      --contexts 4096,32768 --concurrency 1,4,16

  ./spark_bench_plus.py eval --label qwen-eval \\
      --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \\
      --repeats 3

  # the twist: capacity at a given gpu_util, auto-restarting vLLM per level
  ./spark_bench_plus.py capacity --label qwen-capacity \\
      --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \\
      --vllm-cmd "vllm serve {model} --gpu-memory-utilization {gpu_util} --port {port}" \\
      --gpu-utils 0.6,0.75,0.85,0.95 \\
      --profiles orchestrator,coding_agent,chat_agent \\
      --concurrency 1,2,4,8,16,32

  # vLLM already running at a fixed gpu_util, just profile your own harness's
  # real prompts (edit hermes_tasks.json to match what your harness sends)
  ./spark_bench_plus.py capacity --label hermes-live \\
      --model nvidia/Qwen3.6-35B-A3B-NVFP4 --endpoint http://localhost:8000/v1 \\
      --gpu-utils 0.85 --skip-restart \\
      --profiles my_hermes_profile --profiles-file hermes_tasks.json \\
      --concurrency 1,2,4,8,16,32,48
"""

import argparse, csv, json, os, re, statistics, subprocess, sys, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import eval_scenarios as ev
import agent_profiles as ap

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results")
CSV_FIELDS = ["run_id", "timestamp", "cmd", "model", "label", "endpoint",
              "gpu_util", "profile_or_workload", "concurrency", "context",
              "metric", "value", "unit", "notes"]

# --------------------------------------------------------------------------- #
# Run context / CSV + markdown output (same shape as spark-bench)
# --------------------------------------------------------------------------- #

@dataclass
class Ctx:
    label: str
    cmd: str
    model: str = ""
    endpoint: str = ""
    out_dir: str = DEFAULT_OUT
    run_id: str = ""
    rows: list = field(default_factory=list)
    md: list = field(default_factory=list)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"{self.label}-{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(os.path.join(self.out_dir, "runs"), exist_ok=True)

    def add(self, gpu_util, profile_or_workload, concurrency, metric, value, unit,
             *, context="", notes=""):
        row = {"run_id": self.run_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "cmd": self.cmd, "model": self.model, "label": self.label,
               "endpoint": self.endpoint, "gpu_util": gpu_util,
               "profile_or_workload": profile_or_workload, "concurrency": concurrency,
               "context": context, "metric": metric, "value": value, "unit": unit,
               "notes": notes}
        self.rows.append(row)
        v = f"{value:.2f}" if isinstance(value, float) else value
        print(f"  [{profile_or_workload:<16}] {metric:<20} = {v} {unit}" + (f" ({notes})" if notes else ""))

    def mdln(self, line=""):
        self.md.append(line)

    def flush(self):
        os.makedirs(self.out_dir, exist_ok=True)
        csv_path = os.path.join(self.out_dir, "spark_bench_plus.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            for r in self.rows:
                w.writerow(r)
        md_path = os.path.join(self.out_dir, "runs", f"{self.run_id}.md")
        with open(md_path, "w") as f:
            f.write("\n".join(self.md) + "\n")
        print(f"\n-> appended {len(self.rows)} rows to {csv_path}")
        print(f"-> wrote summary {md_path}")
        return csv_path, md_path

# --------------------------------------------------------------------------- #
# OpenAI-compatible streaming client
# --------------------------------------------------------------------------- #

def chat_stream(endpoint, model, messages, max_tokens, *, temperature=0.3, tools=None, timeout=300):
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    url = endpoint.rstrip("/") + "/chat/completions"
    key = os.environ.get("SPARK_BENCH_API_KEY") or os.environ.get("OPENAI_API_KEY") or "none"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    ttft = None
    text_parts, tool_frags = [], []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text_parts.append(delta["content"])
                    if delta.get("tool_calls"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        tool_frags.extend(delta["tool_calls"])
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return {"error": str(e)[:100], "ttft": None, "total": time.perf_counter() - t0,
                "text": "", "tool_calls": [], "prompt_tokens": 0, "completion_tokens": 0,
                "decode_tps": 0.0, "tpot_ms": 0.0}
    total = time.perf_counter() - t0
    text = "".join(text_parts)
    comp = usage.get("completion_tokens") or max(1, len(text) // 4)
    ptoks = usage.get("prompt_tokens")
    decode_t = max(total - (ttft or total), 1e-6)
    return {"error": None, "ttft": ttft if ttft is not None else total, "total": total,
            "text": text, "tool_calls": _merge_tool_calls(tool_frags),
            "prompt_tokens": ptoks, "completion_tokens": comp,
            "decode_tps": comp / decode_t if comp else 0.0,
            "tpot_ms": (decode_t / max(comp - 1, 1)) * 1000 if comp > 1 else 0.0}

def _merge_tool_calls(fragments):
    by_idx = {}
    for f in fragments:
        i = f.get("index", 0)
        d = by_idx.setdefault(i, {"function": {"name": "", "arguments": ""}})
        fn = f.get("function", {})
        if fn.get("name"):
            d["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            d["function"]["arguments"] += fn["arguments"]
    return list(by_idx.values())

def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]

# --------------------------------------------------------------------------- #
# Tier 1 : interconnect (multi-node Spark clusters only)
# --------------------------------------------------------------------------- #

def _which(b):
    from shutil import which
    return which(b) is not None

def run_tier1(ctx, args):
    ctx.mdln(f"# Tier 1 — Interconnect ({ctx.run_id})\n")
    if not _which("ib_write_bw"):
        ctx.add("na", "interconnect", "na", "ib_write_bw", "MISSING", "",
                 notes="apt install perftest; single-node Spark boxes can skip tier1")
        ctx.mdln("> `ib_write_bw` not found — install `perftest`, or skip tier1 entirely if this "
                  "is a single Spark box with no RoCE fabric.\n")
        return
    ctx.mdln(f"- peer: `{args.peer_ssh}` links: `{args.links}`\n")
    ctx.mdln("| link | test | BW peak (Gb/s) |")
    ctx.mdln("|------|------|---------------:|")
    for spec in args.links.split(","):
        spec = spec.strip()
        if not spec:
            continue
        dev, peer_ip = spec.split(":")
        cmd = ["ib_write_bw", "-d", dev, "-F", "--report_gbits", peer_ip]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            bw = 0.0
            for ln in out.splitlines():
                m = re.match(r"\s*\d+\s+\d+\s+([\d.]+)", ln)
                if m:
                    bw = max(bw, float(m.group(1)))
            ctx.add("na", dev, "na", "ib_write_bw", round(bw, 1), "Gb/s")
            ctx.mdln(f"| `{dev}` | ib_write_bw | {bw:.1f} |")
        except Exception as e:
            ctx.add("na", dev, "na", "ib_write_bw", "FAILED", "", notes=str(e)[:60])
            ctx.mdln(f"| `{dev}` | ib_write_bw | FAILED |")
    ctx.mdln()

# --------------------------------------------------------------------------- #
# Tier 2 : inference
# --------------------------------------------------------------------------- #

FILLER = ("The DGX Spark is a compact AI workstation built around the GB10 "
          "Grace-Blackwell superchip with unified LPDDR5X memory. ")

def _make_prompt(approx_tokens):
    target_chars = int(approx_tokens * 4)
    reps = max(1, target_chars // len(FILLER))
    return FILLER * reps

def run_tier2(ctx, args):
    ctx.mdln(f"# Tier 2 — Inference ({ctx.run_id})\n")
    ctx.mdln(f"- model `{ctx.model}` @ `{ctx.endpoint}`\n")
    ctx.mdln("## Single-stream decode\n")
    ctx.mdln("| context | TTFT (ms) | decode tok/s | TPOT (ms) | prefill tok/s | out toks |")
    ctx.mdln("|--------:|----------:|-------------:|----------:|--------------:|---------:|")
    for cxt in [int(c) for c in args.contexts.split(",") if c.strip()]:
        prompt = _make_prompt(cxt) + ("\n\nWrite a ~400 word essay on how speculative decoding "
                                       "accelerates LLM inference.")
        r = chat_stream(ctx.endpoint, ctx.model, [{"role": "user", "content": prompt}],
                         args.gen_tokens, timeout=args.timeout)
        if r["error"]:
            ctx.add("na", "single_stream", "na", "error", r["error"], "", context=cxt)
            ctx.mdln(f"| {cxt} | ERR | - | - | - | - |")
            continue
        ptoks = r["prompt_tokens"] or cxt
        prefill_tps = ptoks / r["ttft"] if r["ttft"] else 0.0
        ctx.add("na", "single_stream", "na", "ttft_ms", round(r["ttft"] * 1000, 1), "ms", context=ptoks)
        ctx.add("na", "single_stream", "na", "decode_tps", round(r["decode_tps"], 2), "tok/s", context=ptoks)
        ctx.add("na", "single_stream", "na", "tpot_ms", round(r["tpot_ms"], 2), "ms", context=ptoks)
        ctx.mdln(f"| {ptoks} | {r['ttft']*1000:.0f} | {r['decode_tps']:.1f} | {r['tpot_ms']:.1f} | "
                  f"{prefill_tps:.0f} | {r['completion_tokens']} |")
    ctx.mdln()

    concs = [int(c) for c in args.concurrency.split(",") if c.strip()]
    if concs:
        ctx.mdln("## Throughput under concurrency\n")
        ctx.mdln("| batch | agg decode tok/s | per-stream tok/s | TTFT p50 (ms) | TTFT p99 (ms) | completed |")
        ctx.mdln("|------:|-----------------:|-----------------:|--------------:|--------------:|----------:|")
        cprompt = _make_prompt(args.conc_context) + "\n\nExplain in ~300 words how tensor parallelism shards a transformer."
        for n in concs:
            t0 = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(chat_stream, ctx.endpoint, ctx.model,
                                   [{"role": "user", "content": cprompt}], args.gen_tokens,
                                   timeout=args.timeout) for _ in range(n)]
                for f in as_completed(futs):
                    results.append(f.result())
            wall = time.perf_counter() - t0
            oks = [r for r in results if not r["error"]]
            if not oks:
                ctx.add("na", "concurrency", n, "error", "no completed requests", "")
                ctx.mdln(f"| {n} | ERR | - | - | - | 0 |")
                continue
            total_out = sum(r["completion_tokens"] for r in oks)
            ttfts = sorted(r["ttft"] * 1000 for r in oks)
            agg_tps = total_out / wall
            per_tps = statistics.mean(r["decode_tps"] for r in oks)
            ctx.add("na", "concurrency", n, "agg_decode_tps", round(agg_tps, 1), "tok/s")
            ctx.add("na", "concurrency", n, "per_stream_tps", round(per_tps, 2), "tok/s")
            ctx.add("na", "concurrency", n, "ttft_p50_ms", round(_pct(ttfts, 50), 1), "ms")
            ctx.add("na", "concurrency", n, "ttft_p99_ms", round(_pct(ttfts, 99), 1), "ms")
            ctx.mdln(f"| {n} | {agg_tps:.0f} | {per_tps:.1f} | {_pct(ttfts,50):.0f} | "
                      f"{_pct(ttfts,99):.0f} | {len(oks)}/{n} |")
        ctx.mdln()

# --------------------------------------------------------------------------- #
# Tier 3 : real single-shot workloads
# --------------------------------------------------------------------------- #

CODING_PROMPT = ("Write a Python function `merge_intervals(intervals)` that merges a list "
                  "of [start,end] pairs into sorted non-overlapping intervals, handling "
                  "empty input. Output ONLY one python code block.")
TOOL_PROMPT = "What is the current weather in Paris, France? Use the available tool."
_WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]}}}]

def _check_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    if "def merge_intervals" not in code:
        return False
    try:
        ns = {}
        exec(code, ns)
        f = ns.get("merge_intervals")
        return f([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]] and f([]) == []
    except Exception:
        return False

def _build_haystack(approx_tokens, passcode, depth=0.55):
    line = ("In the quarterly review, throughput remained stable across all regional "
            "clusters during the maintenance window. ")
    n = max(2, int(approx_tokens * 4) // len(line))
    lines = [line] * n
    lines[int(n * depth)] = f"IMPORTANT: the secret passcode is {passcode}. Remember it. "
    return "".join(lines)

def run_tier3(ctx, args):
    ctx.mdln(f"# Tier 3 — Real workloads ({ctx.run_id})\n")
    ctx.mdln("| workload | pass | E2E (s) | TTFT (ms) | decode tok/s | out toks |")
    ctx.mdln("|----------|:----:|--------:|----------:|-------------:|---------:|")

    r = chat_stream(ctx.endpoint, ctx.model, [{"role": "user", "content": CODING_PROMPT}],
                     2048, timeout=args.timeout)
    ok = _check_code(r["text"]) if not r["error"] else False
    _tier3_emit(ctx, "coding", ok, r)

    r = chat_stream(ctx.endpoint, ctx.model, [{"role": "user", "content": TOOL_PROMPT}],
                     512, timeout=args.timeout, tools=_WEATHER_TOOL)
    called = any((tc.get("function", {}) or {}).get("name") == "get_weather" for tc in r["tool_calls"])
    _tier3_emit(ctx, "tool_call", called, r)

    passcode = "ZARQON-7741-MERIDIAN"
    haystack = _build_haystack(args.retrieval_context, passcode)
    q = haystack + "\n\nQUESTION: What is the secret passcode mentioned above? Reply with ONLY the passcode."
    r = chat_stream(ctx.endpoint, ctx.model, [{"role": "user", "content": q}], 256, timeout=args.timeout)
    ok = passcode in r["text"].upper() if not r["error"] else False
    _tier3_emit(ctx, "long_ctx_retrieval", ok, r)
    ctx.mdln()

def _tier3_emit(ctx, name, ok, r):
    ctx.add("na", name, "na", "pass", 1 if ok else 0, "bool")
    ctx.add("na", name, "na", "e2e_s", round(r["total"], 2), "s")
    ctx.add("na", name, "na", "ttft_ms", round((r["ttft"] or 0) * 1000, 1), "ms")
    ctx.add("na", name, "na", "decode_tps", round(r["decode_tps"], 2), "tok/s")
    ctx.mdln(f"| {name} | {'✅' if ok else '❌'} | {r['total']:.1f} | {(r['ttft'] or 0)*1000:.0f} | "
              f"{r['decode_tps']:.1f} | {r['completion_tokens']} |")

# --------------------------------------------------------------------------- #
# Eval : graded scenario suite (see eval_scenarios.py)
# --------------------------------------------------------------------------- #

def run_eval(ctx, args):
    ctx.mdln(f"# Eval ({ctx.run_id})\n")
    ctx.mdln(f"- model `{ctx.model}` @ `{ctx.endpoint}` repeats `{args.repeats}`\n")

    def chat_fn(messages, tools, max_tokens):
        r = chat_stream(ctx.endpoint, ctx.model, messages, max_tokens,
                         tools=tools, timeout=args.timeout)
        return r

    def progress(rec):
        print(f"  [eval] {rec['id']:<14} {rec['domain']:<20} score={rec['score']:.2f} "
              f"cons={rec['consistency']:.2f}")

    domains = args.domains.split(",") if args.domains else None
    artifact_dir = os.path.join(args.out_dir, "artifacts", ctx.run_id)
    res = ev.run_suite(chat_fn, repeats=args.repeats, domains=domains,
                        timeout=args.timeout, progress=progress,
                        artifact_dir=artifact_dir, efficiency_target_tps=args.efficiency_target_tps)
    ov = res["overall"]
    ts = res["trial_stats"]

    for k in ("quality", "reliability", "efficiency", "responsiveness", "localscore"):
        ctx.add("na", "overall", "na", k, round(ov[k], 1), "score0-100")

    ctx.mdln(f"## LocalScore {ov['localscore']:.1f}/100\n")
    ctx.mdln("| component | score | weight |")
    ctx.mdln("|-----------|------:|-------:|")
    ctx.mdln(f"| Quality | {ov['quality']:.1f} | 45% |")
    ctx.mdln(f"| Reliability | {ov['reliability']:.1f} | 25% |")
    ctx.mdln(f"| Efficiency | {ov['efficiency']:.1f} | 10% |")
    ctx.mdln(f"| Responsiveness | {ov['responsiveness']:.1f} | 20% |")
    ctx.mdln(f"\nMedian latency {ov['median_latency_s']:.2f}s · mean decode {ov['mean_decode_tps']:.1f} tok/s "
              f"· {ov['n_scenarios']} scenarios\n")

    ctx.mdln("## Trial statistics\n")
    ctx.mdln("| metric | value | meaning |")
    ctx.mdln("|--------|------:|---------|")
    ctx.mdln(f"| Pass@1 | {ts['pass_at_1']}% | scenarios passing on at least 1 repeat |")
    ctx.mdln(f"| Pass@K | {ts['pass_at_k']}% | scenarios passing on ALL repeats |")
    ctx.mdln(f"| Reliability Gap | {ts['reliability_gap']}% | Pass@1 - Pass@K (flakiness cost) |")
    ctx.mdln(f"| Score StdDev | {ts['score_stddev']} | cross-scenario spread |\n")

    ctx.mdln("## Domain breakdown\n")
    ctx.mdln("| domain | n | quality | reliability |")
    ctx.mdln("|--------|--:|--------:|------------:|")
    for d in res["domains"]:
        ctx.add("na", d["domain"], "na", "domain_quality", round(d["quality"], 1), "score0-100")
        ctx.mdln(f"| {d['domain']} | {d['n']} | {d['quality']:.1f} | {d['reliability']:.1f} |")

    ctx.mdln("\n## Per-scenario\n")
    ctx.mdln("| id | domain | score | consistency | latency |")
    ctx.mdln("|----|--------|------:|------------:|--------:|")
    for s in res["scenarios"]:
        ctx.add("na", s["id"], "na", "score", round(s["score"], 3), "frac", notes=s["domain"])
        ctx.mdln(f"| {s['id']} | {s['domain']} | {s['score']:.2f} | {s['consistency']:.2f} | {s['latency']:.1f}s |")
    ctx.mdln()

    if res.get("artifacts"):
        ctx.mdln("## Saved artifacts (visual domain)\n")
        for a in res["artifacts"]:
            ctx.mdln(f"- `{a['id']}` ({a['domain']}, score {a['score']:.2f}): `{a['path']}`")
            ctx.add("na", a["id"], "na", "artifact", a["path"], "file", notes=a["domain"])
        ctx.mdln()
        print(f"  -> {len(res['artifacts'])} artifact(s) saved under {os.path.join(args.out_dir, 'artifacts', ctx.run_id)}")

    print(f"\n=== LocalScore {ov['localscore']:.1f}/100 "
          f"(Q{ov['quality']:.0f} Rel{ov['reliability']:.0f} Eff{ov['efficiency']:.0f} "
          f"Resp{ov['responsiveness']:.0f}) "
          f"Pass@1={ts['pass_at_1']}% Pass@K={ts['pass_at_k']}% RelGap={ts['reliability_gap']}% ===")

# --------------------------------------------------------------------------- #
# vLLM lifecycle (local or SSH) — for the capacity command
# --------------------------------------------------------------------------- #

class VLLMManager:
    def __init__(self, cmd_template, model, port, ssh_host=None, startup_timeout=420):
        self.cmd_template = cmd_template
        self.model = model
        self.port = port
        self.ssh_host = ssh_host
        self.health_url = f"http://localhost:{port}/v1/models"
        self.startup_timeout = startup_timeout
        self.proc = None

    def stop(self):
        if self.ssh_host:
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                             self.ssh_host, "pkill -9 -f 'vllm serve' || true"], capture_output=True)
        elif self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        time.sleep(3)

    def start(self, gpu_util):
        self.stop()
        cmd = self.cmd_template.format(model=self.model, gpu_util=gpu_util, port=self.port)
        print(f"[vllm] starting: {cmd}")
        if self.ssh_host:
            full = f"nohup {cmd} > /tmp/vllm_{self.port}.log 2>&1 & disown"
            subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                             self.ssh_host, full], check=True)
        else:
            log = open(f"/tmp/vllm_{self.port}.log", "w")
            self.proc = subprocess.Popen(cmd.split(), stdout=log, stderr=subprocess.STDOUT)
        self._wait_healthy()

    def _wait_healthy(self):
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            try:
                with urllib.request.urlopen(self.health_url, timeout=5) as r:
                    if r.status == 200:
                        print(f"[vllm] healthy after {time.time()-t0:.0f}s")
                        return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(f"vLLM not healthy within {self.startup_timeout}s (see /tmp/vllm_{self.port}.log)")

# --------------------------------------------------------------------------- #
# Capacity : the twist — profile x gpu_util x concurrency
# --------------------------------------------------------------------------- #

def run_capacity_sweep(ctx, endpoint, model, gpu_util, profile_name, tasks,
                        concurrency_levels, max_tokens, timeout):
    print(f"\n=== profile={profile_name} gpu_util={gpu_util} ===")
    baseline_acc = None
    ceiling = None
    last_row = None
    for n in concurrency_levels:
        t0 = time.perf_counter()
        results = []

        def worker(i):
            task = tasks[i % len(tasks)]
            r = chat_stream(endpoint, model, task["messages"], max_tokens,
                             tools=task.get("tools"), timeout=timeout)
            if r["error"]:
                return {"ok": False, "score": 0.0, "ttft": None}
            score = task["grader"](r["text"], r["tool_calls"])
            return {"ok": True, "score": score, "ttft": r["ttft"]}

        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(worker, i) for i in range(n)]
            for f in as_completed(futs):
                results.append(f.result())
        wall = time.perf_counter() - t0

        oks = [r for r in results if r["ok"]]
        error_rate = 1 - len(oks) / len(results) if results else 1.0
        mean_acc = statistics.mean(r["score"] for r in oks) if oks else 0.0
        ttfts = [r["ttft"] * 1000 for r in oks if r["ttft"] is not None]
        msgs_per_min = (len(oks) / wall) * 60 if wall > 0 else 0.0
        if baseline_acc is None:
            baseline_acc = mean_acc

        breaking = error_rate > 0.10 or (baseline_acc > 0 and mean_acc < 0.70 * baseline_acc)
        if breaking and ceiling is None:
            ceiling = n

        ctx.add(gpu_util, profile_name, n, "accuracy", round(mean_acc, 3), "frac")
        ctx.add(gpu_util, profile_name, n, "error_rate", round(error_rate, 3), "frac")
        ctx.add(gpu_util, profile_name, n, "msgs_per_min", round(msgs_per_min, 1), "msg/min")
        ctx.add(gpu_util, profile_name, n, "ttft_p50_ms", round(_pct(ttfts, 50), 1), "ms")

        flag = "  <-- CEILING" if breaking else ""
        print(f"  conc={n:<4} acc={mean_acc:.2f} err={error_rate:.0%} "
              f"msgs/min={msgs_per_min:.1f} ttft_p50={_pct(ttfts,50):.0f}ms{flag}")
        last_row = (n, mean_acc, error_rate, msgs_per_min, _pct(ttfts, 50))
        if breaking:
            break

    ctx.add(gpu_util, profile_name, ceiling or concurrency_levels[-1], "capacity_ceiling",
             ceiling or -1, "concurrency",
             notes="max sustainable concurrency; -1 = not reached in sweep")
    return last_row, ceiling

def run_capacity(ctx, args):
    ctx.mdln(f"# Capacity ({ctx.run_id})\n")
    profiles = ap.get_profiles([p.strip() for p in args.profiles.split(",")],
                                profiles_file=args.profiles_file)
    gpu_utils = [float(x) for x in args.gpu_utils.split(",") if x.strip()]
    concurrency_levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    mgr = VLLMManager(args.vllm_cmd, ctx.model, args.vllm_port, ssh_host=args.ssh_host)

    all_results = {}
    for gpu_util in gpu_utils:
        if not args.skip_restart:
            mgr.start(gpu_util)
        else:
            print(f"[vllm] --skip-restart: assuming endpoint already at gpu_util={gpu_util}")
        for name, tasks in profiles.items():
            last_row, ceiling = run_capacity_sweep(ctx, ctx.endpoint, ctx.model, gpu_util, name,
                                                     tasks, concurrency_levels, args.max_tokens,
                                                     args.timeout)
            all_results[(gpu_util, name)] = (last_row, ceiling)
    if not args.skip_restart:
        mgr.stop()

    ctx.mdln("## Capacity summary\n")
    ctx.mdln("| gpu_util | profile | ceiling | acc@ceiling | msgs/min@ceiling | TTFT p50@ceiling (ms) |")
    ctx.mdln("|---------:|---------|--------:|------------:|------------------:|-----------------------:|")
    print("\n=== CAPACITY SUMMARY ===")
    for (gpu_util, name), (last_row, ceiling) in all_results.items():
        if not last_row:
            continue
        n, acc, err, mpm, ttft50 = last_row
        ceil_label = str(ceiling) if ceiling else f">{concurrency_levels[-1]}"
        ctx.mdln(f"| {gpu_util} | {name} | {ceil_label} | {acc:.2f} | {mpm:.1f} | {ttft50:.0f} |")
        print(f"gpu_util={gpu_util:<6} profile={name:<14} ceiling={ceil_label:<6} "
              f"acc={acc:.2f} msgs/min={mpm:.1f} ttft_p50={ttft50:.0f}ms")
    ctx.mdln()

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_ctx(args, cmd):
    return Ctx(label=args.label, cmd=cmd, model=getattr(args, "model", "") or "",
               endpoint=getattr(args, "endpoint", "") or "", out_dir=args.out_dir)

def main():
    p = argparse.ArgumentParser(description="spark-bench-plus")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_model=True):
        sp.add_argument("--label", required=True)
        sp.add_argument("--out-dir", default=DEFAULT_OUT)
        if need_model:
            sp.add_argument("--endpoint", required=True)
            sp.add_argument("--model", required=True)
            sp.add_argument("--timeout", type=int, default=300)

    s1 = sub.add_parser("tier1"); common(s1, need_model=False)
    s1.add_argument("--peer-ssh", default="")
    s1.add_argument("--links", default="")

    s2 = sub.add_parser("tier2"); common(s2)
    s2.add_argument("--contexts", default="4096")
    s2.add_argument("--concurrency", default="1,8")
    s2.add_argument("--conc-context", type=int, default=1024)
    s2.add_argument("--gen-tokens", type=int, default=256)

    s3 = sub.add_parser("tier3"); common(s3)
    s3.add_argument("--retrieval-context", type=int, default=8192)

    se = sub.add_parser("eval"); common(se)
    se.add_argument("--repeats", type=int, default=2)
    se.add_argument("--domains", default="")
    se.add_argument("--efficiency-target-tps", type=float, default=20.0,
                     help="decode tok/s treated as fully-efficient (100 score); tune to your box")

    sc = sub.add_parser("capacity"); common(sc)
    sc.add_argument("--profiles", default="orchestrator,coding_agent,chat_agent")
    sc.add_argument("--profiles-file", default=None,
                     help="JSON file with extra/custom profiles — see agent_profiles.py schema")
    sc.add_argument("--gpu-utils", default="0.85")
    sc.add_argument("--concurrency", default="1,2,4,8,16,32")
    sc.add_argument("--max-tokens", type=int, default=512)
    sc.add_argument("--skip-restart", action="store_true")
    sc.add_argument("--vllm-cmd", default="vllm serve {model} --gpu-memory-utilization {gpu_util} --port {port}")
    sc.add_argument("--vllm-port", type=int, default=8000)
    sc.add_argument("--ssh-host", default=None)

    args = p.parse_args()
    ctx = build_ctx(args, args.cmd)

    if args.cmd == "tier1":
        run_tier1(ctx, args)
    elif args.cmd == "tier2":
        run_tier2(ctx, args)
    elif args.cmd == "tier3":
        run_tier3(ctx, args)
    elif args.cmd == "eval":
        run_eval(ctx, args)
    elif args.cmd == "capacity":
        run_capacity(ctx, args)

    ctx.flush()

if __name__ == "__main__":
    main()
