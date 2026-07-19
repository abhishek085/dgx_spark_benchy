#!/usr/bin/env python3
"""
generate_data.py — use a big, high-quality "generator" model (e.g. a natively
NVFP4-trained model like nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) to
produce a larger, more diverse task pool for agent_profiles.py's capacity
profiles than the ~3-7 hand-written tasks each ships with.

Why this matters: the `capacity` and `hermes` commands cycle through a
profile's task list as concurrency climbs (`tasks[i % len(tasks)]`) — with
only a handful of tasks, a high-concurrency sweep just replays the same few
prompts dozens of times, which can make vLLM's prefix caching look faster
than it would on real, varied traffic. A larger generated pool fixes that.

This doesn't manage vLLM lifecycle — load your generator model yourself
first (the same way you would for any other command here), then point this
at its endpoint. It writes a --profiles-file-compatible JSON file (see
agent_profiles.py's PROFILE_JSON_SCHEMA) that quickbench.sh / capacity /
hermes can all consume directly.

Usage:
  ./generate_data.py --endpoint http://localhost:8000/v1 --model <generator-model> \\
      --profiles orchestrator,coding_agent,chat_agent,hermes --count 30 \\
      --out profiles/generated.json
"""

import argparse, json, os, re, sys
import spark_bench_plus as sbp

HERE = os.path.dirname(os.path.abspath(__file__))

TOOL_NAMES = [
    "check_calendar(date, after)", "draft_email(to, subject, body)",
    "create_reminder(title, due)", "run_tests(summary)", "web_search(query)",
    "read_file(path)",
]

PROFILE_GUIDANCE = {
    "orchestrator": (
        "This profile simulates a ReAct-style autonomous agent doing multi-step tool-chain "
        "planning: calendar checks, drafting emails, setting reminders, conditional branches "
        "based on tool results."
    ),
    "coding_agent": (
        "This profile simulates a coding assistant. Ask for a self-contained Python function or "
        "small class with clear, testable behavior. Always use grader_type \"code_exec\" with "
        "2-4 test cases per task, and tell the model in the task prompt to output ONLY one "
        "python code block, no explanation."
    ),
    "chat_agent": (
        "This profile simulates casual multi-turn conversational load — personal-assistant "
        "style requests, light context recall, no tools needed (tools: null)."
    ),
    "hermes": (
        "This profile simulates a general-purpose personal agent harness handling real user "
        "requests: multi-step tool chains mixing calendar/email/reminder/web_search/read_file/"
        "run_tests tools, long-context recall inside a multi-turn conversation, and remembering "
        "stated user preferences across turns. Some tasks should have 3-5 message multi-turn "
        "histories in \"messages\", not just a single user turn."
    ),
}

SCHEMA_INSTRUCTIONS = """Do not think step by step. Do not plan, draft, or describe the tasks in \
prose before writing them. Do not explain your approach. Your response must begin with the \
character '[' and contain NOTHING else but a single valid JSON array — no markdown fences, no \
commentary before or after, no draft version followed by a final version.

Output exactly {count} task objects for a benchmark profile named "{profile}". Each object must \
have this exact shape:

{{
  "id": "<short_unique_snake_case_id>",
  "messages": [{{"role": "user", "content": "<realistic task request>"}}],
  "tools": <null, or an OpenAI-style tools array matching one of the allowed tool names below>,
  "grader_type": "<keyword|tool_sequence|json_valid|code_exec>",
  "grader_args": {{ ... }}
}}

grader_args by grader_type:
  keyword:       {{"keywords": ["word1","word2"]}}  — passes if ANY keyword appears in the reply
  tool_sequence: {{"tools_in_order": ["tool_a","tool_b"]}} — partial credit per tool actually called
  json_valid:    {{"required_fields": ["field1","field2"]}} — reply must contain a JSON object with these fields
  code_exec:     {{"function_name": "fn", "test_cases": [{{"args": [...], "expected": ...}}]}} — the reply's
                 python code block is executed and fn is called against each test case

Allowed tool names for "tools" (use the exact OpenAI function-calling shape; omit "tools" entirely,
i.e. set it to null, for tasks that need no tools): {tool_names}

{profile_guidance}

Make every task meaningfully different from the others — vary the scenario, phrasing, and
difficulty. Do not repeat a task idea. Output nothing but the JSON array.
"""


def _strip_reasoning(text):
    """Reasoning models (e.g. Nemotron Super) prepend a <think>/<reasoning> trace before the
    actual answer. That trace often contains its own stray '[' / ']' characters (citations,
    example arrays, etc.), which breaks a naive greedy bracket-matching regex — it grabs from
    the first stray '[' in the reasoning all the way to the real closing ']', capturing garbage
    in between. Strip known reasoning-block tags first so extraction only sees the answer."""
    return re.sub(r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
                   "", text, flags=re.DOTALL | re.IGNORECASE)


def _iter_object_spans(text):
    """Yield each top-level, string-aware, balanced '{...}' span in text (nested braces inside
    a span, e.g. a task's "tools" definitions, don't end it early)."""
    depth, start, in_str, esc = 0, None, False, False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def _extract_json_array(text):
    """Extract individual task objects rather than requiring the whole array to parse as one
    blob. Observed directly: Nemotron Super's output is very often *almost* valid — one stray
    bracket or similar glitch somewhere — and requiring strict json.loads() on the entire array
    throws away an otherwise-good batch over a single local defect. Pulling out each balanced
    '{...}' span and parsing those independently recovers every good task and just drops the
    one bad object, instead of failing the whole batch."""
    text = _strip_reasoning(text)
    tasks = []
    for candidate in _iter_object_spans(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "messages" in obj:
            tasks.append(obj)
    if not tasks:
        raise ValueError(f"no valid task objects found in generator output "
                          f"(saw {len(text)} chars after stripping reasoning tags)")
    return tasks


def generate_profile(endpoint, model, profile, count, max_tokens, timeout, id_offset=0):
    prompt = SCHEMA_INSTRUCTIONS.format(
        count=count, profile=profile,
        tool_names=", ".join(TOOL_NAMES),
        profile_guidance=PROFILE_GUIDANCE.get(profile, ""),
    )
    r = sbp.chat_stream(endpoint, model, [{"role": "user", "content": prompt}],
                         max_tokens, temperature=0.9, timeout=timeout)
    if r["error"]:
        raise RuntimeError(f"generator call failed: {r['error']}")
    try:
        raw_tasks = _extract_json_array(r["text"])
    except ValueError as e:
        debug_path = os.path.join(HERE, f"_debug_gen_{profile}.txt")
        with open(debug_path, "w") as f:
            f.write(r["text"])
        preview = _strip_reasoning(r["text"]).strip()
        preview = (preview[:300] + " ...(truncated)... " + preview[-300:]) if len(preview) > 700 else preview
        raise ValueError(f"{e}\n  raw response saved to {debug_path}\n  preview: {preview!r}") from None

    valid = []
    for i, t in enumerate(raw_tasks):
        if not isinstance(t, dict) or "messages" not in t:
            print(f"[generate] {profile}: skipping malformed task at index {i}", file=sys.stderr)
            continue
        t["id"] = f"{profile}_gen_{id_offset + len(valid)}"
        valid.append(t)
    return valid


def generate_profile_batched(endpoint, model, profile, total_count, batch_size, max_tokens, timeout):
    """Some reasoning models (e.g. Nemotron Super, observed directly) draft every task out in
    verbose bullet-point prose before ever emitting the real JSON when asked for a large batch
    (e.g. 30) in one call — by the token budget's end they're still drafting task 3, and the
    array never closes. Requesting a handful at a time keeps each call's rambling bounded and
    tractable, at the cost of more round trips."""
    collected = []
    max_attempts = -(-total_count // batch_size) * 2 + 2  # ceil(total/batch)*2 + a couple spare
    attempts = 0
    while len(collected) < total_count and attempts < max_attempts:
        attempts += 1
        remaining = total_count - len(collected)
        this_batch = min(batch_size, remaining)
        print(f"[generate] {profile}: batch of {this_batch} (have {len(collected)}/{total_count}) ...")
        try:
            batch = generate_profile(endpoint, model, profile, this_batch, max_tokens, timeout,
                                      id_offset=len(collected))
        except Exception as e:
            print(f"[generate] {profile}: batch failed ({e}) — retrying", file=sys.stderr)
            continue
        collected.extend(batch)
    return collected


def main():
    ap = argparse.ArgumentParser(description="Generate a diverse task pool using a big generator model")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--profiles", default="orchestrator,coding_agent,chat_agent,hermes")
    ap.add_argument("--count", type=int, default=30, help="tasks to generate per profile")
    ap.add_argument("--batch-size", type=int, default=5,
                     help="tasks requested per call — kept small because reasoning models can "
                          "ramble through a draft of each task before the real JSON, blowing the "
                          "token budget on a large single request")
    ap.add_argument("--max-tokens", type=int, default=8000,
                     help="generation budget per batch call — raise this if batches keep getting "
                          "truncated mid-JSON even at the default --batch-size")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=os.path.join(HERE, "profiles", "generated.json"))
    args = ap.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    out = {}
    for profile in profiles:
        print(f"[generate] {profile}: requesting {args.count} tasks from {args.model} "
              f"in batches of {args.batch_size} ...")
        tasks = generate_profile_batched(args.endpoint, args.model, profile, args.count,
                                          args.batch_size, args.max_tokens, args.timeout)
        if not tasks:
            print(f"[generate] FAILED for {profile}: no tasks produced after retries", file=sys.stderr)
            continue
        out[profile] = tasks
        print(f"[generate] {profile}: got {len(tasks)} usable tasks")

    if not out:
        print("no profiles generated successfully — nothing written.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    total = sum(len(v) for v in out.values())
    print(f"\n-> wrote {total} tasks across {len(out)} profile(s) to {args.out}")
    print(f"   use with: --profiles-file {args.out}")


if __name__ == "__main__":
    main()
