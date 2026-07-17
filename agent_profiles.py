"""
agent_profiles.py — pluggable workload profiles for the capacity benchmark.

The capacity benchmark doesn't care what harness you run (Hermes, OpenClaw,
a custom LangChain/LangGraph loop, whatever) — it just needs a realistic
mix of tasks that harness actually sends to the model. This file ships
three generic profiles that cover the common shapes of agent work:

  orchestrator  — multi-step tool-chain planning (the shape of an
                  autonomous agent loop: plan -> call tool -> call tool ->
                  summarize). This is what Hermes-style orchestration,
                  OpenClaw's agent loop, or any ReAct-style harness sends.
  coding_agent  — code generation + execution-verified correctness. This
                  is what Pi-style coding assistants, OpenClaw's coding
                  mode, or Aider/Cursor-like agents send.
  chat_agent    — casual multi-turn conversational load with light context
                  recall. This is what a personal-assistant-style harness
                  (voice assistants, chat bots) sends.
  hermes        — broader personal-agent harness shape: tool chains that
                  include web_search/read_file (not just calendar/email/
                  reminder), long-context recall inside a multi-turn agent
                  session, and multi-turn preference/state tracking. Used
                  by the `hermes` CLI command to compute a composite
                  Hermes Score (quality + capacity ceiling + responsiveness).

Use whichever profile(s) match your harness's actual traffic shape, or
copy one and edit the task list to match your real prompts — that's more
representative than anything generic. You can also point
--profiles-file at a JSON file (see PROFILE_JSON_SCHEMA below) if you'd
rather not touch Python.
"""

import json, re

# --------------------------------------------------------------------------- #
# Shared tool defs
# --------------------------------------------------------------------------- #

CALENDAR_TOOL = [{"type": "function", "function": {
    "name": "check_calendar", "description": "Check calendar availability.",
    "parameters": {"type": "object", "properties": {"date": {"type": "string"},
                    "after": {"type": "string"}}, "required": ["date"]}}}]
EMAIL_TOOL = [{"type": "function", "function": {
    "name": "draft_email", "description": "Draft an email.",
    "parameters": {"type": "object", "properties": {"to": {"type": "string"},
                    "subject": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "subject", "body"]}}}]
REMINDER_TOOL = [{"type": "function", "function": {
    "name": "create_reminder", "description": "Create a reminder.",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"},
                    "due": {"type": "string"}}, "required": ["title"]}}}]
RUN_TESTS_TOOL = [{"type": "function", "function": {
    "name": "run_tests", "description": "Run the test suite against a code change.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                    "required": ["summary"]}}}]
WEB_SEARCH_TOOL = [{"type": "function", "function": {
    "name": "web_search", "description": "Search the web for current information.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}}}]
READ_FILE_TOOL = [{"type": "function", "function": {
    "name": "read_file", "description": "Read the contents of a local file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}}]

def _has_tool(tool_calls, name):
    return any((tc.get("function", {}) or {}).get("name") == name for tc in tool_calls)

def _exec_check(text, fn_name, test_fn):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    if fn_name not in code:
        return 0.0
    try:
        ns = {}
        exec(code, ns)
        f = ns.get(fn_name)
        return 1.0 if test_fn(f) else 0.3
    except Exception:
        return 0.0

# --------------------------------------------------------------------------- #
# Profile: orchestrator (Hermes / OpenClaw / ReAct-style agent loops)
# --------------------------------------------------------------------------- #

def _grade_orch_full_chain(text, tool_calls):
    steps = ["check_calendar", "draft_email", "create_reminder"]
    hit = sum(1 for s in steps if _has_tool(tool_calls, s))
    return hit / len(steps)

def _grade_orch_conditional(text, tool_calls):
    score = 0.5 if _has_tool(tool_calls, "check_calendar") else 0.0
    score += 0.5 if (_has_tool(tool_calls, "draft_email") or _has_tool(tool_calls, "create_reminder")) else 0.0
    return score

ORCHESTRATOR_TASKS = [
    {"id": "orch_full_chain", "tools": CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": _grade_orch_full_chain,
     "messages": [{"role": "user", "content":
        "Plan and execute: check my calendar for tomorrow. If I'm free after 2pm, draft an email to "
        "the team proposing a 3pm sync, then set a reminder to follow up in 2 days. Use the tools."}]},
    {"id": "orch_conditional", "tools": CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": _grade_orch_conditional,
     "messages": [{"role": "user", "content":
        "Check if I have anything on my calendar this Friday morning. If it's clear, draft a quick "
        "email inviting a candidate for a Friday 10am interview."}]},
    {"id": "orch_single_tool", "tools": CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": lambda t, tc: 1.0 if _has_tool(tc, "create_reminder") else 0.0,
     "messages": [{"role": "user", "content":
        "Set a reminder for me to renew my SSH deploy keys next Monday."}]},
]

# --------------------------------------------------------------------------- #
# Profile: coding_agent (Pi / Aider / Cursor-style / OpenClaw coding mode)
# --------------------------------------------------------------------------- #

CODING_AGENT_TASKS = [
    {"id": "code_merge_intervals", "tools": None,
     "grader": lambda t, tc: _exec_check(
         t, "merge_intervals",
         lambda f: f([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]] and f([]) == []),
     "messages": [{"role": "user", "content":
        "Write a Python function merge_intervals(intervals) that merges overlapping [start,end] "
        "pairs into sorted non-overlapping intervals, handling empty input. Output ONLY one python "
        "code block, no explanation."}]},
    {"id": "code_lru_cache", "tools": None,
     "grader": lambda t, tc: _exec_check(
         t, "LRUCache",
         lambda cls: (lambda c: (c.put(1, 1), c.put(2, 2), c.get(1) == 1, c.put(3, 3),
                                  c.get(2) == -1))(cls(2))[2] and True),
     "messages": [{"role": "user", "content":
        "Implement a class LRUCache(capacity) with get(key)->value-or--1 and put(key,value) methods, "
        "O(1) average time. Output ONLY one python code block."}]},
    {"id": "code_fix_and_verify", "tools": RUN_TESTS_TOOL,
     "grader": lambda t, tc: 0.6 * (1 if "def " in t else 0) + 0.4 * (1 if _has_tool(tc, "run_tests") else 0),
     "messages": [{"role": "user", "content":
        "This function has an off-by-one bug: `def last_n(lst, n): return lst[-n+1:]` — it should "
        "return the last n items. Fix it, then call run_tests to verify."}]},
]

# --------------------------------------------------------------------------- #
# Profile: chat_agent (personal assistant / voice assistant style harness)
# --------------------------------------------------------------------------- #

CHAT_AGENT_TASKS = [
    {"id": "chat_quick_list", "tools": None,
     "grader": lambda t, tc: min(1.0, len(re.findall(r"\d+\.|\n-|\n\*", t)) / 3),
     "messages": [{"role": "user", "content": "Give me 3 quick healthy dinner ideas for tonight."}]},
    {"id": "chat_explain", "tools": None,
     "grader": lambda t, tc: 1.0 if len(t.split()) > 40 and "transformer" in t.lower() else 0.4,
     "messages": [{"role": "user", "content": "Explain in a few sentences how transformer attention works."}]},
    {"id": "chat_context_recall", "tools": None,
     "grader": lambda t, tc: 1.0 if any(k in t.lower() for k in ("dallas", "dfw", "texas")) else 0.3,
     "messages": [
        {"role": "user", "content": "I live in the Dallas-Fort Worth area."},
        {"role": "assistant", "content": "Got it — DFW area, noted."},
        {"role": "user", "content": "Recommend a good weekend day trip from where I live."}]},
]

# --------------------------------------------------------------------------- #
# Profile: hermes (general-purpose personal-agent harness)
#
# Broader than `orchestrator` — mixes tool chains (including web_search and
# read_file, not just calendar/email/reminder), long-context recall *inside*
# a multi-turn agent session, and multi-turn preference/state tracking. This
# is the shape used by the `hermes` CLI command to compute a composite
# Hermes Score (task quality + capacity ceiling + responsiveness).
# --------------------------------------------------------------------------- #

def _tool_args_contain(tool_calls, name, keyword):
    for tc in tool_calls:
        fn = tc.get("function", {}) or {}
        if fn.get("name") == name and keyword.lower() in fn.get("arguments", "").lower():
            return True
    return False

def _build_context_filler(approx_tokens, fact_line, depth=0.5):
    line = ("Meeting notes: the quarterly roadmap review covered infra migration timelines, "
            "on-call rotation updates, and hiring targets across regional teams. ")
    n = max(2, int(approx_tokens * 4) // len(line))
    lines = [line] * n
    lines[int(n * depth)] = fact_line
    return "".join(lines)

_DEPLOY_FACT = "IMPORTANT: the deployment window is Saturday 02:00-04:00 UTC. "
_HERMES_LONG_CTX = _build_context_filler(6000, _DEPLOY_FACT)

def _grade_hermes_full_chain(text, tool_calls):
    steps = ["web_search", "read_file", "run_tests"]
    hit = sum(1 for s in steps if _has_tool(tool_calls, s))
    return hit / len(steps)

def _grade_hermes_search_synth(text, tool_calls):
    searched = _has_tool(tool_calls, "web_search")
    synthesized = bool(re.search(r"\d+\.\d+", text))
    return 0.6 * searched + 0.4 * synthesized

def _grade_hermes_long_ctx_tool(text, tool_calls):
    called = _has_tool(tool_calls, "create_reminder")
    time_mentioned = _tool_args_contain(tool_calls, "create_reminder", "01:") or \
        _tool_args_contain(tool_calls, "create_reminder", "1am") or \
        _tool_args_contain(tool_calls, "create_reminder", "saturday")
    return 0.5 * called + 0.5 * time_mentioned

def _grade_hermes_preference_state(text, tool_calls):
    used_email = _has_tool(tool_calls, "draft_email")
    scheduled = _has_tool(tool_calls, "create_reminder") or _has_tool(tool_calls, "check_calendar")
    return 0.6 * used_email + 0.4 * scheduled

HERMES_TASKS = [
    {"id": "hermes_full_ops_chain",
     "tools": WEB_SEARCH_TOOL + READ_FILE_TOOL + RUN_TESTS_TOOL,
     "grader": _grade_hermes_full_chain,
     "messages": [{"role": "user", "content":
        "Read the deployment checklist file, check the web for any known issues with the "
        "library version we're on, and run the test suite before confirming we're ready to "
        "deploy. Use the tools."}]},
    {"id": "hermes_web_search_synthesize",
     "tools": WEB_SEARCH_TOOL,
     "grader": _grade_hermes_search_synth,
     "messages": [{"role": "user", "content":
        "What's the latest stable Python version? Use web search, then tell me the version "
        "number in your final answer."}]},
    {"id": "hermes_long_ctx_tool_use",
     "tools": CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": _grade_hermes_long_ctx_tool,
     "messages": [{"role": "user", "content":
        _HERMES_LONG_CTX + "\n\nGiven the deployment window mentioned somewhere above, "
        "create a reminder for 1 hour before it starts."}]},
    {"id": "hermes_preference_and_state",
     "tools": CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": _grade_hermes_preference_state,
     "messages": [
        {"role": "user", "content": "My name is Alex and I prefer email over Slack for anything urgent."},
        {"role": "assistant", "content": "Got it, Alex — email for anything urgent."},
        {"role": "user", "content": "I also work in the EU timezone, CET."},
        {"role": "assistant", "content": "Noted — CET."},
        {"role": "user", "content":
         "Something urgent just came up with the payment service. Notify me the way I "
         "prefer, and schedule a follow-up standup for tomorrow morning my time."}]},
    {"id": "hermes_conditional_tool_use",
     "tools": CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": lambda t, tc: (0.5 if _has_tool(tc, "check_calendar") else 0.0) +
                              (0.5 if (_has_tool(tc, "draft_email") or _has_tool(tc, "create_reminder")) else 0.0),
     "messages": [{"role": "user", "content":
        "Check if I have anything on my calendar Thursday afternoon. If it's clear, draft an "
        "email inviting the on-call rotation for a Thursday 3pm handoff review."}]},
    {"id": "hermes_simple_ask",
     "tools": None,
     "grader": lambda t, tc: 1.0 if len(t.split()) > 30 else 0.4,
     "messages": [{"role": "user", "content":
        "Summarize the tradeoffs between synchronous and asynchronous agent tool calls in a "
        "few sentences."}]},
    {"id": "hermes_single_tool",
     "tools": WEB_SEARCH_TOOL + READ_FILE_TOOL + RUN_TESTS_TOOL + CALENDAR_TOOL + EMAIL_TOOL + REMINDER_TOOL,
     "grader": lambda t, tc: 1.0 if _has_tool(tc, "read_file") else 0.0,
     "messages": [{"role": "user", "content": "Read config.yaml and tell me what's in it."}]},
]

PROFILES = {
    "orchestrator": ORCHESTRATOR_TASKS,
    "coding_agent": CODING_AGENT_TASKS,
    "chat_agent": CHAT_AGENT_TASKS,
    "hermes": HERMES_TASKS,
}

# --------------------------------------------------------------------------- #
# Optional: load extra/replacement profiles from a JSON file so people who
# don't want to touch Python can still plug in their harness's real tasks.
#
# PROFILE_JSON_SCHEMA:
# {
#   "my_hermes_profile": [
#     {
#       "id": "task_1",
#       "messages": [{"role": "user", "content": "..."}],
#       "tools": null,                      // or an OpenAI-style tools array
#       "grader_type": "keyword",           // "keyword" | "tool_sequence" | "json_valid" | "code_exec"
#       "grader_args": {"keywords": ["dallas", "texas"]}
#       // for grader_type "tool_sequence": {"tools_in_order": ["search_flights","book_flight"]}
#       // for grader_type "json_valid": {"required_fields": ["name","age"]}
#       // for grader_type "code_exec": {"function_name": "merge_intervals",
#       //     "test_cases": [{"args": [[[1,3],[2,6]]], "expected": [[1,6]]}]}
#       //   — extracts the reply's python code block, execs it, calls function_name with each
#       //   test case's "args"/"kwargs", and scores the fraction of cases whose return value
#       //   equals "expected". Lets generated coding tasks be graded by actually running them,
#       //   not just keyword-matched.
#     }
#   ]
# }
# --------------------------------------------------------------------------- #

def _code_exec_check(text, fn_name, test_cases):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    if not fn_name or fn_name not in code:
        return 0.0
    try:
        ns = {}
        exec(code, ns)
        f = ns.get(fn_name)
    except Exception:
        return 0.0
    if f is None:
        return 0.0
    if not test_cases:
        return 0.3
    passed = 0
    for case in test_cases:
        try:
            result = f(*case.get("args", []), **case.get("kwargs", {}))
            if result == case.get("expected"):
                passed += 1
        except Exception:
            pass
    return passed / len(test_cases)

def _grader_from_json_spec(spec):
    gtype = spec.get("grader_type", "keyword")
    args = spec.get("grader_args", {})
    if gtype == "keyword":
        kws = [k.lower() for k in args.get("keywords", [])]
        def g(text, tool_calls):
            return 1.0 if any(k in text.lower() for k in kws) else 0.0
        return g
    if gtype == "tool_sequence":
        order = args.get("tools_in_order", [])
        def g(text, tool_calls):
            if not order:
                return 0.0
            hit = sum(1 for name in order if _has_tool(tool_calls, name))
            return hit / len(order)
        return g
    if gtype == "json_valid":
        required = args.get("required_fields", [])
        def g(text, tool_calls):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return 0.0
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return 0.2
            if not required:
                return 1.0
            return sum(1 for k in required if k in obj) / len(required)
        return g
    if gtype == "code_exec":
        fn_name = args.get("function_name", "")
        cases = args.get("test_cases", [])
        def g(text, tool_calls):
            return _code_exec_check(text, fn_name, cases)
        return g
    raise ValueError(f"unknown grader_type: {gtype}")

def load_profiles_from_json(path):
    with open(path) as f:
        raw = json.load(f)
    loaded = {}
    for name, tasks in raw.items():
        built = []
        for t in tasks:
            built.append({
                "id": t["id"], "messages": t["messages"], "tools": t.get("tools"),
                "grader": _grader_from_json_spec(t),
            })
        loaded[name] = built
    return loaded

def get_profiles(names, profiles_file=None):
    all_profiles = dict(PROFILES)
    if profiles_file:
        all_profiles.update(load_profiles_from_json(profiles_file))
    out = {}
    for n in names:
        if n not in all_profiles:
            raise ValueError(f"unknown profile '{n}'. Available: {list(all_profiles.keys())}")
        out[n] = all_profiles[n]
    return out
