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

PROFILES = {
    "orchestrator": ORCHESTRATOR_TASKS,
    "coding_agent": CODING_AGENT_TASKS,
    "chat_agent": CHAT_AGENT_TASKS,
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
#       "grader_type": "keyword",           // "keyword" | "tool_sequence" | "json_valid"
#       "grader_args": {"keywords": ["dallas", "texas"]}
#       // for grader_type "tool_sequence": {"tools_in_order": ["search_flights","book_flight"]}
#       // for grader_type "json_valid": {"required_fields": ["name","age"]}
#     }
#   ]
# }
# --------------------------------------------------------------------------- #

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
