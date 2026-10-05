"""Live probe driver: loads the lintai_probe plugin in an isolated HERMES_HOME
and exercises the hook + middleware channels through Hermes' REAL code paths.

No LLM keys used. Everything runs in-process.
"""
import json
import os
import sys

PROBE = os.path.expanduser("~/workspace/hermes-probe")
os.environ["HERMES_HOME"] = os.path.join(PROBE, "home")
os.environ["HERMES_PLUGINS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(PROBE, "hermes-agent"))

# Fresh log
log_path = os.path.join(PROBE, "hook_log.jsonl")
if os.path.exists(log_path):
    os.remove(log_path)

from hermes_cli.plugins import (  # noqa: E402
    discover_plugins, invoke_hook, has_hook, has_middleware, iter_hook_callbacks,
)

discover_plugins()
print("has pre_llm_call hook:", has_hook("pre_llm_call"))
print("has post_llm_call hook:", has_hook("post_llm_call"))
print("has llm_request middleware:", has_middleware("llm_request"))
print("pre_llm_call callbacks:", len(iter_hook_callbacks("pre_llm_call")))

# --- 1. pre_llm_call injection through the REAL collector ---
from types import SimpleNamespace  # noqa: E402
from agent.turn_context import _collect_pre_llm_call_context  # noqa: E402

agent = SimpleNamespace(
    session_id="sess-probe-1", model="probe-model", platform="cli",
    _persist_disabled=False, _parent_session_id="", _user_id="u1",
)
injected = _collect_pre_llm_call_context(
    agent,
    effective_task_id="task-1",
    turn_id="turn-1",
    original_user_message="hello world",
    messages=[{"role": "user", "content": "hello world"}],
    conversation_history=[],
)
print("PRE_LLM_CALL INJECTED:", repr(injected[:160]))

# --- 2. llm_request middleware rewrite through the REAL chain ---
from hermes_cli.middleware import apply_llm_request_middleware  # noqa: E402

req = {"model": "probe-model",
       "messages": [{"role": "user", "content": "hello"}]}
res = apply_llm_request_middleware(
    req, task_id="task-1", turn_id="turn-1", api_request_id="api-1",
    session_id="sess-probe-1", platform="cli", model="probe-model",
    provider="probe", base_url="", api_mode="", api_call_count=1,
)
print("MW CHANGED:", res.changed)
print("MW MESSAGES:", json.dumps(res.payload["messages"])[:160])

# --- 3. Fire capture/lifecycle hooks with realistic payloads ---
invoke_hook(
    "post_llm_call", session_id="sess-probe-1", task_id="task-1", turn_id="turn-1",
    user_message="hello world", assistant_response="hi there",
    conversation_history=[{"role": "user", "content": "hello world"},
                          {"role": "assistant", "content": "hi there"}],
    model="probe-model", platform="cli",
)
invoke_hook(
    "on_session_end", session_id="sess-probe-1", task_id="task-1", turn_id="turn-1",
    completed=True, failed=False, interrupted=False,
    turn_exit_reason="text_response(stop)", model="probe-model", platform="cli",
)
invoke_hook("on_session_finalize", session_id="sess-probe-1",
            platform="cli", reason="session_boundary")
invoke_hook("on_session_reset", session_id="sess-probe-1",
            platform="cli", reason="new_session")
invoke_hook("on_session_start", session_id="sess-probe-1",
            model="probe-model", platform="cli")
invoke_hook(
    "post_tool_call", function_name="read_file",
    function_args={"path": "/tmp/x"}, result="file contents here",
    session_id="sess-probe-1", task_id="task-1", turn_id="turn-1",
    tool_call_id="call-1", duration_ms=12, status="ok",
    error_type=None, error_message=None, middleware_trace=[],
)
invoke_hook("transform_llm_output", session_id="sess-probe-1",
            response="hi there", model="probe-model")
print("HOOKS FIRED OK")
