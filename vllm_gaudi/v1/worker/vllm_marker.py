import json
import os
import time


def _now_us() -> int:
    """Return current time in microseconds for Chrome Trace events."""
    try:
        return time.time_ns() // 1000
    except Exception:
        # Fallback if time_ns not available
        return int(time.time() * 1_000_000)


def emit_vllm_chrome_event(msg):
    """
    Emit a Chromium Trace instant event with Message (msg) into TORCH_TRACE structured logs.

    Only emits if TORCH_TRACE environment variable is set (indicating trace logging is active).
    """
    if not os.getenv("TORCH_TRACE"):
        return

    event = {
        "name": "vLLM: " + msg,
        "ph": "i",
        "ts": _now_us(),
        "cat": "marker",
        "args": {},
        "pid": 0,  # TODO(jczaja):make it not hardcoded
        "tid": 0,  # TODO(jczaja):make it not hardcoded
    }

    from torch._logging._internal import trace_structured

    def _metadata():
        return {"chromium_event": {}}

    def _payload():
        return json.dumps(event)

    trace_structured(
        name="chromium_event",
        metadata_fn=_metadata,
        payload_fn=_payload,
    )
    return
