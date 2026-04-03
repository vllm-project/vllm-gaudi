#!/usr/bin/env bash
HAPROXY_PID="/root/granite_lb_haproxy/haproxy.pid"

# Detect OS and define a kill-by-pattern function
kill_by_pattern() {
    local pattern="$1"
    if command -v pkill &>/dev/null; then
        pkill -f "$pattern" 2>/dev/null || true
    else
        # Fallback for RHEL/systems without pkill (procps-ng not installed)
        ps -eo pid,cmd | grep -E "$pattern" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null || true
    fi
}

echo "Stopping HAProxy..."
if [[ -f "$HAPROXY_PID" ]]; then
    kill "$(cat "$HAPROXY_PID")" 2>/dev/null || true
    rm -f "$HAPROXY_PID"
else
    kill_by_pattern "haproxy.*haproxy.cfg"
fi

echo "Stopping vLLM instances..."
kill_by_pattern "vllm serve"

sleep 2
echo "All processes stopped."
