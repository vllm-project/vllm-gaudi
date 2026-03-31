#!/usr/bin/env bash

HAPROXY_PATH="${HAPROXY_PATH:-/tmp}"
HAPROXY_PID="${HAPROXY_PATH}/haproxy.pid"

echo "Stopping HAProxy..."
if [[ -f "$HAPROXY_PID" ]]; then
    kill "$(cat "$HAPROXY_PID")" 2>/dev/null || true
    rm -f "$HAPROXY_PID"
else
    pkill -f "haproxy.*haproxy.cfg" 2>/dev/null || true
fi

echo "Stopping vLLM instances..."
pkill -f "vllm serve" 2>/dev/null || true

sleep 2
echo "All processes stopped."
