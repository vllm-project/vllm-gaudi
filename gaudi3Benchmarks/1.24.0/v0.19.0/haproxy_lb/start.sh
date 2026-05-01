#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
N_INSTANCES="${1:-8}"          # number of vLLM instances (default: 8)
BASE_PORT=30001               # first vLLM backend port
HAPROXY_PORT=30360            # HAProxy frontend port
HAPROXY_STATS_PORT=30361      # HAProxy stats page port
API_KEY="granite4.0h-g3key"
SERVER_CMD="./server_command.sh"
LOG_DIR="/root/granite_lb_haproxy/logs"
HAPROXY_CFG="/root/granite_lb_haproxy/haproxy.cfg"
HAPROXY_PID="/root/granite_lb_haproxy/haproxy.pid"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$LOG_DIR" "$(dirname "$HAPROXY_CFG")"

# ── Validate input ─────────────────────────────────────────────────
if ! [[ "$N_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_INSTANCES must be a positive integer, got '$N_INSTANCES'"
    exit 1
fi

echo "=== Configuration ==="
echo "  Instances:   $N_INSTANCES"
echo "  Ports:       $BASE_PORT - $((BASE_PORT + N_INSTANCES - 1))"
echo "  HAProxy:     http://0.0.0.0:$HAPROXY_PORT"
echo "  Stats:       http://0.0.0.0:$HAPROXY_STATS_PORT/stats"
echo ""

# ── Stop existing processes ────────────────────────────────────────
echo "=== Stopping any existing vllm/haproxy processes ==="
pkill -f "vllm serve" 2>/dev/null || true
if [[ -f "$HAPROXY_PID" ]]; then
    kill "$(cat "$HAPROXY_PID")" 2>/dev/null || true
    rm -f "$HAPROXY_PID"
fi
pkill -f "haproxy.*haproxy.cfg" 2>/dev/null || true
sleep 2

# ── Start vLLM instances ──────────────────────────────────────────
echo "=== Starting $N_INSTANCES vLLM instances ==="
for i in $(seq 0 $((N_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    echo "Starting vLLM instance $i on HPU $i, port $PORT..."

    INSTANCE_CMD=$(sed "s/--port [0-9]*/--port ${PORT}/" "$SCRIPT_DIR/server_command.sh")

    HABANA_VISIBLE_DEVICES=$i \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    VLLM_EXPONENTIAL_BUCKETING=true \
    nohup bash -c "$INSTANCE_CMD" \
        > "$LOG_DIR/vllm_instance_${i}.log" 2>&1 &
    echo "  PID: $!"
done

# ── Wait for vLLM backends ────────────────────────────────────────
echo ""
echo "=== Waiting for all vLLM backends to be ready ==="
for i in $(seq 0 $((N_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    echo -n "Waiting for vLLM instance $i (port $PORT)..."
    timeout=1200
    elapsed=0
    while true; do
        code=$(curl -sS -o /dev/null -w "%{http_code}" \
            -H "Authorization: Bearer ${API_KEY}" \
            "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo "000")
        if [[ "$code" == "200" ]]; then
            echo " READY"
            break
        fi
        if (( elapsed >= timeout )); then
            echo " TIMEOUT (${timeout}s) - check $LOG_DIR/vllm_instance_${i}.log"
            exit 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        echo -n "."
    done
done

# ── Generate HAProxy config ───────────────────────────────────────
echo ""
echo "=== Generating HAProxy configuration ==="
cat > "$HAPROXY_CFG" <<EOF
global
    log stdout format raw local0
    maxconn 4096
    pidfile ${HAPROXY_PID}

defaults
    log     global
    mode    http
    option  httplog
    option  dontlognull
    timeout connect 10s
    timeout client  360s
    timeout server  360s
    retries 2
    option  redispatch

frontend vllm_frontend
    bind *:${HAPROXY_PORT}
    default_backend vllm_backends

backend vllm_backends
    balance leastconn
EOF

for i in $(seq 0 $((N_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    echo "    server vllm_${i} 127.0.0.1:${PORT}" >> "$HAPROXY_CFG"
done

cat >> "$HAPROXY_CFG" <<EOF

listen stats
    bind *:${HAPROXY_STATS_PORT}
    stats enable
    stats uri /stats
    stats refresh 5s
    stats show-legends
EOF

echo "Config written to $HAPROXY_CFG"

# ── Start HAProxy ─────────────────────────────────────────────────
echo ""
echo "=== Starting HAProxy on port $HAPROXY_PORT ==="
haproxy -f "$HAPROXY_CFG" -D -p "$HAPROXY_PID"
echo "HAProxy PID: $(cat "$HAPROXY_PID")"

sleep 2

# ── One-time health check through HAProxy ─────────────────────────
echo ""
echo "=== Running one-time health check through HAProxy ==="
code=$(curl -sS -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${API_KEY}" \
    "http://127.0.0.1:${HAPROXY_PORT}/v1/models" 2>/dev/null || echo "000")
if [[ "$code" == "200" ]]; then
    echo "Health check PASSED (HTTP $code)"
else
    echo "WARNING: Health check returned HTTP $code — verify backends are up"
fi

echo ""
echo "=== Stack is running ==="
echo "HAProxy endpoint:  http://localhost:${HAPROXY_PORT}"
echo "HAProxy stats:     http://localhost:${HAPROXY_STATS_PORT}/stats"
echo ""
echo "Usage:"
echo "  curl -sS http://localhost:${HAPROXY_PORT}/v1/models -H \"Authorization: Bearer ${API_KEY}\""
echo ""
echo "vLLM instance logs: $LOG_DIR/vllm_instance_*.log"
echo "HAProxy config:     $HAPROXY_CFG"
