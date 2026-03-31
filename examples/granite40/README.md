# Granite 4.0-H Load Balancer (HAProxy)

Load-balances multiple vLLM instances serving `ibm-granite/granite-4.0-h-small` on Gaudi3 HPUs using [HAProxy](https://www.haproxy.org/).

## Files

| File | Purpose |
|---|---|
| `start.sh` | Starts N vLLM instances + HAProxy |
| `stop.sh` | Stops everything |
| `server_command.sh` | vLLM serve command template |

## Usage

```bash
# Start with 8 instances (default)
./start.sh

# Start with a custom number of instances
./start.sh 4

# Stop all
./stop.sh
```

### Configuration

Set these environment variables before running `start.sh` to customize behavior:

| Variable | Default | Description |
|---|---|---|
| `HAPROXY_PATH` | `/tmp` | Directory for HAProxy config, PID file, and vLLM logs |
| `HAPROXY_PORT` | `30360` | Client-facing API port (HAProxy frontend) |
| `BASE_PORT` | `30001` | First vLLM backend port; instances use `BASE_PORT`, `BASE_PORT+1`, … |

The HAProxy stats dashboard is always at `HAPROXY_PORT + 1` (default `30361`).

```bash
export HAPROXY_PATH=/my/custom/path
export HAPROXY_PORT=8080
export BASE_PORT=9001
./start.sh
```

## Endpoints

| Endpoint | Default Port | Description |
|---|---|---|
| API | `HAPROXY_PORT` (30360) | HAProxy frontend — drop-in replacement for the LiteLLM proxy |
| Stats | `HAPROXY_PORT+1` (30361) | HAProxy stats dashboard (auto-refresh) |
| Backends | `BASE_PORT` … `BASE_PORT+N-1` (30001–30008) | Individual vLLM instances (one per HPU) |

### Example request

```bash
curl -sS http://localhost:30360/v1/models \
  -H "Authorization: Bearer granite4.0h-g3key"
```

## Load Balancing

HAProxy uses **leastconn** balancing — each new request goes to the backend with the fewest active connections, similar to the `least-busy` strategy in LiteLLM.

Health checks run every 10 seconds against `/v1/models` on each backend. A backend is marked down after 3 consecutive failures and restored after 2 successes.

## Prerequisites

- HAProxy installed (`apt-get install haproxy` or `yum install haproxy`)
- vLLM with Gaudi support
- One Gaudi3 HPU per instance
