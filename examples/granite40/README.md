# Granite 4.0-H Load Balancer (HAProxy)

Load-balances multiple vLLM instances serving `ibm-granite/granite-4.0-h-small` on Gaudi3 HPUs using [HAProxy](https://www.haproxy.org/).

## Files

| File | Purpose |
|---|---|
| `start.sh` | Starts N vLLM instances + HAProxy |
| `stop.sh` | Stops everything |
| `server_command.sh` | vLLM serve command template |

## Prerequisites

- **Gaudi3 HPUs** — one per vLLM instance
- **HAProxy** installed (`apt-get install -y haproxy`)
- **vLLM** with the **vllm-gaudi** plugin installed and working

## Setup

### Baremetal

```bash
# Copy the scripts to a working directory
mkdir -p ~/granite40 && cp examples/granite40/* ~/granite40/
cd ~/granite40

# Install HAProxy if not already present
apt-get install -y haproxy

# Start (8 instances by default)
./start.sh

# Stop
./stop.sh
```

### Docker

Make sure the directory you will use for the scripts is mapped into the container (e.g. `-v /host/path/granite40:/workspace/granite40`).

```bash
# On the host — copy scripts to the mapped directory
mkdir -p /host/path/granite40
cp examples/granite40/* /host/path/granite40/

# Enter the container (adjust image name as needed)
docker exec -it <container_name> bash -c "cd /workspace/granite40 && exec bash"
```

Once inside the container:

```bash
# Install HAProxy if not already present
apt-get install -y haproxy

# Start (8 instances by default)
./start.sh

# Stop
./stop.sh
```

## Configuration

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

Health checks run every 10 seconds against `/v1/models` on each backend. A backend is marked down after 3 consecutive failures and restored after 2 successes. The stats dashboard (`HAPROXY_PORT+1`) shows real-time backend status.
