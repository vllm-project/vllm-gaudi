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

Set `HAPROXY_PATH` to control where HAProxy config, PID file, and vLLM logs are stored. Defaults to `/tmp` if not set.

```bash
export HAPROXY_PATH=/my/custom/path
./start.sh
```

## Endpoints

| Endpoint | Default Port | Description |
|---|---|---|
| API | `30360` | HAProxy frontend — drop-in replacement for the LiteLLM proxy |
| Stats | `30361` | HAProxy stats dashboard (auto-refresh) |

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
