#!/usr/bin/env bash
set -euo pipefail

# Start the Python inference server and Rust client without Nsight Systems.
# Useful to isolate profiler interactions and to test torch.compile behavior.
#
# Example (UDS):
#   bin/run_infer_pair.sh \
#     --init inits/v1_pretrained_50m \
#     --uds /tmp/2048_infer.sock \
#     --device cuda \
#     --client-config config/inference/top-score.toml \
#     --compile-mode default --release

INIT_DIR=""
DEVICE="cuda"
UDS=""
TCP=""
CLIENT_CFG="config/inference/top-score.toml"
COMPILE_MODE="default"   # default|max-autotune|reduce-overhead|none
RELEASE=0
SERVER_TIMEOUT=120
OUTDIR=""

die() { echo "[run_pair] $*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init) INIT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --uds) UDS="$2"; shift 2 ;;
    --tcp) TCP="$2"; shift 2 ;;
    --client-config) CLIENT_CFG="$2"; shift 2 ;;
    --compile-mode) COMPILE_MODE="$2"; shift 2 ;;
    --server-timeout) SERVER_TIMEOUT="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --release) RELEASE=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0" | sed -n '1,80p'
      exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

[[ -n "$INIT_DIR" ]] || die "--init is required"
if [[ -z "$UDS" && -z "$TCP" ]]; then
  die "Specify one of --uds or --tcp"
fi
if [[ -n "$UDS" && -n "$TCP" ]]; then
  die "Specify only one of --uds or --tcp"
fi

need uv
need cargo

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-"runs/${TS}"}
mkdir -p "$OUTDIR"

SRV_LOG="$OUTDIR/server.out.log"
echo "[run_pair] writing logs under: $OUTDIR"
echo "[run_pair] server compile-mode=$COMPILE_MODE"

# Build server address flags
SRV_ADDR_FLAG=()
if [[ -n "$UDS" ]]; then
  if [[ "$UDS" != unix:* ]]; then
    SRV_ADDR_FLAG=(--uds "unix:$UDS")
  else
    SRV_ADDR_FLAG=(--uds "$UDS")
  fi
else
  SRV_ADDR_FLAG=(--tcp "$TCP")
fi

# If using UDS, remove stale socket
if [[ -n "$UDS" ]]; then
  SOCK_PATH=${UDS#unix:}
  if [[ -S "$SOCK_PATH" ]]; then
    echo "[run_pair] removing stale UDS socket: $SOCK_PATH"
    rm -f -- "$SOCK_PATH" || true
  fi
fi

# Start server (unbuffered output)
echo "[run_pair] starting server... (log: $SRV_LOG)"
set -x
PYTHONUNBUFFERED=1 uv run infer-2048 \
  --init "$INIT_DIR" "${SRV_ADDR_FLAG[@]}" --device "$DEVICE" \
  --compile-mode "$COMPILE_MODE" \
  >"$SRV_LOG" 2>&1 &
SRV_PID=$!
set +x

# Wait for bind
if [[ -n "$UDS" ]]; then
  SOCK_PATH=${UDS#unix:}
  echo "[run_pair] waiting for UDS socket: $SOCK_PATH (timeout=${SERVER_TIMEOUT}s)"
  READY=0; T0=$(date +%s)
  while true; do
    if [[ -S "$SOCK_PATH" ]]; then READY=1; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[run_pair] server exited. tail of log:" >&2
      tail -n 200 "$SRV_LOG" >&2 || true
      die "server died before binding"
    fi
    NOW=$(date +%s); (( NOW - T0 >= SERVER_TIMEOUT )) && break
    sleep 0.1
  done
  if [[ $READY -ne 1 ]]; then
    echo "[run_pair] timeout waiting for socket. tail of log:" >&2
    tail -n 200 "$SRV_LOG" >&2 || true
    die "timed out waiting for UDS socket: $SOCK_PATH"
  fi
else
  sleep 1
fi

# Probe gRPC readiness (ensure proper scheme for UDS)
if [[ -n "$UDS" ]]; then
  if [[ "$UDS" != unix:* ]]; then
    TARGET="unix:${UDS}"
  else
    TARGET="$UDS"
  fi
else
  TARGET="$TCP"
fi
echo "[run_pair] probing server readiness at ${TARGET}..."
set -x
uv run python - "$TARGET" "$SERVER_TIMEOUT" << 'PY'
import sys, grpc
target = sys.argv[1]
timeout = float(sys.argv[2])
ch = grpc.insecure_channel(target)
fut = grpc.channel_ready_future(ch)
fut.result(timeout=timeout)
print("READY")
PY
set +x

# Start client
# Build a representative throughput config for the client (overlay)
CLIENT_CFG_OUT="$OUTDIR/client-config.toml"
METRICS_FILE="$OUTDIR/client-metrics.jsonl"
echo "[run_pair] writing client config: $CLIENT_CFG_OUT (metrics: $METRICS_FILE)"
{
  echo "num_seeds = 1024"
  echo "max_retries = 3"
  echo "max_concurrent_games = 512"
  echo ""
  echo "[sampling]"
  echo "strategy = \"Argmax\""
  echo ""
  echo "[orchestrator.connection]"
  if [[ -n "$UDS" ]]; then
    echo "uds_path = \"${SOCK_PATH}\""
  else
    echo "tcp_addr = \"${TCP}\""
  fi
  echo ""
  echo "[orchestrator.batch]"
  echo "flush_us = 200"
  echo "target_batch = 1024"
  echo "max_batch = 1024"
  echo "inflight_batches = 3"
  echo "per_game_inflight = 32"
  echo "queue_cap = 65536"
  echo "metrics_file = \"${METRICS_FILE}\""
  echo "metrics_interval_s = 5.0"
} > "$CLIENT_CFG_OUT"

echo "[run_pair] starting client..."
set -x
if [[ $RELEASE -eq 1 ]]; then
  cargo run -p game-engine --release -- --config "$CLIENT_CFG_OUT"
else
  cargo run -p game-engine -- --config "$CLIENT_CFG_OUT"
fi
set +x

echo "[run_pair] shutting down server..."
kill "$SRV_PID" 2>/dev/null || true
wait "$SRV_PID" 2>/dev/null || true
echo "[run_pair] done. Logs: $SRV_LOG"
