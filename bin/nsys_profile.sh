#!/usr/bin/env bash
set -euo pipefail

# Simple orchestrator to record Nsight Systems traces for both the Python
# inference server and the Rust client (game-engine) in separate files.
#
# Outputs two .nsys-rep files under an output directory (default: profiles/<ts>):
#   - server.nsys-rep: CUDA + CPU + NVTX + OS trace for the server process
#   - client.nsys-rep: CPU/OS + NVTX trace for the Rust orchestrator
#
# Example (UDS):
#   bin/nsys_profile.sh \
#     --init inits/v1_50m \
#     --uds /tmp/2048_infer.sock \
#     --device cuda \
#     --client-config config/inference/top-score.toml
#
# Example (TCP):
#   bin/nsys_profile.sh \
#     --init inits/v1_50m \
#     --tcp 127.0.0.1:50051 \
#     --device cuda \
#     --client-config config/inference/top-score.toml

INIT_DIR=""
DEVICE="cuda"
UDS=""
TCP=""
CLIENT_CFG="config/inference/top-score.toml"
OUTDIR=""
RELEASE=0
SLEEP_SECS=2
SERVER_READY_TIMEOUT=120
SERVER_COMPILE_MODE="none"
SERVER_WARMUP_SIZES=""   # no warmup by default
SERVER_DYNAMIC_BATCH=0    # no dynamic-batch warmup

die() { echo "[nsys_profile] $*" >&2; exit 2; }

need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --init) INIT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --uds) UDS="$2"; shift 2 ;;
    --tcp) TCP="$2"; shift 2 ;;
    --client-config) CLIENT_CFG="$2"; shift 2 ;;
    --sleep) SLEEP_SECS="$2"; shift 2 ;;
    --server-compile-mode) SERVER_COMPILE_MODE="$2"; shift 2 ;;
    --server-warmup) SERVER_WARMUP_SIZES="$2"; shift 2 ;;
    --server-dynamic-batch) SERVER_DYNAMIC_BATCH=1; shift ;;
    --server-timeout) SERVER_READY_TIMEOUT="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --release) RELEASE=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0" | sed -n '1,60p'
      exit 0
      ;;
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

need nsys
need uv
need cargo

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=${OUTDIR:-"profiles/${TS}"}
mkdir -p "$OUTDIR"

SRV_OUT="$OUTDIR/server"
CLI_OUT="$OUTDIR/client"
SRV_LOG="$OUTDIR/server.out.log"

echo "[nsys_profile] writing traces under: $OUTDIR"
echo "[nsys_profile] server compile-mode=$SERVER_COMPILE_MODE, dynamic-batch=$SERVER_DYNAMIC_BATCH, warmup-sizes=${SERVER_WARMUP_SIZES:-none}"
echo "[nsys_profile] server compile-mode=$SERVER_COMPILE_MODE, dynamic-batch=$SERVER_DYNAMIC_BATCH, warmup-sizes=${SERVER_WARMUP_SIZES:-none}"

SRV_ADDR_FLAG=()
if [[ -n "$UDS" ]]; then
  # nsys/python expect scheme prefix for UDS
  if [[ "$UDS" != unix:* ]]; then
    SRV_ADDR_FLAG=(--uds "unix:$UDS")
  else
    SRV_ADDR_FLAG=(--uds "$UDS")
  fi
else
  SRV_ADDR_FLAG=(--tcp "$TCP")
fi

# If using UDS, remove any stale socket file before starting the server
if [[ -n "$UDS" ]]; then
  SOCK_PATH=${UDS#unix:}
  if [[ -S "$SOCK_PATH" ]]; then
    echo "[nsys_profile] removing stale UDS socket: $SOCK_PATH"
    rm -f -- "$SOCK_PATH" || true
  fi
fi

# No automatic multi-size warmup. We do at most a single warmup to mark batch dynamic.

# Start server under nsys in background
echo "[nsys_profile] starting server under Nsight Systems... (log: $SRV_LOG)"
set -x
NSYS_ENV_VARS="PYTHONUNBUFFERED=1,INFER_2048_LOG=0,INFER_2048_NO_CUDAGRAPHS=1"
nsys profile \
  --force-overwrite=true \
  --sample=cpu \
  --trace=cuda,osrt,nvtx \
  --cuda-event-trace=false \
  --env-var="$NSYS_ENV_VARS" \
  -o "$SRV_OUT" \
  uv run infer-2048 --init "$INIT_DIR" "${SRV_ADDR_FLAG[@]}" --device "$DEVICE" \
    --compile-mode "$SERVER_COMPILE_MODE" \
    ${SERVER_WARMUP_SIZES:+--warmup-sizes "$SERVER_WARMUP_SIZES"} \
    $( [[ $SERVER_DYNAMIC_BATCH -eq 1 ]] && echo "--dynamic-batch" ) \
    >"$SRV_LOG" 2>&1 &
SRV_NSYS_PID=$!
set +x

# Stream server logs to console while it starts
tail -n +1 -F "$SRV_LOG" &
TAIL_PID=$!

# Give server time to bind; for UDS, wait until socket exists
if [[ -n "$UDS" ]]; then
  SOCK_PATH=${UDS#unix:}
  echo "[nsys_profile] waiting for UDS socket: $SOCK_PATH (timeout=${SERVER_READY_TIMEOUT}s)"
  READY=0
  T0=$(date +%s)
  while true; do
    if [[ -S "$SOCK_PATH" ]]; then
      READY=1
      break
    fi
    if ! kill -0 "$SRV_NSYS_PID" 2>/dev/null; then
      die "Server profiler process exited early before creating socket; check Nsight output above."
    fi
    NOW=$(date +%s)
    if (( NOW - T0 >= SERVER_READY_TIMEOUT )); then
      break
    fi
    sleep 0.1
  done
  if [[ $READY -ne 1 ]]; then
    echo "[nsys_profile] Timed out waiting for UDS socket. Recent server log:" >&2
    tail -n 200 "$SRV_LOG" >&2 || true
    die "Timed out waiting for UDS socket: $SOCK_PATH"
  fi
else
  sleep "$SLEEP_SECS"
fi

# Actively wait for gRPC channel readiness (covers case where socket exists but server not ready)
echo "[nsys_profile] probing server readiness..."
if [[ -n "$UDS" ]]; then
  if [[ "$UDS" != unix:* ]]; then
    TARGET="unix:${UDS}"
  else
    TARGET="${UDS}"
  fi
else
  TARGET="${TCP}"
fi

set -x
uv run python - "$TARGET" "$SERVER_READY_TIMEOUT" << 'PY'
import sys, time, grpc

target = sys.argv[1]
timeout_s = float(sys.argv[2])

if not target:
    print("no target provided", file=sys.stderr)
    sys.exit(2)

channel = grpc.insecure_channel(target)
fut = grpc.channel_ready_future(channel)
try:
    fut.result(timeout=timeout_s)
    print("READY")
    sys.exit(0)
except Exception as e:
    print(f"NOT_READY: {e}", file=sys.stderr)
    sys.exit(1)
PY
RC=$?
set +x
if [[ $RC -ne 0 ]]; then
  die "Server not ready within ${SERVER_READY_TIMEOUT}s at ${TARGET}"
fi

# Run client without Nsight (server-only profiling)
echo "[nsys_profile] starting client (no Nsight)..."
set -x
if [[ $RELEASE -eq 1 ]]; then
  cargo run -p game-engine --release -- --config "$CLIENT_CFG"
else
  cargo run -p game-engine -- --config "$CLIENT_CFG"
fi
set +x

# Stop server (and thus stop server-side nsys)
echo "[nsys_profile] stopping server..."
kill $SRV_NSYS_PID || true
wait $SRV_NSYS_PID || true

# Stop log tailer
kill $TAIL_PID 2>/dev/null || true

echo "[nsys_profile] done. Open server trace in Nsight Systems (.nsys-rep):"
echo "  $SRV_OUT.nsys-rep"
