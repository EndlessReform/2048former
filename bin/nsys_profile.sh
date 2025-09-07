#!/usr/bin/env bash
set -euo pipefail

# Simple orchestrator to record Nsight Systems traces for both the Python
# inference server and the Rust client (game-engine) in separate files.
#
# Outputs two .qdrep files under an output directory (default: profiles/<ts>):
#   - server.qdrep: CUDA + CPU + Python trace for the server process
#   - client.qdrep: CPU/OS trace for the Rust orchestrator
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

echo "[nsys_profile] writing traces under: $OUTDIR"

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

# Start server under nsys in background
echo "[nsys_profile] starting server under Nsight Systems..."
set -x
nsys profile \
  --force-overwrite=true \
  --sample=cpu \
  --trace=cuda,osrt,nvtx,python \
  -o "$SRV_OUT" \
  uv run infer-2048 --init "$INIT_DIR" "${SRV_ADDR_FLAG[@]}" --device "$DEVICE" &
SRV_NSYS_PID=$!
set +x

# Give server time to bind; for UDS, wait until socket exists
if [[ -n "$UDS" ]]; then
  SOCK_PATH=${UDS#unix:}
  echo "[nsys_profile] waiting for UDS socket: $SOCK_PATH"
  for i in {1..50}; do
    [[ -S "$SOCK_PATH" ]] && break
    sleep 0.1
  done
else
  sleep "$SLEEP_SECS"
fi

# Run client under nsys (CPU/OS trace)
echo "[nsys_profile] starting client under Nsight Systems..."
set -x
if [[ $RELEASE -eq 1 ]]; then
  nsys profile --force-overwrite=true --sample=cpu --trace=osrt,nvtx -o "$CLI_OUT" \
    cargo run -p game-engine --release -- --config "$CLIENT_CFG"
else
  nsys profile --force-overwrite=true --sample=cpu --trace=osrt,nvtx -o "$CLI_OUT" \
    cargo run -p game-engine -- --config "$CLIENT_CFG"
fi
set +x

# Stop server (and thus stop server-side nsys)
echo "[nsys_profile] stopping server..."
kill $SRV_NSYS_PID || true
wait $SRV_NSYS_PID || true

echo "[nsys_profile] done. Open traces in Nsight Systems (.qdrep):"
echo "  $SRV_OUT.qdrep"
echo "  $CLI_OUT.qdrep"

