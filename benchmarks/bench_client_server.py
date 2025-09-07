from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import json
import grpc  # type: ignore


@dataclass
class RunPaths:
    outdir: Path
    client_cfg: Path
    results_file: Path
    metrics_file: Path
    meta_file: Path


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_outdir(outdir: Optional[str]) -> RunPaths:
    base = Path(outdir or f"bench_runs/{now_ts()}")
    base.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        outdir=base,
        client_cfg=base / "client-config.toml",
        results_file=base / "results.jsonl",
        metrics_file=base / "client-metrics.jsonl",
        meta_file=base / "run_meta.json",
    )


def read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_toml(d: dict, path: Path) -> None:
    # Minimal TOML writer sufficient for our flat structure
    def emit_table(fp, key, table):
        fp.write(f"[{key}]\n")
        for k, v in table.items():
            if isinstance(v, (int, float)):
                fp.write(f"{k} = {v}\n")
            elif isinstance(v, str):
                fp.write(f"{k} = \"{v}\"\n")
            elif isinstance(v, bool):
                fp.write(f"{k} = {'true' if v else 'false'}\n")
        fp.write("\n")

    with path.open("w", encoding="utf-8") as fp:
        # Top-level simple keys
        for k in ("num_seeds", "max_retries", "max_concurrent_games"):
            if k in d:
                fp.write(f"{k} = {d[k]}\n")

        if "sampling" in d:
            emit_table(fp, "sampling", d["sampling"])  # expects {strategy: "Argmax"}
        if "orchestrator" in d:
            orch = d["orchestrator"].copy()
            if "connection" in orch:
                emit_table(fp, "orchestrator.connection", orch["connection"])  # uds_path/tcp_addr
            if "batch" in orch:
                emit_table(fp, "orchestrator.batch", orch["batch"])  # flush_us, target_batch, ...
            if "report" in orch:
                emit_table(fp, "orchestrator.report", orch["report"])  # results_file


def make_client_cfg(base_cfg_path: Path, rp: RunPaths, uds: Optional[str], tcp: Optional[str]) -> Path:
    cfg = read_toml(base_cfg_path)
    # Ensure nested tables exist
    orch = cfg.setdefault("orchestrator", {})
    conn = orch.setdefault("connection", {})
    batch = orch.setdefault("batch", {})
    report = orch.setdefault("report", {})
    # Connection override from CLI bind
    if uds:
        conn["uds_path"] = uds
        conn.pop("tcp_addr", None)
    elif tcp:
        conn["tcp_addr"] = tcp
        conn.pop("uds_path", None)
    # Ensure metrics + results paths are present (if not set in base)
    batch.setdefault("metrics_file", str(rp.metrics_file))
    batch.setdefault("metrics_interval_s", 5.0)
    report.setdefault("results_file", str(rp.results_file))
    write_toml(cfg, rp.client_cfg)
    return rp.client_cfg


def wait_for_socket(uds: Optional[str], pid: int, timeout_s: float) -> None:
    if not uds:
        time.sleep(1.0)
        return
    sock_path = uds[5:] if uds.startswith("unix:") else uds
    t0 = time.time()
    while True:
        if Path(sock_path).exists():
            return
        if time.time() - t0 > timeout_s:
            print("[bench] timeout waiting for socket", file=sys.stderr)
            raise SystemExit(2)
        # If server died, abort early
        try:
            os.kill(pid, 0)
        except OSError:
            print("[bench] server exited during startup", file=sys.stderr)
            raise SystemExit(2)
        time.sleep(0.1)


def probe_grpc_ready(target: str, timeout_s: float) -> None:
    if target and not target.startswith("unix:") and target.startswith("/"):
        target = "unix:" + target
    ch = grpc.insecure_channel(target)
    fut = grpc.channel_ready_future(ch)
    fut.result(timeout=timeout_s)


def start_server(init_dir: str, bind: str, device: str, compile_mode: str, no_cudagraphs: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("INFER_2048_LOG", "0")
    if no_cudagraphs:
        env["INFER_2048_NO_CUDAGRAPHS"] = "1"
    cmd = [
        "uv",
        "run",
        "infer-2048",
        "--init",
        init_dir,
        "--device",
        device,
        "--compile-mode",
        compile_mode,
    ]
    if bind.startswith("unix:"):
        cmd += ["--uds", bind]
    elif ":" in bind or bind.startswith("http"):
        cmd += ["--tcp", bind]
    else:
        cmd += ["--uds", "unix:" + bind]
    # Only surface stderr to the console; discard stdout to avoid clutter.
    devnull = subprocess.DEVNULL
    popen_kwargs: dict = {"stdout": devnull, "stderr": None, "env": env}
    # Start server in its own process group so we can reliably terminate the whole tree.
    if os.name == "posix":  # Linux/macOS
        popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]
    elif os.name == "nt":  # Windows
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    return subprocess.Popen(cmd, **popen_kwargs)


def terminate_server(proc: subprocess.Popen, uds_norm: Optional[str], timeout: float = 5.0) -> None:
    try:
        if proc.poll() is not None:
            return
        if os.name == "posix":
            # Send SIGINT to the whole process group (uv + python child)
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except Exception:
                proc.send_signal(signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        proc.kill()
                else:
                    proc.kill()
                _ = proc.wait(timeout=timeout)
    finally:
        # Cleanup UDS path if provided
        if uds_norm:
            sock_path = uds_norm[5:] if uds_norm.startswith("unix:") else uds_norm
            try:
                if sock_path and Path(sock_path).exists():
                    Path(sock_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass


def start_client(client_cfg: Path, release: bool) -> int:
    cmd = [
        "cargo",
        "run",
        "-p",
        "game-engine",
    ]
    if release:
        cmd.append("--release")
    cmd += ["--", "--config", str(client_cfg)]
    # Only stderr to console, silence stdout
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None)
    return proc.wait()


def summarize_results(results_file: Path) -> None:
    # Simple JSONL reader
    scores, steps, tops = [], [], []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            scores.append(int(rec["score"]))
            steps.append(int(rec["steps"]))
            tops.append(int(rec["highest_tile"]))
    import statistics

    if not scores:
        print("No games found in results file.")
        return
    thresholds = [1024, 2048, 4096, 8192, 16384, 32768]
    reached_counts = {thr: sum(1 for t in tops if t >= thr) for thr in thresholds}
    print("\n=== Benchmark Summary (client/server) ===")
    print(f"games: {len(scores)}")
    print(f"scores    -> mean: {statistics.fmean(scores):.2f}  max: {max(scores)}  min: {min(scores)}")
    print(f"steps     -> mean: {statistics.fmean(steps):.2f}  max: {max(steps)}  min: {min(steps)}")
    print("reached   -> " + ", ".join(f"{thr}: {reached_counts[thr]}/{len(scores)}" for thr in thresholds))
    print(f"top tile  -> max: {max(tops)}  min: {min(tops)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run compiled server + Rust client and summarize results")
    ap.add_argument("--init", required=True, help="Path to init directory (config.json + model.safetensors)")
    # Bind is optional; if not provided, read it from the TOML under orchestrator.connection
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--uds", help="UDS path, e.g., /tmp/2048_infer.sock or unix:/tmp/…")
    g.add_argument("--tcp", help="TCP addr, e.g., 127.0.0.1:50051 or http://127.0.0.1:50051")
    ap.add_argument("--device", default="cuda", help="Server device: cuda|cpu|mps")
    ap.add_argument("--compile-mode", default="default", help="Server compile mode: default|max-autotune|reduce-overhead|none")
    ap.add_argument("--config", default="config/inference/top-score.toml", help="Client TOML config path")
    ap.add_argument("--outdir", default=None, help="Run output directory (default bench_runs/<ts>)")
    ap.add_argument("--no-cudagraphs", action="store_true", help="Disable Inductor CUDA graphs via env")
    ap.add_argument("--no-summary", action="store_true", help="Skip printing summary at the end")
    ap.add_argument("--debug", action="store_true", help="Run client in debug (non-release) mode")
    ap.add_argument("--timeout", type=float, default=120.0, help="Server readiness timeout seconds")

    args = ap.parse_args()
    rp = ensure_outdir(args.outdir)

    # Determine bind address from CLI or TOML config
    bind = args.uds or args.tcp
    uds_norm: Optional[str] = None
    tcp_norm: Optional[str] = None
    if not bind:
        base = read_toml(Path(args.config))
        conn = base.get("orchestrator", {}).get("connection", {})
        uds_from_cfg = conn.get("uds_path")
        tcp_from_cfg = conn.get("tcp_addr")
        if uds_from_cfg:
            bind = uds_from_cfg
        elif tcp_from_cfg:
            bind = tcp_from_cfg
        else:
            print("[bench] No --uds/--tcp provided and no bind found in TOML", file=sys.stderr)
            raise SystemExit(2)

    # Normalize bind for server/client
    if args.uds:
        uds_norm = args.uds if args.uds.startswith("unix:") else f"unix:{args.uds}"
    elif args.tcp:
        tcp_norm = args.tcp
    else:
        if str(bind).startswith("unix:") or str(bind).startswith("/"):
            uds_norm = bind if str(bind).startswith("unix:") else f"unix:{bind}"
        else:
            tcp_norm = str(bind)

    client_cfg = make_client_cfg(Path(args.config), rp, uds=uds_norm[5:] if uds_norm else None, tcp=tcp_norm)

    print(f"[bench] output dir: {rp.outdir}")
    print(f"[bench] server: device={args.device} compile={args.compile_mode} bind={bind}")
    print(f"[bench] client config: {client_cfg}")

    # Persist run metadata for provenance
    meta = {
        "ts": now_ts(),
        "init_dir": str(Path(args.init).resolve()),
        "device": args.device,
        "compile_mode": args.compile_mode,
        "bind": str(bind),
        "uds": uds_norm,
        "tcp": tcp_norm,
        "config": str(Path(args.config).resolve()),
        "client_release": not args.debug,
        "no_cudagraphs": bool(args.no_cudagraphs),
        "results_file": str(rp.results_file),
        "metrics_file": str(rp.metrics_file),
        "client_cfg": str(rp.client_cfg),
    }
    with rp.meta_file.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    # Start server
    srv = start_server(args.init, uds_norm or tcp_norm, args.device, args.compile_mode, args.no_cudagraphs)
    try:
        # Wait for bind and readiness
        wait_for_socket(uds_norm, srv.pid, args.timeout)
        probe_grpc_ready(uds_norm or tcp_norm, args.timeout)
        # Start client
        rc = start_client(client_cfg, release=(not args.debug))
        if rc != 0:
            print(f"[bench] client exited with code {rc}", file=sys.stderr)
    finally:
        terminate_server(srv, uds_norm, timeout=5.0)

    if not args.no_summary:
        summarize_results(rp.results_file)


if __name__ == "__main__":
    main()
