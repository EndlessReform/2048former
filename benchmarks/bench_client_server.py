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
from threading import Thread
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
    server_stderr: Path


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    stderr_log: Optional[Path] = None
    stderr_thread: Optional[Thread] = None


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
        server_stderr=base / "server-stderr.log",
    )


def read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_toml(d: dict, path: Path) -> None:
    # Minimal TOML writer sufficient for our flat structure
    def emit_table(fp, key, table):
        fp.write(f"[{key}]\n")
        for k, v in table.items():
            # IMPORTANT: check bool before int — bool is a subclass of int in Python
            if isinstance(v, bool):
                fp.write(f"{k} = {'true' if v else 'false'}\n")
            elif isinstance(v, (int, float)):
                fp.write(f"{k} = {v}\n")
            elif isinstance(v, str):
                fp.write(f"{k} = \"{v}\"\n")
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
            # Emit simple orchestrator keys (e.g., head_order, argmax_only, inline_embeddings, fixed_seed, random_seeds)
            simple = {
                k: v
                for k, v in orch.items()
                if k not in {"connection", "batch", "report"}
                and isinstance(v, (int, float, str, bool))
            }
            if simple:
                emit_table(fp, "orchestrator", simple)
            if "connection" in orch:
                emit_table(fp, "orchestrator.connection", orch["connection"])  # uds_path/tcp_addr
            if "batch" in orch:
                emit_table(fp, "orchestrator.batch", orch["batch"])  # flush_us, target_batch, ...
            if "report" in orch:
                emit_table(fp, "orchestrator.report", orch["report"])  # results_file


def _sampling_grid_variants(base_cfg: dict) -> list[tuple[str, dict]]:
    """Detect arrays under [sampling] and return (label, cfg_variant) for each Cartesian combo.

    Supported array-eligible keys: any scalar field under [sampling], including strategy.
    If no arrays are present, returns a single ("base", base_cfg) entry.
    """
    import itertools

    def is_array(x) -> bool:
        return isinstance(x, list) and len(x) > 0

    cfg = json.loads(json.dumps(base_cfg))  # deep copy via JSON
    s = cfg.get("sampling", {})
    # Collect keys with array values (sorted for stable labeling)
    keys: list[str] = []
    values: list[list[object]] = []
    for k in sorted(s.keys()):
        v = s[k]
        if is_array(v):
            keys.append(k)
            values.append(list(v))

    if not keys:
        return [("base", cfg)]

    variants: list[tuple[str, dict]] = []
    for combo in itertools.product(*values):
        var = json.loads(json.dumps(cfg))
        # Apply scalar choices
        for k, val in zip(keys, combo):
            var.setdefault("sampling", {})[k] = val
        # Build stable, short-ish label
        parts: list[str] = []
        for k, val in zip(keys, combo):
            sval = str(val)
            sval = sval.replace("/", "_").replace(":", "_").replace(" ", "").replace(",", "-")
            sval = sval.replace(".", "p")  # make FS friendly
            parts.append(f"{k}-{sval}")
        label = "_".join(parts)
        variants.append((label, var))
    return variants


def _collect_run_metrics(results_file: Path) -> dict:
    scores, steps, tops = [], [], []
    if not results_file.exists():
        return {"games": 0}
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scores.append(int(rec["score"]))
            steps.append(int(rec["steps"]))
            tops.append(int(rec["highest_tile"]))
    import statistics
    if not scores:
        return {"games": 0}
    thresholds = [1024, 2048, 4096, 8192, 16384, 32768]
    reached_counts = {thr: sum(1 for t in tops if t >= thr) for thr in thresholds}
    return {
        "games": len(scores),
        "score_mean": float(statistics.fmean(scores)),
        "score_max": int(max(scores)),
        "score_min": int(min(scores)),
        "steps_mean": float(statistics.fmean(steps)),
        "steps_max": int(max(steps)),
        "steps_min": int(min(steps)),
        "top_max": int(max(tops)),
        "top_min": int(min(tops)),
        **{f"reach_{thr}": reached_counts[thr] for thr in thresholds},
    }


def _format_summary_text(results_file: Path, title: str = "Benchmark Summary (client/server)") -> str:
    m = _collect_run_metrics(results_file)
    lines: list[str] = []
    lines.append("\n=== " + title + " ===")
    games = int(m.get("games", 0) or 0)
    if games == 0:
        lines.append("games: 0")
        return "\n".join(lines)
    lines.append(f"games: {games}")
    lines.append(
        f"scores    -> mean: {m['score_mean']:.2f}  max: {m['score_max']}  min: {m['score_min']}"
    )
    lines.append(
        f"steps     -> mean: {m['steps_mean']:.2f}  max: {m['steps_max']}  min: {m['steps_min']}"
    )
    thresholds = [1024, 2048, 4096, 8192, 16384, 32768]
    lines.append(
        "reached   -> "
        + ", ".join(f"{thr}: {int(m.get(f'reach_{thr}', 0))}/{games}" for thr in thresholds)
    )
    lines.append(f"top tile  -> max: {m['top_max']}  min: {m['top_min']}")
    return "\n".join(lines)


def _write_grid_csv_summary(outdir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Determine columns
    cols = [
        "label",
        "games",
        "score_mean",
        "score_max",
        "score_min",
        "steps_mean",
        "steps_max",
        "steps_min",
        "top_max",
        "reach_1024",
        "reach_2048",
        "reach_4096",
        "reach_8192",
        "reach_16384",
        "reach_32768",
    ]
    path = outdir / "grid_summary.csv"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(",".join(cols) + "\n")
        for r in rows:
            fp.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


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


def make_client_cfg_for_variant(
    base_cfg: dict,
    base_outdir: Path,
    label: str,
    uds: Optional[str],
    tcp: Optional[str],
) -> Path:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    orch = cfg.setdefault("orchestrator", {})
    conn = orch.setdefault("connection", {})
    batch = orch.setdefault("batch", {})
    report = orch.setdefault("report", {})

    # Connection override
    if uds:
        conn["uds_path"] = uds
        conn.pop("tcp_addr", None)
    elif tcp:
        conn["tcp_addr"] = tcp
        conn.pop("uds_path", None)

    subdir = base_outdir / label
    subdir.mkdir(parents=True, exist_ok=True)
    batch.setdefault("metrics_file", str(subdir / "client-metrics.jsonl"))
    batch.setdefault("metrics_interval_s", 5.0)
    report.setdefault("results_file", str(subdir / "results.jsonl"))

    path = subdir / "client-config.toml"
    write_toml(cfg, path)
    return path


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


def _start_stderr_tee(proc: subprocess.Popen, log_path: Path) -> Thread:
    def _pump() -> None:
        assert proc.stderr is not None
        with log_path.open("w", encoding="utf-8") as fp:
            for chunk in proc.stderr:
                fp.write(chunk)
                fp.flush()
                sys.stderr.write(chunk)
                sys.stderr.flush()

    thread = Thread(target=_pump, daemon=True)
    thread.start()
    return thread


def start_server(
    init_dir: str,
    bind: str,
    device: str,
    compile_mode: str,
    no_cudagraphs: bool,
    *,
    stderr_log: Optional[Path] = None,
) -> ServerHandle:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("INFER_2048_LOG", "0")
    if no_cudagraphs:
        env["INFER_2048_NO_CUDAGRAPHS"] = "1"
    # UDLR canonical; no head-order env needed
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
    popen_kwargs: dict = {"stdout": devnull, "env": env}
    if stderr_log:
        popen_kwargs.update({"stderr": subprocess.PIPE, "text": True, "bufsize": 1})
    else:
        popen_kwargs.update({"stderr": None})
    # Start server in its own process group so we can reliably terminate the whole tree.
    if os.name == "posix":  # Linux/macOS
        popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]
    elif os.name == "nt":  # Windows
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(cmd, **popen_kwargs)
    thread: Optional[Thread] = None
    if stderr_log:
        thread = _start_stderr_tee(proc, stderr_log)
    return ServerHandle(proc=proc, stderr_log=stderr_log, stderr_thread=thread)


def terminate_server(handle: ServerHandle, uds_norm: Optional[str], timeout: float = 5.0) -> None:
    proc = handle.proc
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
        if handle.stderr_thread is not None:
            handle.stderr_thread.join(timeout=1.0)
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
        "--bin",
        "game-engine",
    ]
    if release:
        cmd.append("--release")
    cmd += ["--", "--config", str(client_cfg)]
    # Only stderr to console, silence stdout
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None)
    return proc.wait()


def summarize_results(results_file: Path) -> str:
    text = _format_summary_text(results_file)
    print(text)
    return text


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
    ap.add_argument("--grid-resume", action="store_true", help="Skip variants with an existing results.jsonl")

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

    base_cfg = read_toml(Path(args.config))
    variants = _sampling_grid_variants(base_cfg)

    server_device = args.device.strip()

    # If no grid present, behave like before with a single config.
    single_cfg = None
    if len(variants) == 1 and variants[0][0] == "base":
        client_cfg = make_client_cfg(
            Path(args.config),
            rp,
            uds=uds_norm[5:] if uds_norm else None,
            tcp=tcp_norm,
        )
        single_cfg = client_cfg
        print(f"[bench] output dir: {rp.outdir}")
        print(f"[bench] server: device={server_device} compile={args.compile_mode} bind={bind}")
        print(f"[bench] client config: {client_cfg}")
    else:
        print(f"[bench] output dir: {rp.outdir}  (grid {len(variants)} variants)")
        print(f"[bench] server: device={server_device} compile={args.compile_mode} bind={bind}")

    # Persist run metadata for provenance
    meta = {
        "ts": now_ts(),
        "init_dir": str(Path(args.init).resolve()),
        "device": server_device,
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
        "server_stderr": str(rp.server_stderr),
    }
    with rp.meta_file.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    # Start server (once for all variants)
    # Propagate client's intended head order to server so outputs are normalized consistently.
    srv = start_server(
        args.init,
        uds_norm or tcp_norm,
        server_device,
        args.compile_mode,
        args.no_cudagraphs,
        stderr_log=rp.server_stderr,
    )
    try:
        # Wait for bind and readiness
        wait_for_socket(uds_norm, srv.proc.pid, args.timeout)
        probe_grpc_ready(uds_norm or tcp_norm, args.timeout)
        if single_cfg is not None:
            # Single run
            rc = start_client(single_cfg, release=(not args.debug))
            if rc != 0:
                print(f"[bench] client exited with code {rc}", file=sys.stderr)
        else:
            # Grid search: run each variant sequentially
            grid_rows: list[dict] = []
            for idx, (label, cfg_variant) in enumerate(variants):
                cfg_path = make_client_cfg_for_variant(
                    cfg_variant,
                    rp.outdir,
                    f"grid_{idx:03d}_" + label,
                    uds=uds_norm[5:] if uds_norm else None,
                    tcp=tcp_norm,
                )
                print(f"[bench] grid[{idx+1}/{len(variants)}] -> {cfg_path}")
                res_file = cfg_path.parent / "results.jsonl"
                if args.grid_resume and res_file.exists() and res_file.stat().st_size > 0:
                    print("[bench] resume: results exist, skipping")
                else:
                    rc = start_client(cfg_path, release=(not args.debug))
                    if rc != 0:
                        print(f"[bench] client exited with code {rc}", file=sys.stderr)
                # accumulate metrics row
                row = {"label": label}
                row.update(_collect_run_metrics(res_file))
                grid_rows.append(row)
            _write_grid_csv_summary(rp.outdir, grid_rows)
    finally:
        terminate_server(srv, uds_norm, timeout=5.0)

    server_rc = srv.proc.returncode
    if server_rc not in (0, None):
        print(
            f"[bench] server exited with code {server_rc}; see {rp.server_stderr} for details",
            file=sys.stderr,
        )
    elif rp.server_stderr.exists() and rp.server_stderr.stat().st_size > 0:
        print(f"[bench] server stderr captured at {rp.server_stderr}")

    if not args.no_summary:
        if single_cfg is not None:
            text = summarize_results(rp.results_file)
            # Persist the exact summary to a text file in the run directory
            with (rp.outdir / "summary.txt").open("w", encoding="utf-8") as fp:
                fp.write(text + "\n")
        else:
            # Summarize each variant's results and write a combined summary file
            out_txt = rp.outdir / "summary.txt"
            with out_txt.open("w", encoding="utf-8") as fp:
                for idx, (label, _cfg_variant) in enumerate(variants):
                    res_file = rp.outdir / (f"grid_{idx:03d}_" + label) / "results.jsonl"
                    if res_file.exists():
                        header = f"[bench] Summary for {label}"
                        print("\n" + header)
                        text = _format_summary_text(res_file, title=f"Benchmark Summary (client/server) — {label}")
                        print(text)
                        fp.write(header + "\n")
                        fp.write(text + "\n")


if __name__ == "__main__":
    main()
