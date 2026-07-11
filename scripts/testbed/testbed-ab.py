#!/usr/bin/env python3
"""A/B benchmark runner for the fake-NUMA testbed (docs/numa-testbed.md).

Runs INSIDE the testbed container (see scripts/testbed/run-testbed.ps1); stdlib only.
Unlike scripts/numa-ab.py it has no systemd-run/perf/numastat dependencies — inside a
Docker Desktop container there is no systemd, no PMU, and only one real NUMA node.
Placement fidelity comes from the GGML_NUMA_STATS counters instead.

Subcommands:
  run --config cfg.json [--out DIR]   alternating A/B/A/B reps + one stats rep per variant
  report DIR [DIR...]                 mean +/- stdev, % delta vs first variant, counters
  smoke --model PATH [--bin PATH]     functional gate: inert / activation / greedy identity /
                                      counter sanity. Non-zero exit on any failure.

Config schema (JSON):
{
  "name": "baseline",
  "binary": "/src/build-testbed/bin/llama-cli",   // default for variants
  "model": "/models/foo.gguf",
  "prompt": "...", | "prompt_file": "path",
  "n_predict": 128, "threads": 8, "reps": 5, "warmup": 1,
  "common_args": ["-c", "4096"],
  "common_env": {"GGML_NUMA_FAKE": "2", "GGML_NUMA_XGMI_GBPS": "60"},
  "variants": [
    {"name": "A-mirror", "args": ["--numa", "mirror"], "env": {}},
    {"name": "B-candidate", "binary": "/src-b/build-testbed/bin/llama-cli",
     "args": ["--numa", "mirror"], "env": {"GGML_NUMA_PIN": "cpu"}}
  ]
}
"""
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys

DEFAULT_PROMPT = ("Write a detailed essay about the history of computing, starting with "
                  "mechanical calculators and ending with modern GPUs.")

RE_TIMING = re.compile(
    r"llama_print_timings:\s*(prompt eval|eval)\s+time\s*=.*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second")
RE_LOAD = re.compile(r"llama_print_timings:\s*load time\s*=\s*([\d.]+)\s*ms")
RE_STAT = re.compile(r"numa_stats:\s*(.*)$")


def parse_llama_output(text):
    metrics = {}
    for m in RE_TIMING.finditer(text):
        key = "pp_tps" if m.group(1) == "prompt eval" else "tg_tps"
        metrics.setdefault(key, []).append(float(m.group(2)))
    metrics = {k: (round(sum(v) / len(v), 2) if v else None) for k, v in metrics.items()}
    m = RE_LOAD.search(text)
    if m:
        metrics["load_ms"] = float(m.group(1))
    return metrics


def parse_numa_stats(text):
    """Collect key=value pairs from every 'numa_stats:' line (later windows override)."""
    stats = {}
    for m in RE_STAT.finditer(text):
        parts = m.group(1).split()
        prefix = ""
        for p in parts:
            if "=" not in p:
                prefix = p + "_"  # e.g. 'node0' prefixes that line's keys
                continue
            k, _, v = p.partition("=")
            try:
                stats[prefix + k] = float(v) if "." in v else int(v)
            except ValueError:
                stats[prefix + k] = v
    return stats


def build_cmd(cfg, variant, stats_rep):
    binary = variant.get("binary", cfg["binary"])
    cmd = [binary, "-m", cfg["model"], "-t", str(cfg.get("threads", 8)),
           "-n", str(cfg.get("n_predict", 128)), "--temp", "0", "--seed", "1"]
    if cfg.get("prompt_file"):
        cmd += ["-f", cfg["prompt_file"]]
    else:
        cmd += ["-p", cfg.get("prompt", DEFAULT_PROMPT)]
    cmd += cfg.get("common_args", [])
    cmd += variant.get("args", [])

    env = dict(os.environ)
    env.update(cfg.get("common_env", {}))
    env.update(variant.get("env", {}))
    # timing reps run clean; the dedicated stats rep turns the counters on
    env["GGML_NUMA_STATS"] = "1" if stats_rep else "0"
    return cmd, env


def run_once(cfg, variant, stats_rep, log_path):
    cmd, env = build_cmd(cfg, variant, stats_rep)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=cfg.get("timeout_s", 3600))
    text = r.stdout + "\n" + r.stderr
    with open(log_path, "w") as f:
        f.write("# cmd: %s\n# env overrides: %s\n\n%s" % (
            " ".join(cmd),
            {k: env[k] for k in list(cfg.get("common_env", {})) + list(variant.get("env", {})) + ["GGML_NUMA_STATS"] if k in env},
            text))
    if r.returncode != 0:
        raise RuntimeError("%s exited %d (log: %s)" % (variant["name"], r.returncode, log_path))
    rec = {"variant": variant["name"], "stats_rep": stats_rep,
           "metrics": parse_llama_output(text)}
    if stats_rep:
        rec["numa_stats"] = parse_numa_stats(text)
        rec["stdout"] = r.stdout  # kept so smoke/identity checks can reuse stats reps
    return rec


def cmd_run(args):
    with open(args.config) as f:
        cfg = json.load(f)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or os.path.join("testbed-results", "%s-%s" % (stamp, cfg.get("name", "run")))
    os.makedirs(out_dir, exist_ok=True)
    variants = cfg["variants"]
    reps = int(cfg.get("reps", 5))
    warmup = int(cfg.get("warmup", 1))
    records = []

    for v in variants:
        for w in range(warmup):
            print("[warmup %d/%d] %s" % (w + 1, warmup, v["name"]), flush=True)
            run_once(cfg, v, False, os.path.join(out_dir, "warmup-%s-%d.log" % (v["name"], w)))

    # alternating order so drift (thermals, background load) hits all variants equally
    for rep in range(reps):
        for v in variants:
            print("[rep %d/%d] %s" % (rep + 1, reps, v["name"]), flush=True)
            rec = run_once(cfg, v, False, os.path.join(out_dir, "rep%d-%s.log" % (rep, v["name"])))
            rec["rep"] = rep
            records.append(rec)

    for v in variants:
        print("[stats rep] %s" % v["name"], flush=True)
        rec = run_once(cfg, v, True, os.path.join(out_dir, "stats-%s.log" % v["name"]))
        rec.pop("stdout", None)
        records.append(rec)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": cfg, "records": records}, f, indent=2)
    print("\nresults written to %s\n" % out_dir)
    print(render_report(out_dir))
    return 0


def summarize(records, variants):
    rows = []
    base = None
    for name in variants:
        timing = [r["metrics"] for r in records if r["variant"] == name and not r["stats_rep"]]
        stats = next((r.get("numa_stats") for r in records if r["variant"] == name and r["stats_rep"]), None)
        row = {"name": name, "n": len(timing), "stats": stats or {}}
        for key in ("pp_tps", "tg_tps"):
            vals = [t[key] for t in timing if t.get(key) is not None]
            row[key] = statistics.mean(vals) if vals else None
            row[key + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if base is None:
            base = row
        for key in ("pp_tps", "tg_tps"):
            row[key + "_delta"] = (100.0 * (row[key] - base[key]) / base[key]
                                   if row[key] and base[key] else None)
        rows.append(row)
    return rows


def render_report(out_dir):
    with open(os.path.join(out_dir, "results.json")) as f:
        data = json.load(f)
    variants = [v["name"] for v in data["config"]["variants"]]
    rows = summarize(data["records"], variants)

    def fmt(v, pat="%.2f"):
        return pat % v if v is not None else "-"

    lines = ["## %s (%s)" % (data["config"].get("name", "run"), out_dir), "",
             "| variant | n | pp t/s | tg t/s | pp Δ% | tg Δ% |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %d | %s ± %s | %s ± %s | %s | %s |" % (
            r["name"], r["n"],
            fmt(r["pp_tps"]), fmt(r["pp_tps_sd"]),
            fmt(r["tg_tps"]), fmt(r["tg_tps_sd"]),
            fmt(r["pp_tps_delta"], "%+.2f"), fmt(r["tg_tps_delta"], "%+.2f")))
    lines.append("")
    counter_keys = ["kv_repl_bytes", "kv_repl_calls", "barriers_hier", "barriers_flat",
                    "throttle_sleep_us", "populate_bytes", "resync_bytes"]
    node_keys = sorted({k for r in rows for k in r["stats"] if k.startswith("node")})
    if any(r["stats"] for r in rows):
        lines += ["| variant | " + " | ".join(node_keys + counter_keys) + " |",
                  "|---" * (1 + len(node_keys) + len(counter_keys)) + "|"]
        for r in rows:
            cells = [str(r["stats"].get(k, "-")) for k in node_keys + counter_keys]
            lines.append("| %s | %s |" % (r["name"], " | ".join(cells)))
        lines.append("")
    return "\n".join(lines)


def cmd_report(args):
    for d in args.dirs:
        print(render_report(d))
    return 0


# ---------------------------------------------------------------- smoke

def smoke_run(binary, model, extra_args, env_overrides, n_predict=32):
    cmd = [binary, "-m", model, "-t", "8", "-n", str(n_predict),
           "--temp", "0", "--seed", "1", "-p", "The capital of France is"]
    cmd += extra_args
    env = dict(os.environ)
    env.pop("GGML_NUMA_FAKE", None)
    env.pop("GGML_NUMA_XGMI_GBPS", None)
    env.pop("GGML_NUMA_PIN", None)
    env["GGML_NUMA_STATS"] = "1"
    env.update(env_overrides)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("smoke run failed (%d): %s\n%s" % (r.returncode, " ".join(cmd), r.stderr[-2000:]))
    return r.stdout, r.stderr


def cmd_smoke(args):
    binary = args.bin
    model = args.model
    failures = []

    def check(name, ok, detail=""):
        print("[%s] %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
        if not ok:
            failures.append(name)

    # 1. inert: no fake env, no --numa -> shim must leave zero trace
    out0, err0 = smoke_run(binary, model, [], {"GGML_NUMA_STATS": "0"})
    check("inert (no FAKE banner without env)", "FAKE NUMA" not in out0 + err0)

    # 2. activation: fake 2-node mirror must engage the whole stack
    out1, err1 = smoke_run(binary, model, ["--numa", "mirror"], {"GGML_NUMA_FAKE": "2"})
    log1 = out1 + err1
    check("activation: FAKE banner", "FAKE NUMA topology" in log1)
    check("activation: weights mirrored", re.search(r"duplicat\w* \d+ weight tensors", log1) is not None,
          "(expect 'duplicated N weight tensors')")
    check("activation: kv mirrored", re.search(r"mirror\w* \d+ KV", log1, re.I) is not None,
          "(expect 'mirrored N KV tensors')")
    stats1 = parse_numa_stats(log1)
    check("stats: node0+node1 resolve hits", stats1.get("node0_resolve_hit", 0) > 0
          and stats1.get("node1_resolve_hit", 0) > 0, str({k: v for k, v in stats1.items() if "resolve" in k}))
    check("stats: zero fallbacks", stats1.get("node0_resolve_fallback", 0) == 0
          and stats1.get("node1_resolve_fallback", 0) == 0)
    check("stats: kv replication ran", stats1.get("kv_repl_bytes", 0) > 0)

    # 3. greedy identity across shim/throttle/pin variants
    out2, _ = smoke_run(binary, model, ["--numa", "mirror"],
                        {"GGML_NUMA_FAKE": "2", "GGML_NUMA_XGMI_GBPS": "8"})
    out3, _ = smoke_run(binary, model, ["--numa", "mirror"],
                        {"GGML_NUMA_FAKE": "2", "GGML_NUMA_PIN": "cpu"})
    check("identity: baseline == mirror", out0 == out1)
    check("identity: mirror == mirror+throttle", out1 == out2)
    check("identity: mirror == mirror+pin", out1 == out3)

    print("\nsmoke: %s" % ("OK" if not failures else "FAILED: " + ", ".join(failures)))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run an A/B config")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)

    p = sub.add_parser("report", help="re-render reports for result dirs")
    p.add_argument("dirs", nargs="+")

    p = sub.add_parser("smoke", help="functional gate on a small model")
    p.add_argument("--model", required=True)
    p.add_argument("--bin", default="/src/build-testbed/bin/llama-cli")

    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "report":
        return cmd_report(args)
    return cmd_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
