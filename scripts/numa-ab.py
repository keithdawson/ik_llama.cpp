#!/usr/bin/env python3
"""A/B harness for NUMA placement experiments with ik_llama.cpp.

Runs a llama.cpp workload inside a cgroup with a chosen core split, samples
memory placement (numa_maps / numastat) and cross-socket traffic (perf or
AMD uProf) while it runs, parses the llama timings from its output, and
saves everything as JSON so configurations can be compared side by side.

Requires only the Python 3 standard library. Linux only (except `report`).

Examples
--------
# A: hybrid flags off (plain mirror), leaving cores 0-15 + SMT siblings for vLLM
sudo ./scripts/numa-ab.py run --name mirror-plain --reserve-node0 16 -- \
    ./build/bin/llama-cli -m model.gguf --numa mirror -t 160 -ngl 999 --cpu-moe \
    -p "<long prompt here>" -n 128 --ignore-eos

# B: gpu-node binding on
sudo ./scripts/numa-ab.py run --name gpu-node1 --reserve-node0 16 -- \
    ./build/bin/llama-cli -m model.gguf --numa mirror --numa-gpu-node 1 \
    -t 160 -ngl 999 --cpu-moe -p "<long prompt here>" -n 128 --ignore-eos

# C: compute-buffer binding on top
sudo ./scripts/numa-ab.py run --name bind-compute --reserve-node0 16 -- \
    ./build/bin/llama-cli -m model.gguf --numa mirror --numa-gpu-node 1 \
    --numa-bind-compute -t 160 -ngl 999 --cpu-moe -p "<long prompt>" -n 128 --ignore-eos

# live placement view of an already-running server
./scripts/numa-ab.py monitor --pid $(pidof llama-server)

# compare all recorded runs
./scripts/numa-ab.py report
"""

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "numa-ab-results")

# candidate perf event pairs for (local DRAM fills, remote/far DRAM fills), newest first
PERF_EVENT_PAIRS = [
    ("ls_dmnd_fills_from_sys.dram_io_near", "ls_dmnd_fills_from_sys.dram_io_far"),
    ("ls_dmnd_fills_from_sys.mem_io_local", "ls_dmnd_fills_from_sys.mem_io_remote"),
]


# ---------------------------------------------------------------- cpu topology

def parse_cpulist(s):
    cpus = set()
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def format_cpulist(cpus):
    cpus = sorted(cpus)
    ranges = []
    start = prev = cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append((start, prev))
        start = prev = c
    ranges.append((start, prev))
    return ",".join("%d" % a if a == b else "%d-%d" % (a, b) for a, b in ranges)


def node_cpus():
    """{node_id: set(cpu ids)} from sysfs."""
    nodes = {}
    base = "/sys/devices/system/node"
    for entry in sorted(os.listdir(base)):
        m = re.fullmatch(r"node(\d+)", entry)
        if m:
            with open(os.path.join(base, entry, "cpulist")) as f:
                nodes[int(m.group(1))] = parse_cpulist(f.read())
    return nodes


def smt_siblings(cpu):
    try:
        with open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu) as f:
            return parse_cpulist(f.read())
    except OSError:
        return {cpu}


def compute_allowed_cpus(reserve_node0):
    """All CPUs minus the first `reserve_node0` physical cores of node 0 (with SMT siblings)."""
    nodes = node_cpus()
    allowed = set()
    for cpus in nodes.values():
        allowed |= cpus
    if reserve_node0 > 0:
        seen = set()
        reserved = set()
        n_cores = 0
        for cpu in sorted(nodes.get(0, set())):
            if cpu in seen:
                continue
            sibs = smt_siblings(cpu)
            seen |= sibs
            reserved |= sibs
            n_cores += 1
            if n_cores >= reserve_node0:
                break
        allowed -= reserved
        print("reserving node-0 cores for other workloads: %s" % format_cpulist(reserved))
    return allowed


# ---------------------------------------------------------------- telemetry

def read_numa_maps(pid):
    """Summarize /proc/<pid>/numa_maps: per-node MiB, plus the largest mappings."""
    per_node = {}
    mappings = []
    try:
        with open("/proc/%d/numa_maps" % pid) as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in lines:
        toks = line.split()
        if not toks:
            continue
        page_kb = 4
        for t in toks:
            if t.startswith("kernelpagesize_kB="):
                page_kb = int(t.split("=")[1])
        entry = {"addr": toks[0], "nodes": {}, "mib": 0.0}
        label = "anon"
        for t in toks[1:]:
            if t.startswith("file="):
                label = os.path.basename(t[5:])
            m = re.fullmatch(r"N(\d+)=(\d+)", t)
            if m:
                mib = int(m.group(2)) * page_kb / 1024.0
                node = int(m.group(1))
                entry["nodes"][node] = entry["nodes"].get(node, 0.0) + mib
                entry["mib"] += mib
                per_node[node] = per_node.get(node, 0.0) + mib
        if entry["mib"] > 0:
            entry["label"] = label
            mappings.append(entry)
    mappings.sort(key=lambda e: -e["mib"])
    return {
        "per_node_mib": {str(k): round(v, 1) for k, v in sorted(per_node.items())},
        "top_mappings": [
            {
                "label": e["label"],
                "mib": round(e["mib"], 1),
                "per_node_mib": {str(k): round(v, 1) for k, v in sorted(e["nodes"].items())},
            }
            for e in mappings[:10]
        ],
    }


def read_sys_numastat():
    """{node: {counter: value}} from /sys/devices/system/node/node*/numastat."""
    out = {}
    base = "/sys/devices/system/node"
    for entry in os.listdir(base):
        m = re.fullmatch(r"node(\d+)", entry)
        if not m:
            continue
        counters = {}
        try:
            with open(os.path.join(base, entry, "numastat")) as f:
                for line in f:
                    k, v = line.split()
                    counters[k] = int(v)
        except OSError:
            continue
        out[int(m.group(1))] = counters
    return out


def read_node_free_mib():
    out = {}
    base = "/sys/devices/system/node"
    for entry in os.listdir(base):
        m = re.fullmatch(r"node(\d+)", entry)
        if not m:
            continue
        try:
            with open(os.path.join(base, entry, "meminfo")) as f:
                for line in f:
                    if "MemFree" in line:
                        out[int(m.group(1))] = round(int(line.split()[-2]) / 1024.0, 1)
        except OSError:
            pass
    return out


def probe_perf_events():
    if not shutil.which("perf"):
        return None
    for near, far in PERF_EVENT_PAIRS:
        r = subprocess.run(
            ["perf", "stat", "-a", "-x", ",", "-e", "%s,%s" % (near, far), "--", "sleep", "0.1"],
            capture_output=True, text=True)
        if r.returncode == 0 and "<not supported>" not in r.stderr and "<not counted>" not in r.stderr:
            return (near, far)
    return None


def perf_sample(events, seconds):
    """Returns dict {event: count} or None."""
    near, far = events
    r = subprocess.run(
        ["perf", "stat", "-a", "-x", ",", "-e", "%s,%s" % (near, far), "--", "sleep", str(seconds)],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    out = {}
    for line in r.stderr.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].replace(" ", "").isdigit():
            out[parts[2]] = int(parts[0])
    if len(out) < 2:
        return None
    total = sum(out.values())
    out["remote_ratio_pct"] = round(100.0 * out.get(far, 0) / total, 2) if total else 0.0
    return out


# ---------------------------------------------------------------- output parsing

RE_TIMING = re.compile(
    r"llama_print_timings:\s*(prompt eval|eval)\s+time\s*=.*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second")
RE_BENCH = re.compile(r"\|\s*(pp\d+|tg\d+|pp\d+\+tg\d+)\s*\|\s*([\d.]+)\s*(?:\xb1|\+/-)?\s*[\d.]*\s*\|\s*$")


def parse_llama_output(text):
    metrics = {}
    for m in RE_TIMING.finditer(text):
        key = "pp_tps" if m.group(1) == "prompt eval" else "tg_tps"
        metrics.setdefault(key, []).append(float(m.group(2)))
    for line in text.splitlines():
        m = RE_BENCH.search(line)
        if m:
            key = "pp_tps" if m.group(1).startswith("pp") else "tg_tps"
            metrics.setdefault(key, []).append(float(m.group(2)))
    return {k: (round(sum(v) / len(v), 2) if v else None) for k, v in metrics.items()}


# ---------------------------------------------------------------- run

def find_workload_pid(launcher_pid, cmd0):
    """With systemd-run --scope the workload is a child of the launcher."""
    try:
        kids = subprocess.run(["pgrep", "-P", str(launcher_pid)],
                              capture_output=True, text=True).stdout.split()
        if kids:
            return int(kids[0])
    except (ValueError, OSError):
        pass
    return launcher_pid


def cmd_run(args, workload):
    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "%s-%s" % (args.name, stamp)
    log_path = os.path.join(args.results_dir, tag + ".log")

    if args.allowed_cpus:
        allowed = parse_cpulist(args.allowed_cpus)
    elif args.reserve_node0:
        allowed = compute_allowed_cpus(args.reserve_node0)
    else:
        allowed = None

    launch = []
    if allowed is not None:
        if shutil.which("systemd-run") and os.geteuid() == 0:
            launch = ["systemd-run", "--scope", "--collect", "-q",
                      "-p", "AllowedCPUs=%s" % format_cpulist(allowed), "--"]
        else:
            print("WARNING: need root + systemd-run for an enforced core split "
                  "(numactl/taskset would be overridden by ggml's own pinning). "
                  "Running without confinement.", file=sys.stderr)

    events = None
    if args.bw_tool in ("auto", "perf"):
        events = probe_perf_events()
        if events:
            print("perf events: %s / %s" % events)
        elif args.bw_tool == "perf":
            print("WARNING: no usable perf DRAM-fill events found", file=sys.stderr)

    print("launching: %s" % " ".join(launch + workload))
    logf = open(log_path, "w")
    proc = subprocess.Popen(launch + workload, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    output_chunks = []

    def pump():
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            output_chunks.append(line)
            if not args.quiet:
                sys.stdout.write(line)
    t = threading.Thread(target=pump, daemon=True)
    t.start()

    time.sleep(2)
    wpid = find_workload_pid(proc.pid, workload[0])
    samples = []
    numastat_start = read_sys_numastat()
    try:
        while proc.poll() is None:
            snap = {
                "t": round(time.time(), 1),
                "node_free_mib": read_node_free_mib(),
                "proc": read_numa_maps(wpid),
            }
            if events:
                snap["perf"] = perf_sample(events, min(5, max(1, args.interval - 1)))
            samples.append(snap)
            deadline = time.time() + args.interval
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
    proc.wait()
    t.join(timeout=5)
    logf.close()

    numastat_end = read_sys_numastat()
    deltas = {}
    for node, counters in numastat_end.items():
        deltas[str(node)] = {k: counters[k] - numastat_start.get(node, {}).get(k, 0)
                             for k in counters}

    text = "".join(output_chunks)
    metrics = parse_llama_output(text)

    # AMD uProf: one raw capture is only useful while the workload runs, so it is
    # not attempted here post-hoc; point users at the doc instead.
    result = {
        "name": args.name,
        "timestamp": stamp,
        "cmd": workload,
        "allowed_cpus": format_cpulist(allowed) if allowed else "all",
        "exit_code": proc.returncode,
        "metrics": metrics,
        "numastat_delta": deltas,
        "samples": samples,
        "log": os.path.basename(log_path),
    }
    out_path = os.path.join(args.results_dir, tag + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)

    print("\n=== %s ===" % args.name)
    print("exit code : %s" % proc.returncode)
    print("pp t/s    : %s" % metrics.get("pp_tps"))
    print("tg t/s    : %s" % metrics.get("tg_tps"))
    last = next((s["proc"] for s in reversed(samples) if s.get("proc")), None)
    if last:
        print("proc RSS per node (MiB): %s" % last["per_node_mib"])
        print("largest mappings:")
        for e in last["top_mappings"][:5]:
            print("  %8.0f MiB  %-12s  %s" % (e["mib"], e["label"], e["per_node_mib"]))
    perfs = [s["perf"]["remote_ratio_pct"] for s in samples if s.get("perf")]
    if perfs:
        print("remote DRAM-fill ratio: avg %.2f%% (n=%d)" % (sum(perfs) / len(perfs), len(perfs)))
    print("saved: %s" % out_path)


# ---------------------------------------------------------------- monitor

def cmd_monitor(args):
    events = probe_perf_events() if args.bw else None
    prev = read_sys_numastat()
    print("Ctrl+C to stop")
    while True:
        time.sleep(args.interval)
        now = read_sys_numastat()
        maps = read_numa_maps(args.pid)
        stampstr = datetime.datetime.now().strftime("%H:%M:%S")
        if maps is None:
            print("%s pid %d gone" % (stampstr, args.pid))
            return
        line = "%s rss/node MiB %s" % (stampstr, maps["per_node_mib"])
        misses = {n: now[n].get("numa_miss", 0) - prev.get(n, {}).get("numa_miss", 0) for n in now}
        line += "  numa_miss/int %s" % misses
        if events:
            p = perf_sample(events, min(5, args.interval))
            if p:
                line += "  remote-fill %.2f%%" % p["remote_ratio_pct"]
        print(line)
        prev = now


# ---------------------------------------------------------------- report

def cmd_report(args):
    rows = []
    if not os.path.isdir(args.results_dir):
        print("no results in %s" % args.results_dir)
        return
    for fn in sorted(os.listdir(args.results_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(args.results_dir, fn)) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        last = next((s["proc"] for s in reversed(r.get("samples", [])) if s.get("proc")), None)
        perfs = [s["perf"]["remote_ratio_pct"] for s in r.get("samples", []) if s.get("perf")]
        rows.append({
            "name": r["name"],
            "when": r["timestamp"],
            "pp": r["metrics"].get("pp_tps"),
            "tg": r["metrics"].get("tg_tps"),
            "rss": last["per_node_mib"] if last else {},
            "remote": round(sum(perfs) / len(perfs), 2) if perfs else None,
            "cpus": r.get("allowed_cpus", "?"),
        })
    if not rows:
        print("no results in %s" % args.results_dir)
        return
    fmt = "%-20s %-15s %10s %10s %9s %-28s %s"
    print(fmt % ("name", "when", "pp t/s", "tg t/s", "remote%", "rss per node (MiB)", "cpus"))
    for r in rows:
        print(fmt % (r["name"], r["when"],
                     r["pp"] if r["pp"] is not None else "-",
                     r["tg"] if r["tg"] is not None else "-",
                     r["remote"] if r["remote"] is not None else "-",
                     json.dumps(r["rss"]), r["cpus"]))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run a workload and record placement + throughput")
    p.add_argument("--name", required=True, help="label for this configuration")
    p.add_argument("--allowed-cpus", help="explicit AllowedCPUs list, e.g. 16-191,208-383")
    p.add_argument("--reserve-node0", type=int, default=0,
                   help="exclude the first N physical cores of node 0 (plus SMT siblings)")
    p.add_argument("--interval", type=int, default=10, help="telemetry sample interval, seconds")
    p.add_argument("--bw-tool", choices=["auto", "perf", "none"], default="auto")
    p.add_argument("--quiet", action="store_true", help="do not echo workload output")
    p.add_argument("--results-dir", default=RESULTS_DIR)

    p = sub.add_parser("monitor", help="live placement view of a running process")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--bw", action="store_true", help="also sample perf DRAM-fill counters")

    p = sub.add_parser("report", help="tabulate recorded runs")
    p.add_argument("--results-dir", default=RESULTS_DIR)

    argv = sys.argv[1:]
    workload = []
    if "--" in argv:
        i = argv.index("--")
        workload = argv[i + 1:]
        argv = argv[:i]
    args = ap.parse_args(argv)

    if args.cmd != "report" and sys.platform != "linux":
        sys.exit("this tool must run on the Linux server (only 'report' works elsewhere)")

    if args.cmd == "run":
        if not workload:
            sys.exit("run: pass the workload after '--', e.g. numa-ab.py run --name a -- ./llama-cli ...")
        cmd_run(args, workload)
    elif args.cmd == "monitor":
        cmd_monitor(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
