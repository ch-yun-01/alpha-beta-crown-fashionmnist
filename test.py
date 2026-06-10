"""
test.py
-------
Demonstrates running alpha-beta-CROWN on our external model (FashionMNIST CNN)
and summarizes the verification results, as required by the assignment.

It launches `abcrown.py --config <config>` as a subprocess inside the cloned
alpha-beta-CROWN repository, captures the log, and parses per-instance results
(verified / falsified / timeout) and runtimes into a summary table.

Usage (after running setup steps in README.md):
    python test.py --abcrown_dir ../alpha-beta-CROWN/complete_verifier \
                   --config configs/fashion_mnist_eps_2_255.yaml

Add --all to run every config in the configs/ directory.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter


def run_abcrown(abcrown_dir, config_path):
    """Run abcrown.py with the given config and return the full stdout log."""
    config_abs = os.path.abspath(config_path)
    cmd = [sys.executable, "abcrown.py", "--config", config_abs]
    print(f"\n=== Running: {' '.join(cmd)} (cwd={abcrown_dir}) ===\n")
    proc = subprocess.run(cmd, cwd=abcrown_dir, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print("WARNING: abcrown.py exited with non-zero status. Last lines:")
        print("\n".join(log.splitlines()[-30:]))
    return log


def parse_results(log):
    """Parse per-instance verification results from the abcrown log.

    abcrown prints lines like:
        Result: safe in 3.1415 seconds
        Result: unsafe-pgd in 0.5 seconds
        Result: timeout
    and a final summary block ("final verified acc", "mean time", ...).
    """
    results = []
    # Matches e.g. "Result: safe in 12.3456 seconds" or "Result: timeout"
    pattern = re.compile(
        r"Result:\s+([a-zA-Z\-_]+)(?:\s+in\s+([\d.]+)\s+seconds)?")
    for line in log.splitlines():
        m = pattern.search(line)
        if m:
            status = m.group(1)
            runtime = float(m.group(2)) if m.group(2) else None
            results.append((status, runtime))
    return results


def normalize_status(status):
    """Map abcrown status strings to the three categories in the assignment."""
    s = status.lower()
    if "unsafe" in s:          # unsafe-pgd / unsafe-bab -> counterexample found
        return "falsified"
    if "safe" in s:            # safe / safe-incomplete -> property holds
        return "verified"
    if "timeout" in s or "unknown" in s:
        return "timeout"
    return s


def summarize(config_path, results, out_lines):
    out_lines.append(f"\nConfig: {config_path}")
    out_lines.append(f"{'idx':>4}  {'raw status':<18} {'category':<10} {'time (s)':>9}")
    counter = Counter()
    times = []
    for i, (status, runtime) in enumerate(results):
        cat = normalize_status(status)
        counter[cat] += 1
        if runtime is not None:
            times.append(runtime)
        t = f"{runtime:9.2f}" if runtime is not None else "        -"
        out_lines.append(f"{i:>4}  {status:<18} {cat:<10} {t}")
    total = sum(counter.values())
    out_lines.append(f"\nTotal instances: {total}")
    for cat in ("verified", "falsified", "timeout"):
        out_lines.append(f"  {cat:<10}: {counter.get(cat, 0)}")
    if times:
        out_lines.append(f"  mean time : {sum(times) / len(times):.2f} s")
        out_lines.append(f"  max time  : {max(times):.2f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abcrown_dir", type=str,
                        default="../alpha-beta-CROWN/complete_verifier",
                        help="Path to alpha-beta-CROWN/complete_verifier")
    parser.add_argument("--config", type=str,
                        default="configs/fashion_mnist_eps_2_255.yaml")
    parser.add_argument("--all", action="store_true",
                        help="Run all YAML configs in configs/")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.abcrown_dir, "abcrown.py")):
        sys.exit(f"abcrown.py not found in {args.abcrown_dir}. "
                 "Clone alpha-beta-CROWN first (see README.md).")

    configs = (sorted(os.path.join("configs", f)
                      for f in os.listdir("configs") if f.endswith(".yaml"))
               if args.all else [args.config])

    out_lines = ["alpha-beta-CROWN verification summary",
                 "=" * 45]
    for cfg in configs:
        log = run_abcrown(args.abcrown_dir, cfg)
        log_file = f"log_{os.path.splitext(os.path.basename(cfg))[0]}.txt"
        with open(log_file, "w") as f:
            f.write(log)
        print(f"full log saved to {log_file}")
        results = parse_results(log)
        if not results:
            out_lines.append(f"\nConfig: {cfg} -> no results parsed "
                             f"(check {log_file} for errors)")
            continue
        summarize(cfg, results, out_lines)

    summary = "\n".join(out_lines)
    print("\n" + summary)
    with open("results_summary.txt", "w") as f:
        f.write(summary + "\n")
    print("\nsummary saved to results_summary.txt")


if __name__ == "__main__":
    main()
