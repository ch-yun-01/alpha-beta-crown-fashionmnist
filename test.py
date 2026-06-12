"""
test.py
-------
Runs alpha-beta-CROWN on our external models (FashionMNIST CNN and the MNIST FC
network reused from Assignment #3) and summarizes the verification results, as
required by the assignment.

For each YAML config it launches `abcrown.py --config <config>` as a subprocess
inside the cloned alpha-beta-CROWN repository, captures the log, and parses the
per-instance results (verified / falsified / timeout) and runtimes into a
per-config summary table.

Usage:
    # Run *all* configs in configs/ (default) — reproduces every experiment.
    python test.py

    # Run a single config.
    python test.py --config configs/fashion_mnist_eps_2_255.yaml

Each config's outputs are written to its own folder:
    results/<config_stem>/results_summary.txt
    results/<config_stem>/log_<config_stem>.txt
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run_abcrown(abcrown_dir: Path, config_path: Path) -> str:
    """Run abcrown.py with the given config and return the full stdout/stderr log."""
    config_abs = config_path.resolve()
    cmd = [sys.executable, "abcrown.py", "--config", str(config_abs)]

    print(f"\n=== Running: {' '.join(cmd)} ===")
    print(f"cwd = {abcrown_dir.resolve()}\n")

    proc = subprocess.run(
        cmd,
        cwd=str(abcrown_dir),
        capture_output=True,
        text=True,
    )

    log = proc.stdout + proc.stderr

    if proc.returncode != 0:
        print("WARNING: abcrown.py exited with non-zero status.")
        print("Last 30 log lines:")
        print("\n".join(log.splitlines()[-30:]))

    return log


def parse_results(log: str):
    """Parse per-instance verification results from the abcrown log.

    Expected examples:
        Result: safe in 3.1415 seconds
        Result: unsafe-pgd in 0.5 seconds
        Result: timeout
        Result: unknown
    """
    results = []

    pattern = re.compile(
        r"Result:\s+([a-zA-Z0-9_\-]+)(?:\s+in\s+([\d.]+)\s+seconds)?"
    )

    for line in log.splitlines():
        m = pattern.search(line)
        if m:
            status = m.group(1)
            runtime = float(m.group(2)) if m.group(2) is not None else None
            results.append((status, runtime))

    return results


def normalize_status(status: str) -> str:
    """Map abcrown status strings to the assignment categories."""
    s = status.lower()

    if "unsafe" in s:
        return "falsified"

    if "safe" in s:
        return "verified"

    if "timeout" in s or "unknown" in s:
        return "timeout"

    return "unknown"


def summarize(config_path: Path, results, out_lines):
    """Append a human-readable summary for one config."""
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
    for cat in ("verified", "falsified", "timeout", "unknown"):
        if counter.get(cat, 0) > 0 or cat != "unknown":
            out_lines.append(f"  {cat:<10}: {counter.get(cat, 0)}")

    if times:
        out_lines.append(f"  mean time : {sum(times) / len(times):.2f} s")
        out_lines.append(f"  max time  : {max(times):.2f} s")
    else:
        out_lines.append("  mean time : N/A")
        out_lines.append("  max time  : N/A")


def run_one_config(abcrown_dir: Path, cfg: Path, results_root: Path):
    """Run one config and write its summary + log into results/<config_stem>/."""
    out_dir = results_root / cfg.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log = run_abcrown(abcrown_dir, cfg)

    log_file = out_dir / f"log_{cfg.stem}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log)
    print(f"Full log saved to {log_file}")

    results = parse_results(log)

    out_lines = [
        "alpha-beta-CROWN verification summary",
        "=" * 45,
    ]

    if not results:
        out_lines.append(
            f"\nConfig: {cfg} -> no per-instance results parsed "
            f"(check {log_file} for errors or changed log format)"
        )
    else:
        summarize(cfg, results, out_lines)

    summary = "\n".join(out_lines)
    summary_file = out_dir / "results_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(summary)
    print(f"\nSummary saved to {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run alpha-beta-CROWN verification and summarize results."
    )

    parser.add_argument(
        "--abcrown_dir",
        type=str,
        default="alpha-beta-CROWN/complete_verifier",
        help="Path to alpha-beta-CROWN/complete_verifier",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run a single YAML config. If omitted, all configs in configs/ are run.",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory for per-config result folders.",
    )

    args = parser.parse_args()

    project_root = Path.cwd()
    abcrown_dir = Path(args.abcrown_dir)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    abcrown_py = abcrown_dir / "abcrown.py"
    if not abcrown_py.is_file():
        sys.exit(
            f"abcrown.py not found in {abcrown_dir}.\n"
            "Clone alpha-beta-CROWN under the project root first:\n"
            "  git clone --recursive "
            "https://github.com/Verified-Intelligence/alpha-beta-CROWN.git"
        )

    # No --config given -> run every config in configs/ (full reproduction).
    if args.config is None:
        config_dir = project_root / "configs"
        configs = sorted(config_dir.glob("*.yaml"))
        if not configs:
            sys.exit("No YAML config files found in configs/.")
        print(f"No --config given: running all {len(configs)} configs in configs/.")
    else:
        configs = [Path(args.config)]

    for cfg in configs:
        if not cfg.is_file():
            print(f"Config not found, skipping: {cfg}")
            continue
        run_one_config(abcrown_dir, cfg, results_root)

    print("\nAll requested configs complete.")


if __name__ == "__main__":
    main()