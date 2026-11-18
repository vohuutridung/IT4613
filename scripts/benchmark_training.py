#!/usr/bin/env python3
"""Compare FP32, FP16 and opt-in XLA using identical candidate streams.

Each case uses the full epoch's LR horizon but stops after a short prefix. Runs
are isolated and never submitted. XLA failure is recorded without changing the
default training path. Launch through run.sh --mode benchmark for the TF env.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--effective-batch", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--prefetch-batches", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    args = ap.parse_args()
    if args.updates <= 5:
        ap.error("updates must exceed the 5 warmup updates")
    root = Path(args.work_root)
    cases = (("fp32", "float32", False), ("fp16", "mixed_float16", False),
             ("fp16-xla", "mixed_float16", True))
    # A resumed prefix would not be comparable to the other fresh cases.
    for name, _, _ in cases:
        if (root / name).exists():
            ap.error("benchmark case already exists; choose a fresh work-root: " + str(root / name))
    rows = []
    for name, precision, xla in cases:
        work = root / name
        command = [sys.executable, str(ROOT / "deployment/kaggle/runner.py"),
                   "--mode", "train", "--prepared-root", args.prepared_root,
                   "--work-root", str(work), "--partition", str(args.partition),
                   "--seed", str(args.seed), "--effective-batch", str(args.effective_batch),
                   "--batch-size", str(args.batch_size), "--replicas", str(args.replicas),
                   "--precision", precision, "--prefetch-batches", str(args.prefetch_batches),
                   "--checkpoint-every", str(args.checkpoint_every),
                   "--stop-after-updates-this-run", str(args.updates),
                   "--train-step-xla" if xla else "--no-train-step-xla"]
        print("Benchmark: " + name, flush=True)
        result = subprocess.run(command, cwd=ROOT)
        row = {"case": name, "precision": precision, "xla": xla,
               "exit_code": result.returncode}
        report = work / "recommendation_runs" / ("partition-" + str(args.partition)) / "training_progress.json"
        if result.returncode == 0 and report.is_file():
            row["performance"] = json.loads(report.read_text())["performance"]
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return int(any(row["exit_code"] for row in rows if not row["xla"]))


if __name__ == "__main__":
    sys.exit(main())
