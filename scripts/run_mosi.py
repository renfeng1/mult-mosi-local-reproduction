import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = REPORTS_DIR / "logs"
DEFAULT_ATTEMPTS = [(128, 1), (64, 2), (32, 4), (16, 8), (8, 16)]
METRIC_PATTERNS = {
    "mae": r"MAE:\s+([0-9.]+)",
    "corr": r"Correlation Coefficient:\s+([0-9.\-]+)",
    "mult_acc_7": r"mult_acc_7:\s+([0-9.]+)",
    "f1": r"F1 score:\s+([0-9.]+)",
    "accuracy": r"Accuracy:\s+([0-9.]+)",
    "total_time_s": r"Total training time \(s\):\s+([0-9.]+)",
    "peak_cuda_mem_mb": r"Peak CUDA memory allocated \(MB\):\s+([0-9.]+)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MulT MOSI reproduction with OOM fallback")
    parser.add_argument("--smoke", action="store_true", help="run a 1-epoch smoke test")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="python executable to use for subprocesses",
    )
    parser.add_argument("--skip-data-setup", action="store_true", help="assume data is ready")
    return parser.parse_args()


def run_data_setup(python_executable):
    cmd = [python_executable, str(ROOT / "scripts" / "setup_mosi_data.py")]
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_command(python_executable, batch_size, grad_accum_steps, smoke):
    run_name = "mosi_smoke" if smoke else "mosi_full"
    cmd = [
        python_executable,
        "main.py",
        "--preset",
        "mosi_paper",
        "--no_prompt",
        "--batch_size",
        str(batch_size),
        "--grad_accum_steps",
        str(grad_accum_steps),
        "--name",
        f"{run_name}_bs{batch_size}_ga{grad_accum_steps}",
    ]
    if smoke:
        cmd.extend(["--num_epochs", "1", "--log_interval", "1"])
    return cmd


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def extract_metrics(log_text):
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = re.findall(pattern, log_text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def attempt_run(python_executable, batch_size, grad_accum_steps, smoke):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mode = "smoke" if smoke else "full"
    log_path = LOGS_DIR / f"{mode}_bs{batch_size}_ga{grad_accum_steps}_{timestamp}.log"
    cmd = build_command(python_executable, batch_size, grad_accum_steps, smoke)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    process = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(process.stdout, encoding="utf-8", errors="ignore")

    output_lower = process.stdout.lower()
    oom = "out of memory" in output_lower or "cuda error: out of memory" in output_lower
    summary = {
        "mode": mode,
        "returncode": process.returncode,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "log_path": str(log_path),
        "oom": oom,
        "metrics": extract_metrics(process.stdout),
    }
    write_json(log_path.with_suffix(".json"), summary)
    return summary


def main():
    args = parse_args()
    if not args.skip_data_setup:
        run_data_setup(args.python)

    attempts = DEFAULT_ATTEMPTS
    final_summary = None
    for batch_size, grad_accum_steps in attempts:
        summary = attempt_run(args.python, batch_size, grad_accum_steps, args.smoke)
        final_summary = summary
        if summary["returncode"] == 0:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        if not summary["oom"]:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            raise SystemExit(summary["returncode"])
        print(f"OOM at batch_size={batch_size}, grad_accum_steps={grad_accum_steps}; retrying...")

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    raise SystemExit(final_summary["returncode"] if final_summary else 1)


if __name__ == "__main__":
    main()
