import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = REPORTS_DIR / "logs"
FIGURES_DIR = REPORTS_DIR / "figures"
LIVE_STATUS_PATH = REPORTS_DIR / "live_status.json"
LIVE_EPOCH_CSV = REPORTS_DIR / "epoch_metrics.csv"
LIVE_GPU_CSV = REPORTS_DIR / "gpu_metrics.csv"

PAPER_METRICS = {"accuracy": 0.8300, "f1": 0.8280, "corr": 0.6980, "mae": 0.8710}
ACCEPTANCE = {"accuracy": 0.8150, "f1": 0.8130, "corr": 0.6500, "mae": 0.9500}

BATCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Batch\s+(\d+)/\s*(\d+)\s+\|\s+Time/Batch\(ms\)\s+([0-9.]+)\s+\|\s+Train Loss\s+([0-9.]+)"
)
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Time\s+([0-9.]+)\s+sec\s+\|\s+Valid Loss\s+([0-9.]+)\s+\|\s+Test Loss\s+([0-9.]+)"
)
FLOAT_PATTERNS = {
    "mae": re.compile(r"MAE:\s+([0-9.]+)"),
    "corr": re.compile(r"Correlation Coefficient:\s+([0-9.\-]+)"),
    "mult_acc_7": re.compile(r"mult_acc_7:\s+([0-9.]+)"),
    "mult_acc_5": re.compile(r"mult_acc_5:\s+([0-9.]+)"),
    "f1": re.compile(r"F1 score:\s+([0-9.]+)"),
    "accuracy": re.compile(r"Accuracy:\s+([0-9.]+)"),
    "total_time_s": re.compile(r"Total training time \(s\):\s+([0-9.]+)"),
    "peak_cuda_mem_mb": re.compile(r"Peak CUDA memory allocated \(MB\):\s+([0-9.]+)"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MulT training with low-overhead live monitoring")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable")
    parser.add_argument("--batch-size", type=int, default=32, help="physical batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="epoch count")
    parser.add_argument("--log-interval", type=int, default=1, help="training log interval")
    parser.add_argument("--name", type=str, default="mosi_full_bs32_ga4", help="run name")
    parser.add_argument("--skip-data-setup", action="store_true", help="assume data already exists")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="live monitor polling interval")
    return parser.parse_args()


def query_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        mem_used, util = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
        return {"memory_used_mb": float(mem_used), "gpu_util_pct": float(util)}
    except Exception:
        return {}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_csv_row(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_eta(seconds):
    if seconds is None:
        return None
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_command(args):
    return [
        args.python,
        "-u",
        "main.py",
        "--preset",
        "mosi_paper",
        "--no_prompt",
        "--batch_size",
        str(args.batch_size),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--name",
        args.name,
        "--num_epochs",
        str(args.epochs),
        "--log_interval",
        str(args.log_interval),
    ]


def maybe_setup_data(args):
    if args.skip_data_setup:
        return
    subprocess.run([args.python, str(ROOT / "scripts" / "setup_mosi_data.py")], cwd=ROOT, check=True)


def evaluate_acceptance(metrics):
    if not metrics:
        return {"pass": False, "reason": "missing_final_metrics"}
    return {
        "pass": (
            metrics.get("accuracy", 0) >= ACCEPTANCE["accuracy"]
            and metrics.get("f1", 0) >= ACCEPTANCE["f1"]
            and metrics.get("corr", 0) >= ACCEPTANCE["corr"]
            and metrics.get("mae", float("inf")) <= ACCEPTANCE["mae"]
        )
    }


def generate_plots(epoch_rows, final_metrics):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_path = FIGURES_DIR / "mosi_metric_comparison.png"
    mae_path = FIGURES_DIR / "mosi_mae_comparison.png"
    loss_path = FIGURES_DIR / "mosi_loss_curves.png"

    compare_keys = ["accuracy", "f1", "corr"]
    x = range(len(compare_keys))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], [PAPER_METRICS[k] for k in compare_keys], width=width, label="Paper")
    plt.bar([i + width / 2 for i in x], [final_metrics.get(k, 0.0) for k in compare_keys], width=width, label="Reproduced")
    plt.xticks(list(x), ["Accuracy", "F1", "Corr"])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("MulT MOSI Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(["Paper", "Reproduced"], [PAPER_METRICS["mae"], final_metrics.get("mae", 0.0)], color=["#4E79A7", "#E15759"])
    plt.ylabel("MAE")
    plt.title("MulT MOSI MAE (lower is better)")
    plt.tight_layout()
    plt.savefig(mae_path, dpi=160)
    plt.close()

    if epoch_rows:
        epochs = [row["epoch"] for row in epoch_rows]
        valid_losses = [row["valid_loss"] for row in epoch_rows]
        test_losses = [row["test_loss"] for row in epoch_rows]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, valid_losses, label="Valid Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.title("MulT MOSI Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=160)
        plt.close()

    return {"metric_comparison": str(metric_path), "mae_comparison": str(mae_path), "loss_curves": str(loss_path)}


def write_report(run_name, log_path, final_metrics, epoch_rows, state, figures):
    report_path = REPORTS_DIR / "mosi_report.md"
    acceptance = evaluate_acceptance(final_metrics)
    lines = [
        "# MulT MOSI Reproduction Report",
        "",
        f"- Run name: `{run_name}`",
        f"- Log: `{log_path}`",
        f"- Final state: `{state}`",
        f"- Acceptance: `{'PASS' if acceptance['pass'] else 'FAIL'}`",
        "",
        "## Paper vs Reproduced",
        "",
        "| Metric | Paper | Reproduced | Acceptance |",
        "| --- | ---: | ---: | ---: |",
        f"| Accuracy | {PAPER_METRICS['accuracy']:.4f} | {final_metrics.get('accuracy', float('nan')):.4f} | >= {ACCEPTANCE['accuracy']:.4f} |",
        f"| F1 | {PAPER_METRICS['f1']:.4f} | {final_metrics.get('f1', float('nan')):.4f} | >= {ACCEPTANCE['f1']:.4f} |",
        f"| Corr | {PAPER_METRICS['corr']:.4f} | {final_metrics.get('corr', float('nan')):.4f} | >= {ACCEPTANCE['corr']:.4f} |",
        f"| MAE | {PAPER_METRICS['mae']:.4f} | {final_metrics.get('mae', float('nan')):.4f} | <= {ACCEPTANCE['mae']:.4f} |",
        "",
        "## Runtime",
        "",
        f"- Total training time (s): `{final_metrics.get('total_time_s', 'n/a')}`",
        f"- Peak CUDA memory (MB): `{final_metrics.get('peak_cuda_mem_mb', 'n/a')}`",
        f"- Epochs completed: `{len(epoch_rows)}`",
        "",
        "## Figures",
        "",
        f"- Metric comparison: `{figures.get('metric_comparison', 'n/a')}`",
        f"- MAE comparison: `{figures.get('mae_comparison', 'n/a')}`",
        f"- Loss curves: `{figures.get('loss_curves', 'n/a')}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_new_lines(lines, args, status, batch_times_ms, epoch_rows, final_metrics):
    for line in lines:
        batch_match = BATCH_RE.search(line)
        if batch_match:
            current_epoch = int(batch_match.group(1))
            current_batch = int(batch_match.group(2))
            total_batches = int(batch_match.group(3))
            batch_time_ms = float(batch_match.group(4))
            train_loss = float(batch_match.group(5))
            batch_times_ms.append(batch_time_ms)

            avg_batch_sec = sum(batch_times_ms) / len(batch_times_ms) / 1000.0
            completed_batches = ((current_epoch - 1) * total_batches) + current_batch
            total_training_batches = args.epochs * total_batches
            eta_seconds = (total_training_batches - completed_batches) * avg_batch_sec

            status.update(
                {
                    "current_epoch": current_epoch,
                    "current_batch": current_batch,
                    "total_batches": total_batches,
                    "last_train_loss": train_loss,
                    "eta_seconds": eta_seconds,
                    "eta_hms": format_eta(eta_seconds),
                }
            )

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            row = {
                "epoch": int(epoch_match.group(1)),
                "epoch_time_s": float(epoch_match.group(2)),
                "valid_loss": float(epoch_match.group(3)),
                "test_loss": float(epoch_match.group(4)),
            }
            epoch_rows.append(row)
            append_csv_row(LIVE_EPOCH_CSV, ["epoch", "epoch_time_s", "valid_loss", "test_loss"], row)
            status.update(
                {
                    "current_epoch": row["epoch"],
                    "last_valid_loss": row["valid_loss"],
                    "last_test_loss": row["test_loss"],
                }
            )

        for key, pattern in FLOAT_PATTERNS.items():
            match = pattern.search(line)
            if match:
                final_metrics[key] = float(match.group(1))


def read_new_log_lines(handle):
    lines = []
    while True:
        position = handle.tell()
        line = handle.readline()
        if not line:
            handle.seek(position)
            break
        lines.append(line.rstrip("\n"))
    return lines


def main():
    args = parse_args()
    maybe_setup_data(args)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        LIVE_STATUS_PATH,
        LIVE_EPOCH_CSV,
        LIVE_GPU_CSV,
        REPORTS_DIR / "mosi_comparison.json",
        REPORTS_DIR / "mosi_report.md",
    ):
        if path.exists():
            path.unlink()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{args.name}_{timestamp}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = build_command(args)
    print("Launching:", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8", buffering=1) as train_log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=train_log, stderr=subprocess.STDOUT, text=True)

    overall_start = time.time()
    last_gpu_poll = 0.0
    last_console_print = 0.0
    batch_times_ms = []
    epoch_rows = []
    final_metrics = {}
    status = {
        "state": "running",
        "pid": process.pid,
        "run_name": args.name,
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(),
        "current_epoch": 0,
        "current_batch": 0,
        "total_batches": None,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "eta_hms": None,
    }
    write_json(LIVE_STATUS_PATH, status)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as reader:
        try:
            while True:
                new_lines = read_new_log_lines(reader)
                if new_lines:
                    parse_new_lines(new_lines, args, status, batch_times_ms, epoch_rows, final_metrics)
                status["elapsed_seconds"] = time.time() - overall_start

                now = time.time()
                if now - last_gpu_poll >= args.poll_seconds:
                    gpu = query_gpu()
                    if gpu:
                        status.update(gpu)
                        append_csv_row(
                            LIVE_GPU_CSV,
                            ["timestamp", "memory_used_mb", "gpu_util_pct"],
                            {
                                "timestamp": datetime.now().isoformat(),
                                "memory_used_mb": gpu["memory_used_mb"],
                                "gpu_util_pct": gpu["gpu_util_pct"],
                            },
                        )
                    last_gpu_poll = now

                write_json(LIVE_STATUS_PATH, status)

                if now - last_console_print >= 30:
                    print(
                        f"epoch={status.get('current_epoch', 0)} "
                        f"batch={status.get('current_batch', 0)}/{status.get('total_batches', '?')} "
                        f"elapsed={format_eta(status.get('elapsed_seconds'))} "
                        f"eta={status.get('eta_hms')}"
                    )
                    last_console_print = now

                if process.poll() is not None:
                    remaining_lines = read_new_log_lines(reader)
                    if remaining_lines:
                        parse_new_lines(remaining_lines, args, status, batch_times_ms, epoch_rows, final_metrics)
                    break

                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            process.terminate()
            status["state"] = "interrupted"
            status["ended_at"] = datetime.now().isoformat()
            write_json(LIVE_STATUS_PATH, status)
            raise

    return_code = process.wait()
    status["state"] = "completed" if return_code == 0 else "failed"
    status["returncode"] = return_code
    status["ended_at"] = datetime.now().isoformat()
    status["elapsed_seconds"] = time.time() - overall_start
    status["eta_seconds"] = 0
    status["eta_hms"] = "00:00:00"
    status["final_metrics"] = final_metrics
    write_json(LIVE_STATUS_PATH, status)

    figures = generate_plots(epoch_rows, final_metrics) if final_metrics else {}
    report_path = write_report(args.name, log_path, final_metrics, epoch_rows, status["state"], figures)
    write_json(
        REPORTS_DIR / "mosi_comparison.json",
        {
            "paper_metrics": PAPER_METRICS,
            "reproduced_metrics": final_metrics,
            "acceptance": evaluate_acceptance(final_metrics),
            "log_path": str(log_path),
            "report_path": str(report_path),
            "figure_paths": figures,
        },
    )
    print(json.dumps({"status": status, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
