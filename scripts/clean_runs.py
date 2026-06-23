from __future__ import annotations

import argparse
import csv
import fnmatch
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
MANIFEST_PATH = ROOT / "experiment_records" / "run_manifest.csv"

CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.ckpt")
DELETE_RUN_PATTERNS = (
    "smoke_*",
    "*gradcheck_tmp*",
    "dvq_fixed_c*_lr1e4_3000_20260623",
    "code_logits_all_triplets_multipos*",
)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def collect_checkpoint_files() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    files: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        files.extend(RUNS_DIR.rglob(pattern))
    return sorted(set(files))


def should_delete_run_dir(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in DELETE_RUN_PATTERNS)


def load_key_runs() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        return {row["run_name"] for row in csv.DictReader(f) if row.get("is_key_run") == "yes"}


def collect_run_dirs(*, delete_non_key_runs: bool) -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    key_runs = load_key_runs()
    dirs = []
    for path in RUNS_DIR.iterdir():
        if not path.is_dir():
            continue
        if should_delete_run_dir(path):
            dirs.append(path)
        elif delete_non_key_runs and path.name not in key_runs:
            dirs.append(path)
    return sorted(set(dirs))


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def print_plan(files: list[Path], dirs: list[Path], limit: int) -> None:
    total_file_bytes = sum(file_size(path) for path in files)
    print(f"Checkpoint files to delete: {len(files)} ({format_bytes(total_file_bytes)})")
    for path in files[:limit]:
        print(f"  file {path.relative_to(ROOT)}")
    if len(files) > limit:
        print(f"  ... {len(files) - limit} more checkpoint files")
    print(f"Run directories to delete: {len(dirs)}")
    for path in dirs[:limit]:
        print(f"  dir  {path.relative_to(ROOT)}")
    if len(dirs) > limit:
        print(f"  ... {len(dirs) - limit} more run directories")


def apply_cleanup(files: list[Path], dirs: list[Path]) -> None:
    for path in files:
        path.unlink(missing_ok=True)
    for path in dirs:
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run or apply V5 run artifact cleanup.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files/directories.")
    parser.add_argument(
        "--delete-non-key-runs",
        action="store_true",
        help="Delete run directories not marked as key runs in experiment_records/run_manifest.csv.",
    )
    parser.add_argument("--list-limit", type=int, default=80, help="Maximum items to print per category.")
    args = parser.parse_args()

    files = collect_checkpoint_files()
    dirs = collect_run_dirs(delete_non_key_runs=args.delete_non_key_runs)
    print_plan(files, dirs, args.list_limit)
    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete these artifacts.")
        return
    apply_cleanup(files, dirs)
    print("Cleanup applied.")


if __name__ == "__main__":
    main()
