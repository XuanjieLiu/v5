from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
RECORDS_DIR = ROOT / "experiment_records"
KEY_RUNS_DIR = RECORDS_DIR / "key_runs"
MANIFEST_PATH = RECORDS_DIR / "run_manifest.csv"


KEY_RUN_NOTES = {
    "lf_direct_vq_plus_mse_ste_5000_20260622": "Direct VQ-Plus full-table baseline.",
    "dvq_fixed_ratio01_h2048x3_c01_lr1e4_5000_20260623": "Direct VQ-Plus small-split overfit, 2048x3.",
    "dvq_fixed_ratio01_h4096x4_c01_lr1e4_5000_20260623": "Direct VQ-Plus small-split overfit, 4096x4; train target reached.",
    "dvq_fixed_ratio03_h2048x3_c01_lr1e4_5000_20260623": "Direct VQ-Plus 30 percent split capacity screen, 2048x3.",
    "dvq_fixed_ratio03_h4096x4_c01_lr1e4_5000_20260623": "Direct VQ-Plus 30 percent split capacity screen, 4096x4.",
    "dvq_fixed_expcommit_c0_lr1e4_3000_20260623": "Direct VQ-Plus explicit commit sweep.",
    "dvq_fixed_expcommit_c001_lr1e4_3000_20260623": "Direct VQ-Plus explicit commit sweep.",
    "dvq_fixed_expcommit_c003_lr1e4_3000_20260623": "Direct VQ-Plus explicit commit sweep.",
    "dvq_fixed_expcommit_c01_lr1e4_3000_20260623": "Direct VQ-Plus explicit commit sweep.",
    "dvq_fixed_expcommit_c0_lr3e4_3000_20260623": "Direct VQ-Plus explicit commit/LR sweep.",
    "dvq_fixed_raw1_vq0_c0_lr1e4_3000_20260623": "Direct VQ-Plus raw-target sweep.",
    "dvq_fixed_raw1_vq1_c0_lr1e4_3000_20260623": "Direct VQ-Plus raw-target sweep.",
    "dvq_fixed_raw1_vq0_c01_lr1e4_3000_20260623": "Direct VQ-Plus raw-target/commit sweep.",
    "dvq_fixed_raw1_vq1_c01_lr1e4_3000_20260623": "Direct VQ-Plus raw-target/commit sweep.",
    "dvq_fixed_raw1_vq0_c0_lr3e4_3000_20260623": "Direct VQ-Plus raw-target/LR sweep.",
    "lf_vq_ce_hard_200_20260621": "Hard target-image VQ CE 200-step screen.",
    "lf_vq_ce_soft_200_20260621": "Soft target-image VQ CE 200-step screen.",
    "lf_vq_ce_hard_1000_20260621": "Hard target-image VQ CE 1000-step run.",
    "lf_vq_ce_hard_resume5k_20260621": "Hard target-image VQ CE 5000-step best number run.",
    "lf_vq_ce_hard_lr3e5_from5k_to6k_20260622": "Hard target-image VQ CE lower-LR continuation.",
    "lf_vq_ce_hard_big_2000_20260622": "Hard target-image VQ CE bigger plus-net screen.",
    "lf_pca_d2_mse_ste_200_20260621": "PCA d=2 MSE+STE screen.",
    "lf_pca_d4_mse_ste_200_20260621": "PCA d=4 MSE+STE screen.",
    "lf_pca_d8_mse_ste_200_20260621": "PCA d=8 MSE+STE screen.",
    "lf_learned_d2_recon_mse_ste_200_20260621": "Learned projector d=2 recon+MSE+STE screen.",
    "lf_learned_d4_recon_mse_ste_200_20260621": "Learned projector d=4 recon+MSE+STE screen.",
    "lf_learned_d8_recon_mse_ste_200_20260621": "Learned projector d=8 recon+MSE+STE screen.",
}


FIELDS = [
    "run_name",
    "is_key_run",
    "note",
    "config_path",
    "metrics_path",
    "num_metric_rows",
    "final_step",
    "train_count",
    "eval_count",
    "best_train_number_step",
    "best_train_number_accuracy",
    "best_train_code_step",
    "best_train_code_accuracy",
    "best_eval_number_step",
    "best_eval_number_accuracy",
    "best_eval_code_step",
    "best_eval_code_accuracy",
    "best_eval_loss_step",
    "best_eval_loss",
    "final_train_number_accuracy",
    "final_train_code_accuracy",
    "final_eval_number_accuracy",
    "final_eval_code_accuracy",
    "final_eval_loss",
]


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def best_row(rows: list[dict[str, str]], key: str, *, maximize: bool) -> dict[str, str]:
    if not rows:
        return {}
    return (max if maximize else min)(rows, key=lambda row: as_float(row, key))


def row_for_run(run_dir: Path) -> dict[str, str]:
    run_name = run_dir.name
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.csv"
    rows = read_metrics(metrics_path)
    final = rows[-1] if rows else {}
    best_train_num = best_row(rows, "train_number_accuracy", maximize=True)
    best_train_code = best_row(rows, "train_code_accuracy", maximize=True)
    best_eval_num = best_row(rows, "eval_number_accuracy", maximize=True)
    best_eval_code = best_row(rows, "eval_code_accuracy", maximize=True)
    best_eval_loss = best_row(rows, "eval_loss", maximize=False)
    return {
        "run_name": run_name,
        "is_key_run": "yes" if run_name in KEY_RUN_NOTES else "no",
        "note": KEY_RUN_NOTES.get(run_name, ""),
        "config_path": str(config_path.relative_to(ROOT)) if config_path.exists() else "",
        "metrics_path": str(metrics_path.relative_to(ROOT)) if metrics_path.exists() else "",
        "num_metric_rows": str(len(rows)),
        "final_step": final.get("step", ""),
        "train_count": final.get("train_count", ""),
        "eval_count": final.get("eval_count", ""),
        "best_train_number_step": best_train_num.get("step", ""),
        "best_train_number_accuracy": best_train_num.get("train_number_accuracy", ""),
        "best_train_code_step": best_train_code.get("step", ""),
        "best_train_code_accuracy": best_train_code.get("train_code_accuracy", ""),
        "best_eval_number_step": best_eval_num.get("step", ""),
        "best_eval_number_accuracy": best_eval_num.get("eval_number_accuracy", ""),
        "best_eval_code_step": best_eval_code.get("step", ""),
        "best_eval_code_accuracy": best_eval_code.get("eval_code_accuracy", ""),
        "best_eval_loss_step": best_eval_loss.get("step", ""),
        "best_eval_loss": best_eval_loss.get("eval_loss", ""),
        "final_train_number_accuracy": final.get("train_number_accuracy", ""),
        "final_train_code_accuracy": final.get("train_code_accuracy", ""),
        "final_eval_number_accuracy": final.get("eval_number_accuracy", ""),
        "final_eval_code_accuracy": final.get("eval_code_accuracy", ""),
        "final_eval_loss": final.get("eval_loss", ""),
    }


def copy_key_run_files(run_dir: Path) -> None:
    if run_dir.name not in KEY_RUN_NOTES:
        return
    dest = KEY_RUNS_DIR / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ["config.yaml", "metrics.csv"]:
        src = run_dir / filename
        if src.exists():
            shutil.copy2(src, dest / filename)


def main() -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    KEY_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()) if RUNS_DIR.exists() else []
    rows = [row_for_run(run_dir) for run_dir in run_dirs]
    for run_dir in run_dirs:
        copy_key_run_files(run_dir)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    key_count = sum(1 for row in rows if row["is_key_run"] == "yes")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)} with {len(rows)} runs ({key_count} key runs).")
    print(f"Copied key run config/metrics under {KEY_RUNS_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
