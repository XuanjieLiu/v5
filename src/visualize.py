from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metrics(metrics_csv: Path, output_path: Path) -> None:
    if not metrics_csv.exists():
        return
    rows = []
    with metrics_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ["train_loss", "eval_loss", "train_plus_loss", "eval_plus_loss"]:
        if key in rows[0]:
            axes[0].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")
    axes[0].legend(fontsize=8)
    for key in ["train_number_accuracy", "eval_number_accuracy", "train_code_accuracy", "eval_code_accuracy"]:
        if key in rows[0]:
            axes[1].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_heatmap(matrix: np.ndarray, output_path: Path, title: str, total: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    missing = ~np.isfinite(matrix)
    if total is not None:
        missing |= np.asarray(total) <= 0
    masked = np.ma.array(matrix, mask=missing)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f4f4f4")
    im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap)
    valid_count = int(np.size(matrix) - np.count_nonzero(missing))
    ax.set_title(f"{title} ({valid_count} pairs; blank = absent)")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_xticks(range(0, 21, 2))
    ax.set_yticks(range(0, 21, 2))
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_title("Target vs predicted number")
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    ax.set_xticks(range(0, 21, 2))
    ax.set_yticks(range(0, 21, 2))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    return tensor.permute(1, 2, 0).numpy()


@torch.no_grad()
def save_triplet_grid(
    addition_model,
    v3_wrapper,
    loader,
    output_path: Path,
    num_examples: int,
    sample_seed: int,
    step: int,
    source_name: str = "eval/test",
) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and len(dataset) > 0:
        rng = random.Random(int(sample_seed) + int(step) * 1009)
        count = min(int(num_examples), len(dataset))
        indices = rng.sample(range(len(dataset)), count)
        examples = [dataset[index] for index in indices]
        images = torch.stack([example["images"] for example in examples], dim=0)
        numbers = torch.stack([example["numbers"] for example in examples], dim=0)
    else:
        batch = next(iter(loader))
        images = batch["images"][:num_examples]
        numbers = batch["numbers"][:num_examples]
    images = images.to(v3_wrapper.device)
    _, emb_s, content_vq, _, _ = v3_wrapper.encode_and_quantize(images)
    _, pred_vq, _, _ = addition_model.plus_content(v3_wrapper, content_vq[:, 0], content_vq[:, 1])
    pred_imgs = v3_wrapper.decode_for_vis(pred_vq.unsqueeze(1), emb_s[:, 2:3])[:, 0]

    rows = images.size(0)
    fig, axes = plt.subplots(rows, 4, figsize=(8, max(2, rows * 1.5)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    headers = ["a", "b", "target c", "pred c"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header)
    fig.suptitle(f"Random {source_name} triplets at step {step}", y=0.995)
    for row in range(rows):
        row_imgs = [images[row, 0], images[row, 1], images[row, 2], pred_imgs[row]]
        for col, img in enumerate(row_imgs):
            axes[row, col].imshow(_to_image(img))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(
            f"{int(numbers[row, 0])}+{int(numbers[row, 1])}={int(numbers[row, 2])}",
            rotation=0,
            labelpad=35,
            va="center",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_visuals(
    run_dir: Path,
    metrics_csv: Path,
    train_eval: Dict[str, object],
    eval_eval: Dict[str, object],
    addition_model,
    v3_wrapper,
    eval_loader,
    step: int,
    num_examples: int,
    sample_seed: int,
) -> None:
    vis_dir = run_dir / "vis"
    _ensure_dir(vis_dir)
    plot_metrics(metrics_csv, vis_dir / "metrics.png")
    plot_heatmap(
        train_eval["heatmap"],
        vis_dir / "train_addition_heatmap.png",
        "Train addition accuracy",
        total=train_eval.get("heatmap_total"),
    )
    plot_heatmap(
        eval_eval["heatmap"],
        vis_dir / "eval_addition_heatmap.png",
        "Eval addition accuracy",
        total=eval_eval.get("heatmap_total"),
    )
    plot_confusion(eval_eval["confusion"], vis_dir / "eval_confusion.png")
    save_triplet_grid(
        addition_model=addition_model,
        v3_wrapper=v3_wrapper,
        loader=eval_loader,
        output_path=vis_dir / f"triplet_grid_step_{step}.png",
        num_examples=num_examples,
        sample_seed=sample_seed,
        step=step,
    )
