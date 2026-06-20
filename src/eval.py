from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from .data import UppercasePageIndex
from .plus_model import plus_once, full_symmetry_loss


def estimate_codebook_mapping(
    v3_wrapper,
    letter_dir: str | Path,
    batch_size: int,
    fragment_len: int,
    max_pages: int,
) -> Dict[str, object]:
    page_index = UppercasePageIndex(letter_dir, fragment_len=fragment_len)
    fragments, labels = page_index.load_all_fragments(max_pages=max_pages)
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for start in range(0, fragments.size(0), batch_size):
        batch = fragments[start : start + batch_size].unsqueeze(1).to(v3_wrapper.device)
        with torch.no_grad():
            _, _, _, indices, _ = v3_wrapper.encode_and_quantize(batch)
        idx_flat = indices.reshape(-1).detach().cpu()
        label_flat = labels[start : start + batch_size].reshape(-1)
        for atom_idx, label_idx in zip(idx_flat.tolist(), label_flat.tolist()):
            counts[int(atom_idx)][int(label_idx)] += 1

    code_to_letter: Dict[str, str] = {}
    code_to_label: Dict[str, int] = {}
    code_counts: Dict[str, Dict[str, int]] = {}
    for atom_idx, label_counts in counts.items():
        best_label = max(label_counts.items(), key=lambda item: item[1])[0]
        code_to_label[str(atom_idx)] = int(best_label)
        code_to_letter[str(atom_idx)] = chr(ord("A") + int(best_label))
        code_counts[str(atom_idx)] = {str(k): int(v) for k, v in label_counts.items()}
    return {
        "code_to_letter": code_to_letter,
        "code_to_label": code_to_label,
        "counts": code_counts,
        "max_pages": max_pages,
    }


def _indices_to_numbers(indices: torch.Tensor, code_to_number: Dict[int, int]) -> torch.Tensor:
    flat = indices.reshape(-1).detach().cpu().tolist()
    mapped = [code_to_number.get(int(idx), -1) for idx in flat]
    return torch.tensor(mapped, dtype=torch.long).reshape(indices.shape)


@torch.no_grad()
def evaluate_loader(
    plus_net,
    v3_wrapper,
    loader,
    code_to_number: Dict[int, int],
    use_symmetry: bool,
    plus_loss_weight: float,
    symmetry_loss_weight: float,
    pred_commit_loss_weight: float,
) -> Dict[str, object]:
    plus_net.eval()
    total = 0
    pred_code_correct = 0
    pred_number_correct = 0
    loss_total_sum = 0.0
    plus_loss_sum = 0.0
    symmetry_loss_sum = 0.0
    commit_loss_sum = 0.0
    heatmap_correct = torch.zeros(21, 21, dtype=torch.float32)
    heatmap_total = torch.zeros(21, 21, dtype=torch.float32)
    confusion = torch.zeros(21, 21, dtype=torch.float32)

    for batch in loader:
        images = batch["images"].to(v3_wrapper.device, non_blocking=True)
        numbers = batch["numbers"].to(v3_wrapper.device, non_blocking=True)
        _, _, content_vq, target_indices, _ = v3_wrapper.encode_and_quantize(images)
        a = content_vq[:, 0]
        b = content_vq[:, 1]
        c = content_vq[:, 2]
        _, pred_vq, pred_indices, pred_commit = plus_once(plus_net, v3_wrapper, a, b)
        plus_loss = F.mse_loss(pred_vq, c)
        symmetry_loss = torch.zeros((), device=v3_wrapper.device)
        symmetry_commit = torch.zeros((), device=v3_wrapper.device)
        if use_symmetry:
            symmetry_loss, symmetry_commit = full_symmetry_loss(plus_net, v3_wrapper, a, b, c)
        commit_loss = pred_commit + symmetry_commit
        total_loss = (
            plus_loss_weight * plus_loss
            + symmetry_loss_weight * symmetry_loss
            + pred_commit_loss_weight * commit_loss
        )

        batch_size = images.size(0)
        total += batch_size
        loss_total_sum += float(total_loss.item()) * batch_size
        plus_loss_sum += float(plus_loss.item()) * batch_size
        symmetry_loss_sum += float(symmetry_loss.item()) * batch_size
        commit_loss_sum += float(commit_loss.item()) * batch_size

        target_c_indices = target_indices[:, 2].reshape_as(pred_indices)
        code_correct = (pred_indices == target_c_indices).detach().cpu()
        pred_code_correct += int(code_correct.sum().item())

        pred_numbers = _indices_to_numbers(pred_indices, code_to_number).to(numbers.device)
        target_numbers = numbers[:, 2].reshape_as(pred_numbers)
        number_correct = pred_numbers == target_numbers
        pred_number_correct += int(number_correct.sum().item())

        for i in range(batch_size):
            left = int(numbers[i, 0].item())
            right = int(numbers[i, 1].item())
            target = int(target_numbers[i].item())
            pred = int(pred_numbers[i].item())
            heatmap_total[left, right] += 1
            if bool(number_correct[i].item()):
                heatmap_correct[left, right] += 1
            if 0 <= pred <= 20:
                confusion[target, pred] += 1

    denom = max(total, 1)
    heatmap = torch.full((21, 21), float("nan"))
    valid = heatmap_total > 0
    heatmap[valid] = heatmap_correct[valid] / heatmap_total[valid]
    return {
        "loss": loss_total_sum / denom,
        "plus_loss": plus_loss_sum / denom,
        "symmetry_loss": symmetry_loss_sum / denom,
        "commit_loss": commit_loss_sum / denom,
        "code_accuracy": pred_code_correct / denom,
        "number_accuracy": pred_number_correct / denom,
        "count": total,
        "heatmap": heatmap.numpy(),
        "confusion": confusion.numpy(),
    }

