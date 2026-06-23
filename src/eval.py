from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from .data import UppercasePageIndex
from .plus_model import full_symmetry_loss


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


def target_vq_cross_entropy(
    logits: torch.Tensor,
    target_content: torch.Tensor,
    target_indices: torch.Tensor,
    v3_wrapper,
    *,
    mode: str = "hard",
    soft_temperature: float = 1.0,
    soft_top_k: int = 0,
) -> torch.Tensor:
    if mode == "hard":
        if target_indices.ndim > 1:
            target_indices = target_indices[:, 0]
        return F.cross_entropy(logits, target_indices)
    if mode != "soft":
        raise ValueError("target_vq_ce_mode must be 'hard' or 'soft'.")

    codebook = v3_wrapper.model.vq.codebook.detach().to(target_content.device)
    distances = torch.sum((target_content.unsqueeze(1) - codebook.unsqueeze(0)) ** 2, dim=-1)
    target_logits = -distances / max(float(soft_temperature), 1e-8)
    if soft_top_k > 0 and soft_top_k < target_logits.size(-1):
        _, keep = torch.topk(target_logits, k=int(soft_top_k), dim=-1)
        mask = torch.full_like(target_logits, float("-inf"))
        target_logits = mask.scatter(dim=-1, index=keep, src=target_logits.gather(dim=-1, index=keep))
    target_probs = F.softmax(target_logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _empty_eval_accumulators() -> dict:
    return {
        "total": 0,
        "pred_code_correct": 0,
        "pred_number_correct": 0,
        "loss_total_sum": 0.0,
        "plus_loss_sum": 0.0,
        "raw_plus_loss_sum": 0.0,
        "ce_loss_sum": 0.0,
        "projector_recon_loss_sum": 0.0,
        "symmetry_loss_sum": 0.0,
        "commit_loss_sum": 0.0,
        "heatmap_correct": torch.zeros(21, 21, dtype=torch.float32),
        "heatmap_total": torch.zeros(21, 21, dtype=torch.float32),
        "confusion": torch.zeros(21, 21, dtype=torch.float32),
    }


def _finalize_eval(acc: dict) -> Dict[str, object]:
    denom = max(acc["total"], 1)
    heatmap = torch.full((21, 21), float("nan"))
    valid = acc["heatmap_total"] > 0
    heatmap[valid] = acc["heatmap_correct"][valid] / acc["heatmap_total"][valid]
    return {
        "loss": acc["loss_total_sum"] / denom,
        "plus_loss": acc["plus_loss_sum"] / denom,
        "raw_plus_loss": acc["raw_plus_loss_sum"] / denom,
        "target_vq_ce_loss": acc["ce_loss_sum"] / denom,
        "projector_recon_loss": acc["projector_recon_loss_sum"] / denom,
        "symmetry_loss": acc["symmetry_loss_sum"] / denom,
        "commit_loss": acc["commit_loss_sum"] / denom,
        "code_accuracy": acc["pred_code_correct"] / denom,
        "number_accuracy": acc["pred_number_correct"] / denom,
        "count": acc["total"],
        "heatmap": heatmap.numpy(),
        "heatmap_total": acc["heatmap_total"].numpy(),
        "confusion": acc["confusion"].numpy(),
    }


def _accumulate_addition_eval(
    acc: dict,
    *,
    addition_model,
    v3_wrapper,
    content_vq: torch.Tensor,
    target_indices: torch.Tensor,
    numbers: torch.Tensor,
    code_to_number: Dict[int, int],
    use_symmetry: bool,
    plus_loss_weight: float,
    raw_plus_loss_weight: float,
    symmetry_loss_weight: float,
    pred_commit_loss_weight: float,
    target_vq_ce_loss_weight: float,
    target_vq_ce_temperature: float,
    target_vq_ce_mode: str,
    target_vq_soft_temperature: float,
    target_vq_soft_top_k: int,
    projector_recon_loss_weight: float,
) -> None:
    a = content_vq[:, 0]
    b = content_vq[:, 1]
    c = content_vq[:, 2]
    pred_raw, pred_vq, pred_indices, pred_commit = addition_model.plus_content(v3_wrapper, a, b)
    plus_loss = F.mse_loss(pred_vq, c)
    raw_plus_loss = torch.zeros((), device=v3_wrapper.device)
    if raw_plus_loss_weight > 0 and pred_raw.shape == c.shape:
        raw_plus_loss = F.mse_loss(pred_raw, c)
    ce_loss = torch.zeros((), device=v3_wrapper.device)
    if target_vq_ce_loss_weight > 0:
        target_c_indices = target_indices[:, 2].reshape(-1)
        logits = addition_model.codebook_logits(
            v3_wrapper,
            pred_raw,
            temperature=target_vq_ce_temperature,
        )
        ce_loss = target_vq_cross_entropy(
            logits,
            c,
            target_c_indices,
            v3_wrapper,
            mode=target_vq_ce_mode,
            soft_temperature=target_vq_soft_temperature,
            soft_top_k=target_vq_soft_top_k,
        )
    projector_recon_loss = torch.zeros((), device=v3_wrapper.device)
    if projector_recon_loss_weight > 0:
        projector_recon_loss = addition_model.projector_recon_loss(content_vq.reshape(-1, content_vq.size(-1)))
    symmetry_loss = torch.zeros((), device=v3_wrapper.device)
    symmetry_commit = torch.zeros((), device=v3_wrapper.device)
    if use_symmetry:
        symmetry_loss, symmetry_commit = full_symmetry_loss(addition_model, v3_wrapper, a, b, c)
    commit_loss = pred_commit + symmetry_commit
    total_loss = (
        plus_loss_weight * plus_loss
        + raw_plus_loss_weight * raw_plus_loss
        + target_vq_ce_loss_weight * ce_loss
        + projector_recon_loss_weight * projector_recon_loss
        + symmetry_loss_weight * symmetry_loss
        + pred_commit_loss_weight * commit_loss
    )

    batch_size = content_vq.size(0)
    acc["total"] += batch_size
    acc["loss_total_sum"] += float(total_loss.item()) * batch_size
    acc["plus_loss_sum"] += float(plus_loss.item()) * batch_size
    acc["raw_plus_loss_sum"] += float(raw_plus_loss.item()) * batch_size
    acc["ce_loss_sum"] += float(ce_loss.item()) * batch_size
    acc["projector_recon_loss_sum"] += float(projector_recon_loss.item()) * batch_size
    acc["symmetry_loss_sum"] += float(symmetry_loss.item()) * batch_size
    acc["commit_loss_sum"] += float(commit_loss.item()) * batch_size

    target_c_indices = target_indices[:, 2].reshape_as(pred_indices)
    code_correct = (pred_indices == target_c_indices).detach().cpu()
    acc["pred_code_correct"] += int(code_correct.sum().item())

    pred_numbers = _indices_to_numbers(pred_indices, code_to_number).to(numbers.device)
    target_numbers = numbers[:, 2].reshape_as(pred_numbers)
    number_correct = pred_numbers == target_numbers
    acc["pred_number_correct"] += int(number_correct.sum().item())

    for i in range(batch_size):
        left = int(numbers[i, 0].item())
        right = int(numbers[i, 1].item())
        target = int(target_numbers[i].item())
        pred = int(pred_numbers[i].item())
        acc["heatmap_total"][left, right] += 1
        if bool(number_correct[i].item()):
            acc["heatmap_correct"][left, right] += 1
        if 0 <= pred <= 20:
            acc["confusion"][target, pred] += 1


@torch.no_grad()
def evaluate_loader(
    addition_model,
    v3_wrapper,
    loader,
    code_to_number: Dict[int, int],
    use_symmetry: bool,
    plus_loss_weight: float,
    raw_plus_loss_weight: float,
    symmetry_loss_weight: float,
    pred_commit_loss_weight: float,
    target_vq_ce_loss_weight: float = 0.0,
    target_vq_ce_temperature: float = 1.0,
    target_vq_ce_mode: str = "hard",
    target_vq_soft_temperature: float = 1.0,
    target_vq_soft_top_k: int = 0,
    projector_recon_loss_weight: float = 0.0,
) -> Dict[str, object]:
    addition_model.eval()
    acc = _empty_eval_accumulators()

    for batch in loader:
        images = batch["images"].to(v3_wrapper.device, non_blocking=True)
        numbers = batch["numbers"].to(v3_wrapper.device, non_blocking=True)
        _, _, content_vq, target_indices, _ = v3_wrapper.encode_and_quantize(images)
        _accumulate_addition_eval(
            acc,
            addition_model=addition_model,
            v3_wrapper=v3_wrapper,
            content_vq=content_vq,
            target_indices=target_indices,
            numbers=numbers,
            code_to_number=code_to_number,
            use_symmetry=use_symmetry,
            plus_loss_weight=plus_loss_weight,
            raw_plus_loss_weight=raw_plus_loss_weight,
            symmetry_loss_weight=symmetry_loss_weight,
            pred_commit_loss_weight=pred_commit_loss_weight,
            target_vq_ce_loss_weight=target_vq_ce_loss_weight,
            target_vq_ce_temperature=target_vq_ce_temperature,
            target_vq_ce_mode=target_vq_ce_mode,
            target_vq_soft_temperature=target_vq_soft_temperature,
            target_vq_soft_top_k=target_vq_soft_top_k,
            projector_recon_loss_weight=projector_recon_loss_weight,
        )

    return _finalize_eval(acc)


@torch.no_grad()
def evaluate_fixed_codes(
    addition_model,
    v3_wrapper,
    fixed_codes: dict,
    code_to_number: Dict[int, int],
    use_symmetry: bool,
    plus_loss_weight: float,
    raw_plus_loss_weight: float,
    symmetry_loss_weight: float,
    pred_commit_loss_weight: float,
    target_vq_ce_loss_weight: float = 0.0,
    target_vq_ce_temperature: float = 1.0,
    target_vq_ce_mode: str = "hard",
    target_vq_soft_temperature: float = 1.0,
    target_vq_soft_top_k: int = 0,
    projector_recon_loss_weight: float = 0.0,
    batch_size: int = 128,
) -> Dict[str, object]:
    addition_model.eval()
    acc = _empty_eval_accumulators()
    total = int(fixed_codes["content_vq"].size(0))
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        content_vq = fixed_codes["content_vq"][start:stop].to(v3_wrapper.device, non_blocking=True)
        target_indices = fixed_codes["target_indices"][start:stop].to(v3_wrapper.device, non_blocking=True)
        numbers = fixed_codes["numbers"][start:stop].to(v3_wrapper.device, non_blocking=True)
        _accumulate_addition_eval(
            acc,
            addition_model=addition_model,
            v3_wrapper=v3_wrapper,
            content_vq=content_vq,
            target_indices=target_indices,
            numbers=numbers,
            code_to_number=code_to_number,
            use_symmetry=use_symmetry,
            plus_loss_weight=plus_loss_weight,
            raw_plus_loss_weight=raw_plus_loss_weight,
            symmetry_loss_weight=symmetry_loss_weight,
            pred_commit_loss_weight=pred_commit_loss_weight,
            target_vq_ce_loss_weight=target_vq_ce_loss_weight,
            target_vq_ce_temperature=target_vq_ce_temperature,
            target_vq_ce_mode=target_vq_ce_mode,
            target_vq_soft_temperature=target_vq_soft_temperature,
            target_vq_soft_top_k=target_vq_soft_top_k,
            projector_recon_loss_weight=projector_recon_loss_weight,
        )
    return _finalize_eval(acc)
