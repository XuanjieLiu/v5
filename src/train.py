from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .data import (
    UppercasePageIndex,
    enumerate_int_triplets,
    iter_batches_forever,
    make_number_letter_map,
    make_triplet_loader,
    save_split,
    split_triplets,
)
from .eval import estimate_codebook_mapping, evaluate_fixed_codes, evaluate_loader, target_vq_cross_entropy
from .plus_model import (
    AdditivePlusNet,
    AdditionModel,
    ContentPlusNet,
    IdentityProjector,
    LearnedProjector,
    PCAProjector,
    full_symmetry_loss,
)
from .v3_wrapper import FrozenV3, dump_yaml, load_yaml, save_json
from .visualize import write_visuals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_interval": args.eval_interval,
        "device": args.device,
        "run_name": args.run_name,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "plus_hidden_dim": args.plus_hidden_dim,
        "plus_hidden_layers": args.plus_hidden_layers,
        "train_input_mode": args.train_input_mode,
        "train_ratio": args.train_ratio,
        "triplet_split_mode": args.triplet_split_mode,
        "plus_loss_weight": args.plus_loss_weight,
        "raw_plus_loss_weight": args.raw_plus_loss_weight,
        "pred_commit_loss_weight": args.pred_commit_loss_weight,
        "save_top_k": args.save_top_k,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_optimizer": args.resume_optimizer,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(config_device: str) -> torch.device:
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config_device == "cuda":
        print("CUDA is not available; using CPU for smoke/debug run.")
    return torch.device("cpu")


def code_mapping_to_number(mapping: Dict[str, object], number_to_letter: Dict[int, str]) -> Dict[int, int]:
    letter_to_number = {letter: number for number, letter in number_to_letter.items()}
    code_to_number: Dict[int, int] = {}
    for code, letter in mapping["code_to_letter"].items():
        if letter in letter_to_number:
            code_to_number[int(code)] = int(letter_to_number[letter])
    return code_to_number


def _move_to_cpu(payload):
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {key: _move_to_cpu(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_move_to_cpu(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_move_to_cpu(value) for value in payload)
    return payload


@torch.no_grad()
def fit_pca_projector(
    v3: FrozenV3,
    letter_dir: Path,
    fragment_len: int,
    max_pages: int,
    batch_size: int,
    op_dim: int,
) -> PCAProjector:
    page_index = UppercasePageIndex(letter_dir, fragment_len=fragment_len)
    fragments, _ = page_index.load_all_fragments(max_pages=max_pages)
    content_chunks = []
    for start in range(0, fragments.size(0), batch_size):
        batch = fragments[start : start + batch_size].unsqueeze(1).to(v3.device)
        _, _, content_vq, _, _ = v3.encode_and_quantize(batch)
        content_chunks.append(content_vq.reshape(-1, content_vq.size(-1)).detach())
    content = torch.cat(content_chunks, dim=0).float().cpu()
    mean = content.mean(dim=0)
    centered = content - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:op_dim].contiguous()
    return PCAProjector(mean=mean, components=components).to(v3.device)


def build_projector(config: dict, v3: FrozenV3):
    operation_space = str(config.get("operation_space", "identity"))
    content_dim = int(config["content_dim"])
    op_dim = int(config.get("operation_dim", content_dim))
    if operation_space == "identity":
        return IdentityProjector(content_dim=content_dim).to(v3.device)
    if operation_space == "pca":
        if not 0 < op_dim <= content_dim:
            raise ValueError(f"operation_dim={op_dim} must be in [1, {content_dim}] for PCA.")
        return fit_pca_projector(
            v3=v3,
            letter_dir=resolve_path(config["letter_train_dir"]),
            fragment_len=int(config["fragment_len"]),
            max_pages=int(config.get("projector_fit_max_pages", config["mapping_max_pages"])),
            batch_size=int(config.get("eval_batch_size", config["batch_size"])),
            op_dim=op_dim,
        )
    if operation_space == "learned":
        if not 0 < op_dim:
            raise ValueError("operation_dim must be positive for learned projector.")
        return LearnedProjector(
            content_dim=content_dim,
            op_dim=op_dim,
            hidden_dim=int(config.get("projector_hidden_dim", 1024)),
            hidden_layers=int(config.get("projector_hidden_layers", 1)),
        ).to(v3.device)
    raise ValueError(f"Unknown operation_space={operation_space!r}. Use identity, pca, or learned.")


@torch.no_grad()
def encode_fixed_codes(v3: FrozenV3, loader) -> dict:
    content_chunks = []
    index_chunks = []
    number_chunks = []
    for batch in loader:
        images = batch["images"].to(v3.device, non_blocking=True)
        _, _, content_vq, indices, _ = v3.encode_and_quantize(images)
        content_chunks.append(content_vq.detach().cpu())
        index_chunks.append(indices.detach().cpu())
        number_chunks.append(batch["numbers"].detach().cpu())
    return {
        "content_vq": torch.cat(content_chunks, dim=0),
        "target_indices": torch.cat(index_chunks, dim=0),
        "numbers": torch.cat(number_chunks, dim=0),
    }


def sample_fixed_code_batch(fixed_codes: dict, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    total = int(fixed_codes["content_vq"].size(0))
    sample_size = min(int(batch_size), total)
    indices = torch.randint(total, (sample_size,))
    content_vq = fixed_codes["content_vq"][indices].to(device, non_blocking=True)
    target_indices = fixed_codes["target_indices"][indices].to(device, non_blocking=True)
    return content_vq, target_indices


def make_state(step: int, addition_model, optimizer, config: dict, eval_metrics: dict) -> dict:
    return {
        "step": step,
        "addition_model": addition_model.state_dict(),
        "plus_model": addition_model.plus_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "eval_metrics": {
            key: value
            for key, value in eval_metrics.items()
            if key not in {"heatmap", "heatmap_total", "confusion"}
        },
    }


def load_training_checkpoint(
    checkpoint_path: str | Path,
    addition_model,
    optimizer,
    *,
    load_optimizer: bool,
    device: torch.device,
) -> int:
    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device)
    addition_model.load_state_dict(checkpoint["addition_model"])
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


class TopKCheckpoints:
    def __init__(self, run_dir: Path, k: int):
        self.run_dir = run_dir
        self.k = int(k)
        self.records: List[tuple[float, int, dict]] = []
        self._load_existing()
        self._remove_stale_files()

    def _load_existing(self) -> None:
        if self.k <= 0:
            return
        for path in sorted(self.run_dir.glob("best_*.pt")):
            try:
                rank = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if rank > self.k:
                continue
            state = torch.load(path, map_location="cpu")
            eval_metrics = state.get("eval_metrics", {})
            if "loss" not in eval_metrics:
                continue
            self.records.append((float(eval_metrics["loss"]), int(state.get("step", 0)), state))
        self.records.sort(key=lambda item: (item[0], item[1]))
        self.records = self.records[: self.k]

    def _remove_stale_files(self) -> None:
        for path in self.run_dir.glob("best_*.pt"):
            try:
                rank = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if rank > self.k:
                path.unlink(missing_ok=True)

    def update(self, metric: float, step: int, state: dict) -> None:
        state_cpu = _move_to_cpu(deepcopy(state))
        self.records.append((float(metric), int(step), state_cpu))
        self.records.sort(key=lambda item: (item[0], item[1]))
        self.records = self.records[: self.k]
        for idx, (_, _, record_state) in enumerate(self.records, start=1):
            torch.save(record_state, self.run_dir / f"best_{idx}.pt")


def append_metrics(metrics_csv: Path, metrics_jsonl: Path, row: dict) -> None:
    write_header = not metrics_csv.exists()
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    with metrics_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def flatten_eval(prefix: str, payload: dict) -> dict:
    return {
        f"{prefix}_loss": payload["loss"],
        f"{prefix}_plus_loss": payload["plus_loss"],
        f"{prefix}_raw_plus_loss": payload["raw_plus_loss"],
        f"{prefix}_target_vq_ce_loss": payload["target_vq_ce_loss"],
        f"{prefix}_projector_recon_loss": payload["projector_recon_loss"],
        f"{prefix}_symmetry_loss": payload["symmetry_loss"],
        f"{prefix}_commit_loss": payload["commit_loss"],
        f"{prefix}_code_accuracy": payload["code_accuracy"],
        f"{prefix}_number_accuracy": payload["number_accuracy"],
        f"{prefix}_count": payload["count"],
    }


def train(config: dict) -> Path:
    setup_seed(int(config["seed"]))
    device = choose_device(str(config.get("device", "cuda")))
    run_dir = resolve_path(config["output_root"]) / str(config["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"

    config = dict(config)
    config["resolved_run_dir"] = str(run_dir)
    dump_yaml(config, run_dir / "config.yaml")

    number_to_letter = make_number_letter_map(int(config["number_min"]), int(config["number_max"]))
    save_json({str(k): v for k, v in number_to_letter.items()}, run_dir / "number_letter_map.json")

    triplets = enumerate_int_triplets(int(config["number_min"]), int(config["number_max"]))
    triplet_split_mode = str(config.get("triplet_split_mode", "random"))
    if triplet_split_mode == "random":
        train_triplets, eval_triplets = split_triplets(
            triplets=triplets,
            train_ratio=float(config["train_ratio"]),
            seed=int(config["split_seed"]),
        )
    elif triplet_split_mode == "all":
        train_triplets = list(triplets)
        eval_triplets = list(triplets)
    else:
        raise ValueError("triplet_split_mode must be 'random' or 'all'.")
    save_split(run_dir / "split.json", train_triplets, eval_triplets)

    train_loader = make_triplet_loader(
        triplets=train_triplets,
        letter_dir=resolve_path(config["letter_train_dir"]),
        number_to_letter=number_to_letter,
        batch_size=int(config["batch_size"]),
        fragment_len=int(config["fragment_len"]),
        deterministic=False,
        sample_seed=int(config["seed"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
    )
    train_eval_loader = make_triplet_loader(
        triplets=train_triplets,
        letter_dir=resolve_path(config["letter_train_dir"]),
        number_to_letter=number_to_letter,
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        fragment_len=int(config["fragment_len"]),
        deterministic=True,
        sample_seed=int(config["seed"]) + 10000,
        shuffle=False,
        num_workers=0,
    )
    eval_loader = make_triplet_loader(
        triplets=eval_triplets,
        letter_dir=resolve_path(config["letter_eval_dir"]),
        number_to_letter=number_to_letter,
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        fragment_len=int(config["fragment_len"]),
        deterministic=True,
        sample_seed=int(config["seed"]) + 20000,
        shuffle=False,
        num_workers=0,
    )

    v3 = FrozenV3(
        v3_root=resolve_path(config["v3_root"]),
        v3_config_path=resolve_path(config["v3_config"]),
        v3_checkpoint_path=resolve_path(config["v3_checkpoint"]),
        device=device,
    )
    if int(config["content_dim"]) != v3.content_dim:
        raise ValueError(f"content_dim={config['content_dim']} does not match V3 d_emb_c={v3.content_dim}.")

    mapping = estimate_codebook_mapping(
        v3_wrapper=v3,
        letter_dir=resolve_path(config["letter_train_dir"]),
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        fragment_len=int(config["fragment_len"]),
        max_pages=int(config["mapping_max_pages"]),
    )
    save_json(mapping, run_dir / "v3_codebook_mapping.json")
    code_to_number = code_mapping_to_number(mapping, number_to_letter)

    projector = build_projector(config, v3)
    config["resolved_operation_space"] = str(config.get("operation_space", "identity"))
    config["resolved_operation_dim"] = int(getattr(projector, "op_dim", config["content_dim"]))
    prediction_mode = str(config.get("prediction_mode", "content"))
    if prediction_mode not in {"content", "code_logits"}:
        raise ValueError("prediction_mode must be 'content' or 'code_logits'.")
    config["resolved_prediction_mode"] = prediction_mode
    plus_operator = str(config.get("plus_operator", "mlp"))
    if plus_operator not in {"mlp", "additive"}:
        raise ValueError("plus_operator must be 'mlp' or 'additive'.")
    if plus_operator == "additive" and prediction_mode != "content":
        raise ValueError("plus_operator='additive' currently requires prediction_mode='content'.")
    config["resolved_plus_operator"] = plus_operator
    n_vq_codes = int(v3.model.vq.codebook.size(0))
    plus_output_dim = n_vq_codes if prediction_mode == "code_logits" else int(config["resolved_operation_dim"])
    dump_yaml(config, run_dir / "config.yaml")
    if plus_operator == "additive":
        plus_net = AdditivePlusNet(
            content_dim=int(config["resolved_operation_dim"]),
            learn_scale=bool(config.get("additive_learn_scale", True)),
            learn_bias=bool(config.get("additive_learn_bias", True)),
        )
    else:
        plus_net = ContentPlusNet(
            content_dim=int(config["resolved_operation_dim"]),
            hidden_dim=int(config["plus_hidden_dim"]),
            hidden_layers=int(config["plus_hidden_layers"]),
            output_dim=plus_output_dim,
        )
    addition_model = AdditionModel(
        projector=projector,
        plus_net=plus_net,
        prediction_mode=prediction_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        addition_model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_step = 0
    if config.get("resume_checkpoint"):
        start_step = load_training_checkpoint(
            config["resume_checkpoint"],
            addition_model,
            optimizer,
            load_optimizer=bool(config.get("resume_optimizer", False)),
            device=device,
        )
        set_optimizer_learning_rate(optimizer, float(config["learning_rate"]))
        config["resolved_resume_step"] = start_step
        dump_yaml(config, run_dir / "config.yaml")
        if start_step >= int(config["max_steps"]):
            raise ValueError(f"resume step {start_step} must be below max_steps={config['max_steps']}.")
    topk = TopKCheckpoints(run_dir=run_dir, k=int(config["save_top_k"]))
    train_iter = iter_batches_forever(train_loader)
    train_input_mode = str(config.get("train_input_mode", "image"))
    if train_input_mode not in {"image", "fixed_v3_codes"}:
        raise ValueError("train_input_mode must be 'image' or 'fixed_v3_codes'.")
    fixed_train_codes = None
    fixed_eval_codes = None
    if train_input_mode == "fixed_v3_codes":
        fixed_train_codes = encode_fixed_codes(v3, train_eval_loader)
        fixed_eval_codes = encode_fixed_codes(v3, eval_loader)

    use_symmetry = bool(config["use_symmetry"])
    plus_loss_weight = float(config["plus_loss_weight"])
    raw_plus_loss_weight = float(config.get("raw_plus_loss_weight", 0.0))
    symmetry_loss_weight = float(config["symmetry_loss_weight"])
    pred_commit_loss_weight = float(config["pred_commit_loss_weight"])
    target_vq_ce_loss_weight = float(config.get("target_vq_ce_loss_weight", 0.0))
    target_vq_ce_temperature = float(config.get("target_vq_ce_temperature", 1.0))
    target_vq_ce_mode = str(config.get("target_vq_ce_mode", "hard"))
    if target_vq_ce_mode not in {"hard", "soft"}:
        raise ValueError("target_vq_ce_mode must be 'hard' or 'soft'.")
    target_vq_soft_temperature = float(config.get("target_vq_soft_temperature", 1.0))
    target_vq_soft_top_k = int(config.get("target_vq_soft_top_k", 0))
    projector_recon_loss_weight = float(config.get("projector_recon_loss_weight", 0.0))

    for step in range(start_step + 1, int(config["max_steps"]) + 1):
        addition_model.train()
        if train_input_mode == "fixed_v3_codes":
            content_vq, target_indices = sample_fixed_code_batch(
                fixed_train_codes,
                batch_size=int(config["batch_size"]),
                device=device,
            )
        else:
            batch = next(train_iter)
            images = batch["images"].to(device, non_blocking=True)
            with torch.no_grad():
                _, _, content_vq, target_indices, _ = v3.encode_and_quantize(images)
        a = content_vq[:, 0]
        b = content_vq[:, 1]
        c = content_vq[:, 2]
        pred_raw, pred_vq, _, pred_commit = addition_model.plus_content(v3, a, b)
        plus_loss = F.mse_loss(pred_vq, c)
        raw_plus_loss = torch.zeros((), device=device)
        if raw_plus_loss_weight > 0 and pred_raw.shape == c.shape:
            raw_plus_loss = F.mse_loss(pred_raw, c)
        target_vq_ce_loss = torch.zeros((), device=device)
        if target_vq_ce_loss_weight > 0:
            logits = addition_model.codebook_logits(
                v3,
                pred_raw,
                temperature=target_vq_ce_temperature,
            )
            target_c_indices = target_indices[:, 2].reshape(-1)
            target_vq_ce_loss = target_vq_cross_entropy(
                logits,
                c,
                target_c_indices,
                v3,
                mode=target_vq_ce_mode,
                soft_temperature=target_vq_soft_temperature,
                soft_top_k=target_vq_soft_top_k,
            )
        projector_recon_loss = torch.zeros((), device=device)
        if projector_recon_loss_weight > 0:
            projector_recon_loss = addition_model.projector_recon_loss(content_vq.reshape(-1, content_vq.size(-1)))
        symmetry_loss = torch.zeros((), device=device)
        symmetry_commit = torch.zeros((), device=device)
        if use_symmetry:
            symmetry_loss, symmetry_commit = full_symmetry_loss(addition_model, v3, a, b, c)
        commit_loss = pred_commit + symmetry_commit
        loss = (
            plus_loss_weight * plus_loss
            + raw_plus_loss_weight * raw_plus_loss
            + target_vq_ce_loss_weight * target_vq_ce_loss
            + projector_recon_loss_weight * projector_recon_loss
            + symmetry_loss_weight * symmetry_loss
            + pred_commit_loss_weight * commit_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == start_step + 1 or step % int(config["eval_interval"]) == 0 or step == int(config["max_steps"]):
            eval_kwargs = {
                "addition_model": addition_model,
                "v3_wrapper": v3,
                "code_to_number": code_to_number,
                "use_symmetry": use_symmetry,
                "plus_loss_weight": plus_loss_weight,
                "raw_plus_loss_weight": raw_plus_loss_weight,
                "symmetry_loss_weight": symmetry_loss_weight,
                "pred_commit_loss_weight": pred_commit_loss_weight,
                "target_vq_ce_loss_weight": target_vq_ce_loss_weight,
                "target_vq_ce_temperature": target_vq_ce_temperature,
                "target_vq_ce_mode": target_vq_ce_mode,
                "target_vq_soft_temperature": target_vq_soft_temperature,
                "target_vq_soft_top_k": target_vq_soft_top_k,
                "projector_recon_loss_weight": projector_recon_loss_weight,
            }
            if train_input_mode == "fixed_v3_codes":
                train_eval = evaluate_fixed_codes(
                    fixed_codes=fixed_train_codes,
                    batch_size=int(config.get("eval_batch_size", config["batch_size"])),
                    **eval_kwargs,
                )
                eval_eval = evaluate_fixed_codes(
                    fixed_codes=fixed_eval_codes,
                    batch_size=int(config.get("eval_batch_size", config["batch_size"])),
                    **eval_kwargs,
                )
            else:
                train_eval = evaluate_loader(loader=train_eval_loader, **eval_kwargs)
                eval_eval = evaluate_loader(loader=eval_loader, **eval_kwargs)
            row = {"step": step}
            row.update(flatten_eval("train", train_eval))
            row.update(flatten_eval("eval", eval_eval))
            append_metrics(metrics_csv, metrics_jsonl, row)
            state = make_state(step, addition_model, optimizer, config, eval_eval)
            torch.save(state, run_dir / "latest.pt")
            topk.update(float(eval_eval["loss"]), step, state)
            write_visuals(
                run_dir=run_dir,
                metrics_csv=metrics_csv,
                train_eval=train_eval,
                eval_eval=eval_eval,
                addition_model=addition_model,
                v3_wrapper=v3,
                eval_loader=eval_loader,
                step=step,
                num_examples=int(config["vis_num_examples"]),
                sample_seed=int(config.get("vis_sample_seed", config["seed"])),
            )
            print(
                f"step={step} loss={eval_eval['loss']:.6f} "
                f"eval_num_acc={eval_eval['number_accuracy']:.4f} "
                f"eval_code_acc={eval_eval['code_accuracy']:.4f}"
            )

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--plus_hidden_dim", type=int, default=None)
    parser.add_argument("--plus_hidden_layers", type=int, default=None)
    parser.add_argument("--train_input_mode", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--triplet_split_mode", type=str, default=None)
    parser.add_argument("--plus_loss_weight", type=float, default=None)
    parser.add_argument("--raw_plus_loss_weight", type=float, default=None)
    parser.add_argument("--pred_commit_loss_weight", type=float, default=None)
    parser.add_argument("--save_top_k", type=int, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume_optimizer", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_cli_overrides(config, args)
    run_dir = train(config)
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
