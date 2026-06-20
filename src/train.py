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
    enumerate_int_triplets,
    iter_batches_forever,
    make_number_letter_map,
    make_triplet_loader,
    save_split,
    split_triplets,
)
from .eval import estimate_codebook_mapping, evaluate_loader
from .plus_model import ContentPlusNet, full_symmetry_loss, plus_once
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


def make_state(step: int, plus_net, optimizer, config: dict, eval_metrics: dict) -> dict:
    return {
        "step": step,
        "plus_model": plus_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "eval_metrics": {
            key: value
            for key, value in eval_metrics.items()
            if key not in {"heatmap", "confusion"}
        },
    }


class TopKCheckpoints:
    def __init__(self, run_dir: Path, k: int):
        self.run_dir = run_dir
        self.k = int(k)
        self.records: List[tuple[float, int, dict]] = []

    def update(self, metric: float, step: int, state: dict) -> None:
        state_cpu = deepcopy(state)
        for tensor_state in [state_cpu["plus_model"]]:
            for key, value in tensor_state.items():
                if isinstance(value, torch.Tensor):
                    tensor_state[key] = value.detach().cpu()
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
    train_triplets, eval_triplets = split_triplets(
        triplets=triplets,
        train_ratio=float(config["train_ratio"]),
        seed=int(config["split_seed"]),
    )
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

    plus_net = ContentPlusNet(
        content_dim=int(config["content_dim"]),
        hidden_dim=int(config["plus_hidden_dim"]),
        hidden_layers=int(config["plus_hidden_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        plus_net.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    topk = TopKCheckpoints(run_dir=run_dir, k=int(config["save_top_k"]))
    train_iter = iter_batches_forever(train_loader)

    use_symmetry = bool(config["use_symmetry"])
    plus_loss_weight = float(config["plus_loss_weight"])
    symmetry_loss_weight = float(config["symmetry_loss_weight"])
    pred_commit_loss_weight = float(config["pred_commit_loss_weight"])

    for step in range(1, int(config["max_steps"]) + 1):
        plus_net.train()
        batch = next(train_iter)
        images = batch["images"].to(device, non_blocking=True)
        with torch.no_grad():
            _, _, content_vq, _, _ = v3.encode_and_quantize(images)
        a = content_vq[:, 0]
        b = content_vq[:, 1]
        c = content_vq[:, 2]
        _, pred_vq, _, pred_commit = plus_once(plus_net, v3, a, b)
        plus_loss = F.mse_loss(pred_vq, c)
        symmetry_loss = torch.zeros((), device=device)
        symmetry_commit = torch.zeros((), device=device)
        if use_symmetry:
            symmetry_loss, symmetry_commit = full_symmetry_loss(plus_net, v3, a, b, c)
        commit_loss = pred_commit + symmetry_commit
        loss = (
            plus_loss_weight * plus_loss
            + symmetry_loss_weight * symmetry_loss
            + pred_commit_loss_weight * commit_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(config["eval_interval"]) == 0 or step == int(config["max_steps"]):
            train_eval = evaluate_loader(
                plus_net=plus_net,
                v3_wrapper=v3,
                loader=train_eval_loader,
                code_to_number=code_to_number,
                use_symmetry=use_symmetry,
                plus_loss_weight=plus_loss_weight,
                symmetry_loss_weight=symmetry_loss_weight,
                pred_commit_loss_weight=pred_commit_loss_weight,
            )
            eval_eval = evaluate_loader(
                plus_net=plus_net,
                v3_wrapper=v3,
                loader=eval_loader,
                code_to_number=code_to_number,
                use_symmetry=use_symmetry,
                plus_loss_weight=plus_loss_weight,
                symmetry_loss_weight=symmetry_loss_weight,
                pred_commit_loss_weight=pred_commit_loss_weight,
            )
            row = {"step": step}
            row.update(flatten_eval("train", train_eval))
            row.update(flatten_eval("eval", eval_eval))
            append_metrics(metrics_csv, metrics_jsonl, row)
            state = make_state(step, plus_net, optimizer, config, eval_eval)
            torch.save(state, run_dir / "latest.pt")
            topk.update(float(eval_eval["loss"]), step, state)
            write_visuals(
                run_dir=run_dir,
                metrics_csv=metrics_csv,
                train_eval=train_eval,
                eval_eval=eval_eval,
                plus_net=plus_net,
                v3_wrapper=v3,
                eval_loader=eval_loader,
                step=step,
                num_examples=int(config["vis_num_examples"]),
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
    args = parser.parse_args()

    config = load_yaml(args.config)
    config = apply_cli_overrides(config, args)
    run_dir = train(config)
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()

