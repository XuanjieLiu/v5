from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable

import torch

from .data import (
    enumerate_int_triplets,
    make_number_letter_map,
    make_triplet_loader,
    split_triplets,
)
from .eval import _indices_to_numbers, estimate_codebook_mapping
from .train import code_mapping_to_number, resolve_path
from .v3_wrapper import FrozenV3, load_yaml, save_json


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "cuda":
        print("CUDA is not available; using CPU.")
    return torch.device("cpu")


def nested_counter_to_dict(counter: Dict[int, Counter]) -> dict:
    return {
        str(key): {str(inner_key): int(value) for inner_key, value in counts.items()}
        for key, counts in sorted(counter.items())
    }


def summarize_mapping(mapping: dict, number_min: int, number_max: int) -> dict:
    label_to_code_counts: Dict[int, Counter] = defaultdict(Counter)
    code_to_label_counts: Dict[int, Counter] = defaultdict(Counter)
    for code_text, counts in mapping["counts"].items():
        code = int(code_text)
        for label_text, count in counts.items():
            label = int(label_text)
            label_to_code_counts[label][code] += int(count)
            code_to_label_counts[code][label] += int(count)

    labels = list(range(number_min, number_max + 1))
    label_rows = {}
    label_total = 0
    label_correct = 0
    for label in labels:
        counts = label_to_code_counts[label]
        total = sum(counts.values())
        if total == 0:
            label_rows[str(label)] = {
                "letter": chr(ord("A") + label),
                "total": 0,
                "majority_code": None,
                "majority_count": 0,
                "majority_purity": None,
                "top_codes": [],
            }
            continue
        majority_code, majority_count = counts.most_common(1)[0]
        label_total += total
        label_correct += majority_count
        label_rows[str(label)] = {
            "letter": chr(ord("A") + label),
            "total": int(total),
            "majority_code": int(majority_code),
            "majority_count": int(majority_count),
            "majority_purity": majority_count / total,
            "top_codes": [[int(code), int(count)] for code, count in counts.most_common(5)],
        }

    code_rows = {}
    code_total = 0
    code_correct = 0
    for code, counts in sorted(code_to_label_counts.items()):
        restricted_counts = Counter(
            {label: count for label, count in counts.items() if number_min <= label <= number_max}
        )
        total = sum(restricted_counts.values())
        if total == 0:
            continue
        majority_label, majority_count = restricted_counts.most_common(1)[0]
        code_total += total
        code_correct += majority_count
        code_rows[str(code)] = {
            "total": int(total),
            "majority_label": int(majority_label),
            "majority_letter": chr(ord("A") + majority_label),
            "majority_count": int(majority_count),
            "majority_purity": majority_count / total,
            "top_labels": [[int(label), int(count)] for label, count in restricted_counts.most_common(5)],
        }

    return {
        "label_majority_micro_purity": label_correct / max(label_total, 1),
        "code_majority_micro_purity": code_correct / max(code_total, 1),
        "observed_code_count": len(code_to_label_counts),
        "label_to_code": label_rows,
        "code_to_label": code_rows,
        "raw_label_to_code_counts": nested_counter_to_dict(label_to_code_counts),
        "raw_code_to_label_counts": nested_counter_to_dict(code_to_label_counts),
    }


def make_majority_code_by_label(mapping_summary: dict) -> Dict[int, int]:
    result = {}
    for label_text, row in mapping_summary["label_to_code"].items():
        if row["majority_code"] is not None:
            result[int(label_text)] = int(row["majority_code"])
    return result


@torch.no_grad()
def evaluate_majority_oracle(
    *,
    name: str,
    loader,
    v3: FrozenV3,
    majority_code_by_label: Dict[int, int],
    code_to_number: Dict[int, int],
) -> dict:
    total = 0
    oracle_code_correct = 0
    oracle_number_correct = 0
    target_code_number_correct = 0
    target_code_counts = Counter()
    target_number_counts = Counter()
    oracle_pred_code_counts = Counter()
    oracle_confusion = torch.zeros(21, 21, dtype=torch.float32)

    for batch in loader:
        images = batch["images"].to(v3.device, non_blocking=True)
        numbers = batch["numbers"].to(v3.device, non_blocking=True)
        _, _, _, target_indices, _ = v3.encode_and_quantize(images)

        target_codes = target_indices[:, 2].reshape(-1).detach().cpu()
        target_numbers = numbers[:, 2].reshape(-1).detach().cpu()
        oracle_codes = torch.tensor(
            [majority_code_by_label.get(int(number), -1) for number in target_numbers.tolist()],
            dtype=torch.long,
        )

        oracle_code_correct += int((oracle_codes == target_codes).sum().item())
        oracle_numbers = _indices_to_numbers(oracle_codes, code_to_number).reshape(-1)
        target_code_numbers = _indices_to_numbers(target_codes, code_to_number).reshape(-1)
        oracle_number_correct += int((oracle_numbers == target_numbers).sum().item())
        target_code_number_correct += int((target_code_numbers == target_numbers).sum().item())
        total += int(target_codes.numel())

        target_code_counts.update(int(code) for code in target_codes.tolist())
        target_number_counts.update(int(number) for number in target_numbers.tolist())
        oracle_pred_code_counts.update(int(code) for code in oracle_codes.tolist())
        for target, pred in zip(target_numbers.tolist(), oracle_numbers.tolist()):
            if 0 <= int(target) < oracle_confusion.size(0) and 0 <= int(pred) < oracle_confusion.size(1):
                oracle_confusion[int(target), int(pred)] += 1

    denom = max(total, 1)
    return {
        "name": name,
        "count": total,
        "oracle_exact_code_accuracy": oracle_code_correct / denom,
        "oracle_number_accuracy": oracle_number_correct / denom,
        "target_code_number_accuracy": target_code_number_correct / denom,
        "target_code_counts": {str(k): int(v) for k, v in sorted(target_code_counts.items())},
        "target_number_counts": {str(k): int(v) for k, v in sorted(target_number_counts.items())},
        "oracle_pred_code_counts": {str(k): int(v) for k, v in sorted(oracle_pred_code_counts.items())},
        "oracle_number_confusion": oracle_confusion.numpy().tolist(),
    }


def print_summary(payload: dict) -> None:
    mapping = payload["mapping_summary"]
    print(f"Observed V3 codes: {mapping['observed_code_count']}")
    print(f"Label->majority-code micro purity: {mapping['label_majority_micro_purity']:.4f}")
    print(f"Code->majority-label micro purity: {mapping['code_majority_micro_purity']:.4f}")
    for split in payload["triplet_oracles"]:
        print(
            f"{split['name']}: oracle_exact_code_acc={split['oracle_exact_code_accuracy']:.4f} "
            f"oracle_number_acc={split['oracle_number_accuracy']:.4f} "
            f"target_code_number_acc={split['target_code_number_accuracy']:.4f} "
            f"count={split['count']}"
        )


def build_loader(config: dict, triplets: Iterable[tuple[int, int, int]], letter_dir: str, seed_offset: int):
    number_to_letter = make_number_letter_map(int(config["number_min"]), int(config["number_max"]))
    return make_triplet_loader(
        triplets=list(triplets),
        letter_dir=resolve_path(letter_dir),
        number_to_letter=number_to_letter,
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        fragment_len=int(config["fragment_len"]),
        deterministic=True,
        sample_seed=int(config["seed"]) + seed_offset,
        shuffle=False,
        num_workers=0,
    )


def analyze(config: dict, device: torch.device) -> dict:
    number_min = int(config["number_min"])
    number_max = int(config["number_max"])
    number_to_letter = make_number_letter_map(number_min, number_max)
    triplets = enumerate_int_triplets(number_min, number_max)
    train_triplets, eval_triplets = split_triplets(
        triplets=triplets,
        train_ratio=float(config["train_ratio"]),
        seed=int(config["split_seed"]),
    )

    v3 = FrozenV3(
        v3_root=resolve_path(config["v3_root"]),
        v3_config_path=resolve_path(config["v3_config"]),
        v3_checkpoint_path=resolve_path(config["v3_checkpoint"]),
        device=device,
    )
    mapping = estimate_codebook_mapping(
        v3_wrapper=v3,
        letter_dir=resolve_path(config["letter_train_dir"]),
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        fragment_len=int(config["fragment_len"]),
        max_pages=int(config["mapping_max_pages"]),
    )
    mapping_summary = summarize_mapping(mapping, number_min, number_max)
    majority_code_by_label = make_majority_code_by_label(mapping_summary)
    code_to_number = code_mapping_to_number(mapping, number_to_letter)

    train_loader = build_loader(config, train_triplets, config["letter_train_dir"], 10000)
    eval_loader = build_loader(config, eval_triplets, config["letter_eval_dir"], 20000)
    triplet_oracles = [
        evaluate_majority_oracle(
            name="train_fixed",
            loader=train_loader,
            v3=v3,
            majority_code_by_label=majority_code_by_label,
            code_to_number=code_to_number,
        ),
        evaluate_majority_oracle(
            name="eval_fixed",
            loader=eval_loader,
            v3=v3,
            majority_code_by_label=majority_code_by_label,
            code_to_number=code_to_number,
        ),
    ]

    return {
        "config": {
            "v3_root": str(resolve_path(config["v3_root"])),
            "v3_config": str(resolve_path(config["v3_config"])),
            "v3_checkpoint": str(resolve_path(config["v3_checkpoint"])),
            "letter_train_dir": str(resolve_path(config["letter_train_dir"])),
            "letter_eval_dir": str(resolve_path(config["letter_eval_dir"])),
            "number_min": number_min,
            "number_max": number_max,
            "train_ratio": float(config["train_ratio"]),
            "split_seed": int(config["split_seed"]),
            "mapping_max_pages": int(config["mapping_max_pages"]),
        },
        "mapping": mapping,
        "mapping_summary": mapping_summary,
        "code_to_number": {str(k): int(v) for k, v in sorted(code_to_number.items())},
        "majority_code_by_label": {str(k): int(v) for k, v in sorted(majority_code_by_label.items())},
        "triplet_oracles": triplet_oracles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = choose_device(args.device or str(config.get("device", "cuda")))
    payload = analyze(config, device)
    print_summary(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(payload, output_path)
        print(f"Saved diagnostics to {output_path.resolve()}")


if __name__ == "__main__":
    main()

