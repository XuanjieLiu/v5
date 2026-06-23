from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


Triplet = Tuple[int, int, int]

STYLE_NAMES = [
    "black",
    "blue",
    "green",
    "red",
    "teal",
    "purple",
    "orange",
    "brown",
]


def make_number_letter_map(number_min: int = 0, number_max: int = 20) -> Dict[int, str]:
    if number_min != 0:
        raise ValueError("The V5 uppercase mapping currently assumes number_min == 0.")
    letters = list(string.ascii_uppercase)
    if number_max >= len(letters):
        raise ValueError(f"number_max={number_max} needs more than 26 uppercase letters.")
    return {number: letters[number] for number in range(number_min, number_max + 1)}


def enumerate_int_triplets(number_min: int = 0, number_max: int = 20) -> List[Triplet]:
    if number_min != 0:
        raise ValueError("Addition triplets currently assume number_min == 0.")
    triplets: List[Triplet] = []
    for a in range(number_min, number_max + 1):
        for b in range(number_min, number_max + 1):
            c = a + b
            if c <= number_max:
                triplets.append((a, b, c))
    return triplets


def split_triplets(
    triplets: Sequence[Triplet],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Triplet], List[Triplet]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    shuffled = list(triplets)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    train_triplets = sorted(shuffled[:train_size])
    eval_triplets = sorted(shuffled[train_size:])
    return train_triplets, eval_triplets


def save_split(path: Path, train_triplets: Sequence[Triplet], eval_triplets: Sequence[Triplet]) -> None:
    payload = {
        "train": [list(t) for t in train_triplets],
        "eval": [list(t) for t in eval_triplets],
        "train_count": len(train_triplets),
        "eval_count": len(eval_triplets),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_page_path(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Uppercase letter page name must be '<letters>_<style>.png': {path}")
    letters, style = stem.rsplit("_", 1)
    if len(letters) != 26:
        raise ValueError(f"Expected 26 letters in page name, got {len(letters)} for {path}")
    return letters, style


def _pil_to_v3_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


@dataclass(frozen=True)
class PageRecord:
    path: Path
    letters: str
    style: str


class UppercasePageIndex:
    def __init__(self, letter_dir: str | Path, fragment_len: int = 32):
        self.letter_dir = Path(letter_dir)
        self.fragment_len = fragment_len
        self.records = self._load_records()
        if not self.records:
            raise FileNotFoundError(f"No uppercase page PNG files found in {self.letter_dir}")

    def _load_records(self) -> List[PageRecord]:
        paths = sorted(self.letter_dir.glob("*.png"))
        records: List[PageRecord] = []
        for path in paths:
            letters, style = _parse_page_path(path)
            records.append(PageRecord(path=path, letters=letters, style=style))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def sample_fragment(self, letter: str, rng: random.Random) -> Tuple[torch.Tensor, str, Path]:
        record = self.records[rng.randrange(len(self.records))]
        try:
            letter_idx = record.letters.index(letter)
        except ValueError as exc:
            raise ValueError(f"Letter {letter} is missing from {record.path}") from exc
        x0 = letter_idx * self.fragment_len
        with Image.open(record.path) as img:
            fragment = img.crop((x0, 0, x0 + self.fragment_len, img.height))
            tensor = _pil_to_v3_tensor(fragment)
        return tensor, record.style, record.path

    def load_all_fragments(self, max_pages: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        records = self.records[:max_pages] if max_pages is not None else self.records
        all_fragments: List[torch.Tensor] = []
        all_labels: List[int] = []
        letter_to_idx = {letter: idx for idx, letter in enumerate(string.ascii_uppercase)}
        for record in records:
            with Image.open(record.path) as img:
                for pos, letter in enumerate(record.letters):
                    x0 = pos * self.fragment_len
                    fragment = img.crop((x0, 0, x0 + self.fragment_len, img.height))
                    all_fragments.append(_pil_to_v3_tensor(fragment))
                    all_labels.append(letter_to_idx[letter])
        return torch.stack(all_fragments, dim=0), torch.tensor(all_labels, dtype=torch.long)


class AdditionTripletDataset(Dataset):
    def __init__(
        self,
        triplets: Sequence[Triplet],
        letter_dir: str | Path,
        number_to_letter: Dict[int, str],
        fragment_len: int = 32,
        deterministic: bool = False,
        sample_seed: int = 0,
    ):
        self.triplets = list(triplets)
        self.page_index = UppercasePageIndex(letter_dir, fragment_len=fragment_len)
        self.number_to_letter = dict(number_to_letter)
        self.deterministic = deterministic
        self.sample_seed = sample_seed

    def __len__(self) -> int:
        return len(self.triplets)

    def _rng_for_index(self, idx: int) -> random.Random:
        if self.deterministic:
            return random.Random(self.sample_seed + idx * 1009)
        return random

    def __getitem__(self, idx: int):
        triplet = self.triplets[idx]
        rng = self._rng_for_index(idx)
        fragments: List[torch.Tensor] = []
        style_names: List[str] = []
        page_paths: List[str] = []
        for number in triplet:
            letter = self.number_to_letter[int(number)]
            fragment, style, path = self.page_index.sample_fragment(letter, rng)
            fragments.append(fragment)
            style_names.append(style)
            page_paths.append(str(path))
        images = torch.stack(fragments, dim=0)
        numbers = torch.tensor(triplet, dtype=torch.long)
        item = {
            "images": images,
            "numbers": numbers,
            "styles": style_names,
            "page_paths": page_paths,
        }
        return item


def make_triplet_loader(
    triplets: Sequence[Triplet],
    letter_dir: str | Path,
    number_to_letter: Dict[int, str],
    batch_size: int,
    fragment_len: int = 32,
    deterministic: bool = False,
    sample_seed: int = 0,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = AdditionTripletDataset(
        triplets=triplets,
        letter_dir=letter_dir,
        number_to_letter=number_to_letter,
        fragment_len=fragment_len,
        deterministic=deterministic,
        sample_seed=sample_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def iter_batches_forever(loader: DataLoader) -> Iterable[dict]:
    while True:
        for batch in loader:
            yield batch
