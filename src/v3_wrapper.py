from __future__ import annotations

import contextlib
import ast
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        return _load_simple_yaml(path)
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(payload: dict, path: str | Path) -> None:
    try:
        import yaml
    except ImportError:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str):
    value = value.strip()
    if value == "" or value in {"None", "null", "Null", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_simple_yaml(path: str | Path) -> dict:
    """Small fallback for the simple key/value YAML configs used by V3/V5."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


@contextlib.contextmanager
def prepend_sys_path(path: str | Path):
    path_str = str(Path(path).resolve())
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


class FrozenV3:
    def __init__(
        self,
        v3_root: str | Path,
        v3_config_path: str | Path,
        v3_checkpoint_path: str | Path,
        device: torch.device,
    ):
        self.v3_root = Path(v3_root).resolve()
        self.v3_config_path = Path(v3_config_path).resolve()
        self.v3_checkpoint_path = Path(v3_checkpoint_path).resolve()
        self.device = device
        self.config = load_yaml(self.v3_config_path)
        self.model = self._build_and_load_model()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _build_and_load_model(self):
        with prepend_sys_path(self.v3_root):
            from model.factory import get_model

            model = get_model(self.config["dataloader"], self.config["model_config"]).to(self.device)
        checkpoint = torch.load(self.v3_checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model

    @property
    def content_dim(self) -> int:
        return int(self.config["model_config"]["d_emb_c"])

    @torch.no_grad()
    def encode_triplet(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        emb_c, emb_s = self.model.encode(images)
        return emb_c, emb_s

    def quantize_content(self, content: torch.Tensor):
        return self.model.quantize(content, freeze_codebook=True)

    @torch.no_grad()
    def decode_for_vis(self, content_vq: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.decode(content_vq, style)

    @torch.no_grad()
    def encode_and_quantize(self, images: torch.Tensor):
        emb_c, emb_s = self.encode_triplet(images)
        emb_c_vq, indices, commit_loss = self.quantize_content(emb_c)
        return emb_c, emb_s, emb_c_vq, indices, commit_loss


def save_json(payload: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
