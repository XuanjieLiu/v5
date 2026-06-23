from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ContentPlusNet(nn.Module):
    def __init__(
        self,
        content_dim: int = 512,
        hidden_dim: int = 1024,
        hidden_layers: int = 2,
        output_dim: int | None = None,
    ):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative.")
        output_dim = content_dim if output_dim is None else int(output_dim)
        layers = []
        in_dim = content_dim * 2
        if hidden_layers == 0:
            layers.append(nn.Linear(in_dim, output_dim))
        else:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            for _ in range(hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class AdditivePlusNet(nn.Module):
    def __init__(self, content_dim: int, learn_scale: bool = True, learn_bias: bool = True):
        super().__init__()
        self.content_dim = int(content_dim)
        left_scale = torch.ones(self.content_dim)
        right_scale = torch.ones(self.content_dim)
        bias = torch.zeros(self.content_dim)
        if learn_scale:
            self.left_scale = nn.Parameter(left_scale)
            self.right_scale = nn.Parameter(right_scale)
        else:
            self.register_buffer("left_scale", left_scale)
            self.register_buffer("right_scale", right_scale)
        if learn_bias:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("bias", bias)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * self.left_scale + right * self.right_scale + self.bias


class IdentityProjector(nn.Module):
    def __init__(self, content_dim: int):
        super().__init__()
        self.content_dim = int(content_dim)
        self.op_dim = int(content_dim)

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return content

    def decode(self, op_content: torch.Tensor) -> torch.Tensor:
        return op_content

    def recon_loss(self, content: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=content.device, dtype=content.dtype)


class PCAProjector(nn.Module):
    def __init__(self, mean: torch.Tensor, components: torch.Tensor):
        super().__init__()
        if components.ndim != 2:
            raise ValueError("PCA components must have shape [op_dim, content_dim].")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("components", components.detach().clone())
        self.op_dim = int(components.size(0))
        self.content_dim = int(components.size(1))

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return (content - self.mean) @ self.components.t()

    def decode(self, op_content: torch.Tensor) -> torch.Tensor:
        return op_content @ self.components + self.mean

    def recon_loss(self, content: torch.Tensor) -> torch.Tensor:
        recon = self.decode(self.encode(content))
        return F.mse_loss(recon, content)


def _make_mlp(in_dim: int, hidden_dim: int, hidden_layers: int, out_dim: int) -> nn.Sequential:
    if hidden_layers < 0:
        raise ValueError("hidden_layers must be non-negative.")
    if hidden_layers == 0:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class LearnedProjector(nn.Module):
    def __init__(
        self,
        content_dim: int,
        op_dim: int,
        hidden_dim: int,
        hidden_layers: int,
    ):
        super().__init__()
        self.content_dim = int(content_dim)
        self.op_dim = int(op_dim)
        self.encoder = _make_mlp(content_dim, hidden_dim, hidden_layers, op_dim)
        self.decoder = _make_mlp(op_dim, hidden_dim, hidden_layers, content_dim)

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        return self.encoder(content)

    def decode(self, op_content: torch.Tensor) -> torch.Tensor:
        return self.decoder(op_content)

    def recon_loss(self, content: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.decode(self.encode(content)), content)


def commit_loss_mean(commit_loss: torch.Tensor | float | int) -> torch.Tensor:
    if not isinstance(commit_loss, torch.Tensor):
        return torch.tensor(float(commit_loss))
    return commit_loss.mean()


def quantize_content_ste(
    v3_wrapper,
    raw_content: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantized_hard, indices, _ = v3_wrapper.quantize_content(raw_content)
    commit_loss = F.mse_loss(raw_content, quantized_hard.detach())
    quantized = raw_content + (quantized_hard - raw_content).detach()
    return quantized, indices, commit_loss


class AdditionModel(nn.Module):
    def __init__(self, projector: nn.Module, plus_net: ContentPlusNet, prediction_mode: str = "content"):
        super().__init__()
        if prediction_mode not in {"content", "code_logits"}:
            raise ValueError("prediction_mode must be 'content' or 'code_logits'.")
        self.projector = projector
        self.plus_net = plus_net
        self.prediction_mode = prediction_mode

    def plus_content(
        self,
        v3_wrapper,
        left_content: torch.Tensor,
        right_content: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_op = self.projector.encode(left_content)
        right_op = self.projector.encode(right_content)
        raw_pred = self.plus_net(left_op, right_op)
        if self.prediction_mode == "code_logits":
            codebook = v3_wrapper.model.vq.codebook.detach().to(raw_pred.device)
            probs = F.softmax(raw_pred, dim=-1)
            soft_content = probs @ codebook
            indices = raw_pred.argmax(dim=-1)
            hard_content = codebook[indices]
            quantized = soft_content + (hard_content - soft_content).detach()
            commit_loss = torch.zeros((), device=raw_pred.device, dtype=raw_pred.dtype)
            return raw_pred, quantized, indices, commit_loss

        raw_content = self.projector.decode(raw_pred)
        quantized, indices, commit_loss = quantize_content_ste(v3_wrapper, raw_content)
        return raw_content, quantized, indices, commit_loss

    def projector_recon_loss(self, content: torch.Tensor) -> torch.Tensor:
        return self.projector.recon_loss(content)

    def codebook_logits(
        self,
        v3_wrapper,
        raw_content: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if self.prediction_mode == "code_logits":
            return raw_content
        codebook = v3_wrapper.model.vq.codebook.detach().to(raw_content.device)
        if codebook.ndim != 2:
            raise ValueError(f"Expected V3 codebook shape [n_codes, dim], got {tuple(codebook.shape)}.")
        distances = torch.sum((raw_content.unsqueeze(1) - codebook.unsqueeze(0)) ** 2, dim=-1)
        return -distances / max(float(temperature), 1e-8)


def plus_once(
    plus_net: ContentPlusNet,
    v3_wrapper,
    left: torch.Tensor,
    right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model = AdditionModel(IdentityProjector(left.size(-1)), plus_net).to(left.device)
    return model.plus_content(v3_wrapper, left, right)


def full_symmetry_loss(
    addition_model: AdditionModel,
    v3_wrapper,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, ab, _, commit_ab = addition_model.plus_content(v3_wrapper, a, b)
    _, abc_1, _, commit_abc_1 = addition_model.plus_content(v3_wrapper, ab, c)

    _, ac, _, commit_ac = addition_model.plus_content(v3_wrapper, a, c)
    _, acb_1, _, commit_acb_1 = addition_model.plus_content(v3_wrapper, ac, b)
    _, bac_2, _, commit_bac_2 = addition_model.plus_content(v3_wrapper, b, ac)

    _, bc, _, commit_bc = addition_model.plus_content(v3_wrapper, b, c)
    _, abc_2, _, commit_abc_2 = addition_model.plus_content(v3_wrapper, a, bc)

    loss = F.mse_loss(abc_1, acb_1) + F.mse_loss(abc_2, bac_2)
    commit = commit_ab + commit_abc_1 + commit_ac + commit_acb_1 + commit_bac_2 + commit_bc + commit_abc_2
    return loss, commit
