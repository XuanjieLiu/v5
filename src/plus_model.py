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
    ):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative.")
        layers = []
        in_dim = content_dim * 2
        if hidden_layers == 0:
            layers.append(nn.Linear(in_dim, content_dim))
        else:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            for _ in range(hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, content_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


def commit_loss_mean(commit_loss: torch.Tensor | float | int) -> torch.Tensor:
    if not isinstance(commit_loss, torch.Tensor):
        return torch.tensor(float(commit_loss))
    return commit_loss.mean()


def plus_once(
    plus_net: ContentPlusNet,
    v3_wrapper,
    left: torch.Tensor,
    right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = plus_net(left, right)
    quantized, indices, commit_loss = v3_wrapper.quantize_content(raw)
    return raw, quantized, indices, commit_loss_mean(commit_loss).to(raw.device)


def full_symmetry_loss(
    plus_net: ContentPlusNet,
    v3_wrapper,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, ab, _, commit_ab = plus_once(plus_net, v3_wrapper, a, b)
    _, abc_1, _, commit_abc_1 = plus_once(plus_net, v3_wrapper, ab, c)

    _, ac, _, commit_ac = plus_once(plus_net, v3_wrapper, a, c)
    _, acb_1, _, commit_acb_1 = plus_once(plus_net, v3_wrapper, ac, b)
    _, bac_2, _, commit_bac_2 = plus_once(plus_net, v3_wrapper, b, ac)

    _, bc, _, commit_bc = plus_once(plus_net, v3_wrapper, b, c)
    _, abc_2, _, commit_abc_2 = plus_once(plus_net, v3_wrapper, a, bc)

    loss = F.mse_loss(abc_1, acb_1) + F.mse_loss(abc_2, bac_2)
    commit = commit_ab + commit_abc_1 + commit_ac + commit_acb_1 + commit_bac_2 + commit_bc + commit_abc_2
    return loss, commit

