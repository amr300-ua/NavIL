"""Arquitecturas de política. Ciclo 1: MLP 92 -> 128 -> 128 -> 64 -> 2."""

from typing import Sequence

import torch
from torch import nn

from il_policy.contract import INPUT_SIZE, OUTPUT_SIZE


class MLPPolicy(nn.Module):
    """MLP de Behavioral Cloning.

    Entrada: vector normalizado (LiDAR + objetivo), forma (B, 92).
    Salida: (v, w) normalizados en [-1, 1] gracias a ``tanh``; la clase
    ``ILPolicy`` los reescala a m/s y rad/s.
    """

    def __init__(self, input_size: int = INPUT_SIZE,
                 hidden: Sequence[int] = (128, 128, 64),
                 output_size: int = OUTPUT_SIZE, dropout: float = 0.0):
        super().__init__()
        self.config = {
            'arch': 'mlp', 'input_size': int(input_size),
            'hidden': [int(h) for h in hidden],
            'output_size': int(output_size), 'dropout': float(dropout),
        }
        layers = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def build_model(config: dict) -> nn.Module:
    """Reconstruye un modelo a partir de ``model.config`` (guardado en el checkpoint)."""
    arch = config.get('arch', 'mlp')
    if arch == 'mlp':
        return MLPPolicy(
            input_size=config.get('input_size', INPUT_SIZE),
            hidden=config.get('hidden', (128, 128, 64)),
            output_size=config.get('output_size', OUTPUT_SIZE),
            dropout=config.get('dropout', 0.0),
        )
    raise ValueError(f'Arquitectura desconocida: {arch}')


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
