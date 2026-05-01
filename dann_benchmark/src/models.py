from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedMaskedDendriteLayer(nn.Module):
    """
    A lightweight dendritic layer.

    Each dendrite samples a fixed subset of input features, computes a weighted
    sum plus bias, and applies a local nonlinearity. The resulting dendritic
    activations are combined at the soma level through learnable cable weights.
    """

    def __init__(
        self,
        input_dim: int,
        soma_units: int,
        branches_per_soma: int,
        sample_size: int,
        dendrite_indices: torch.Tensor,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if dendrite_indices.ndim != 2:
            raise ValueError("dendrite_indices must be [num_dendrites, sample_size]")

        self.input_dim = int(input_dim)
        self.soma_units = int(soma_units)
        self.branches_per_soma = int(branches_per_soma)
        self.sample_size = int(sample_size)
        self.num_dendrites = self.soma_units * self.branches_per_soma
        self.negative_slope = float(negative_slope)

        if dendrite_indices.shape != (self.num_dendrites, self.sample_size):
            raise ValueError(
                "dendrite_indices shape mismatch: "
                f"expected {(self.num_dendrites, self.sample_size)} got {tuple(dendrite_indices.shape)}"
            )

        self.register_buffer("dendrite_indices", dendrite_indices.long())
        self.synaptic_weights = nn.Parameter(torch.empty(self.num_dendrites, self.sample_size))
        self.synaptic_bias = nn.Parameter(torch.zeros(self.num_dendrites))
        self.cable_weights = nn.Parameter(torch.empty(self.soma_units, self.branches_per_soma))
        self.soma_bias = nn.Parameter(torch.zeros(self.soma_units))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.synaptic_weights, a=self.negative_slope)
        fan_in = self.sample_size
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.synaptic_bias, -bound, bound)
        nn.init.kaiming_uniform_(self.cable_weights, a=self.negative_slope)
        bound_soma = 1.0 / math.sqrt(self.branches_per_soma)
        nn.init.uniform_(self.soma_bias, -bound_soma, bound_soma)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, input_dim]
        if x.ndim != 2:
            raise ValueError("input x must be [batch, input_dim]")
        gathered = x[:, self.dendrite_indices]  # [B, D, S]
        dendritic_preact = (gathered * self.synaptic_weights.unsqueeze(0)).sum(dim=-1) + self.synaptic_bias
        dendritic_act = F.leaky_relu(dendritic_preact, negative_slope=self.negative_slope)
        dendritic_act = dendritic_act.view(x.size(0), self.soma_units, self.branches_per_soma)
        soma_preact = (dendritic_act * self.cable_weights.unsqueeze(0)).sum(dim=-1) + self.soma_bias
        soma_act = F.leaky_relu(soma_preact, negative_slope=self.negative_slope)
        return soma_act, dendritic_act


class DendriticANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        soma_units: int,
        branches_per_soma: int,
        sample_size: int,
        dendrite_indices: torch.Tensor,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.dendritic = FixedMaskedDendriteLayer(
            input_dim=input_dim,
            soma_units=soma_units,
            branches_per_soma=branches_per_soma,
            sample_size=sample_size,
            dendrite_indices=dendrite_indices,
            negative_slope=negative_slope,
        )
        self.classifier = nn.Linear(soma_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        soma, _ = self.dendritic(x)
        return self.classifier(soma)


class NaiveBranchedLinear(nn.Module):
    """
    Control model for the reviewer attack:
    branching exists, but there is no dendrite-level nonlinearity.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        soma_units: int,
        branches_per_soma: int,
        sample_size: int,
        dendrite_indices: torch.Tensor,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.soma_units = soma_units
        self.branches_per_soma = branches_per_soma
        self.sample_size = sample_size
        self.num_dendrites = soma_units * branches_per_soma
        self.negative_slope = negative_slope

        self.register_buffer("dendrite_indices", dendrite_indices.long())
        self.synaptic_weights = nn.Parameter(torch.empty(self.num_dendrites, self.sample_size))
        self.synaptic_bias = nn.Parameter(torch.zeros(self.num_dendrites))
        self.cable_weights = nn.Parameter(torch.empty(self.soma_units, self.branches_per_soma))
        self.soma_bias = nn.Parameter(torch.zeros(self.soma_units))
        self.classifier = nn.Linear(self.soma_units, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.synaptic_weights, a=self.negative_slope)
        bound = 1.0 / math.sqrt(self.sample_size)
        nn.init.uniform_(self.synaptic_bias, -bound, bound)
        nn.init.kaiming_uniform_(self.cable_weights, a=self.negative_slope)
        bound_soma = 1.0 / math.sqrt(self.branches_per_soma)
        nn.init.uniform_(self.soma_bias, -bound_soma, bound_soma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gathered = x[:, self.dendrite_indices]
        branch_linear = (gathered * self.synaptic_weights.unsqueeze(0)).sum(dim=-1) + self.synaptic_bias
        branch_linear = branch_linear.view(x.size(0), self.soma_units, self.branches_per_soma)
        soma = (branch_linear * self.cable_weights.unsqueeze(0)).sum(dim=-1) + self.soma_bias
        soma = F.leaky_relu(soma, negative_slope=self.negative_slope)
        return self.classifier(soma)


class VanillaANN(nn.Module):
    """
    Same hidden-layer counts as the dendritic network, but fully connected.
    Hidden layer 1 uses num_dendrites units and hidden layer 2 uses soma_units.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_dendrites: int,
        soma_units: int,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_dendrites)
        self.fc2 = nn.Linear(num_dendrites, soma_units)
        self.fc3 = nn.Linear(soma_units, num_classes)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope)
        return self.fc3(x)


class FlatMLP(nn.Module):
    """
    Parameter-matched control. Two fully connected hidden layers with equal width.
    """

    def __init__(self, input_dim: int, num_classes: int, width: int, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, num_classes)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.negative_slope)
        return self.fc3(x)


@dataclass
class ModelBundle:
    model: nn.Module
    effective_name: str
    trainable_params: int


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_param_matched_width(target_params: int, input_dim: int, num_classes: int) -> int:
    best_w = 2
    best_diff = float("inf")
    for w in range(2, 4097):
        params = (input_dim + 1) * w + (w + 1) * w + (w + 1) * num_classes
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_w = w
    return best_w
