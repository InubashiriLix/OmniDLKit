import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, List

from .utils import get_activation_factory


@dataclass
class MLPCfg:
    input_size: int
    hidden_size_layers: List[int]
    output_size: int
    act_: str


class MLP(nn.Module):
    def __init__(self, cfg: MLPCfg) -> None:
        super().__init__()
        self.cfg = cfg

        # the multiple layers
        layers = []

        # add each layers
        temp_last_size: int = self.cfg.input_size

        # NOTE: it includes the header layer
        for layer_size in self.cfg.hidden_size_layers:
            layers.append(nn.Linear(temp_last_size, layer_size))
            layers.append(get_activation_factory(self.cfg.act_))
            temp_last_size = layer_size

        # readout layer
        layers.append(nn.Linear(self.cfg.hidden_size_layers[-1], self.cfg.output_size))

        # the net!
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
