import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, List
from copy import deepcopy

from .utils import get_activation


@dataclass
class MLPChannelCfg:
    """Configuration for a single MLP channel.

    Attributes:
        channel_name: Name identifier for the channel.
        hidden: Number of hidden units in the channel's MLP layer.
        act_: Activation function type (ReLU/Sigmoid/Tanh/LeakyReLU/ELU/GELU/
            Softplus/Swish).
        dropout: Dropout probability for regularization.
    """

    channel_name: str
    hidden: int
    act_: str
    dropout: float


@dataclass
class MLPHeadCfg:
    """Configuration for the fusion head.

    Attributes:
        hidden: Number of hidden units in the head's hidden layer.
        dropout: Dropout probability for regularization.
        act_: Activation function type for the hidden layer.
        n_cls: Number of output classes.
    """

    hidden: int
    dropout: float
    act_: str
    n_cls: int


class FusionMLP(nn.Module):
    """Multi-channel MLP with fusion head for multi-modal inputs.

    This module processes multiple input channels through separate MLP branches,
    concatenates their outputs, and passes them through a fusion head for final
    prediction. Each channel can have different dimensions and configurations.

    Note:
        Channel order matters and depends on Python 3.7+ dict ordering.

    Attributes:
        channels_cfg: List of channel configurations.
        channels: ModuleList of MLP branches for each input channel.
        head: Sequential fusion head for final prediction.
    """

    def __init__(
        self,
        channels: Dict[str, Tuple[MLPChannelCfg, torch.Tensor]],
        head_cfg: MLPHeadCfg,
    ):
        """Initializes the FusionMLP.

        Args:
            channels: Dictionary mapping channel names to tuples of
                (MLPChannelCfg, sample_tensor). The sample_tensor is used to
                infer input dimensions for each channel.
            head_cfg: Configuration for the fusion head.
        """
        super().__init__()

        self.channels_cfg: List[MLPChannelCfg] = []
        self.channels = nn.ModuleList()

        for name, v in channels.items():
            cfg, sample_tensor = v
            self.channels_cfg.append(cfg)
            self.channels.append(
                nn.Sequential(
                    nn.Linear(sample_tensor.shape[1], cfg.hidden),
                    deepcopy(get_activation(cfg.act_)),
                    nn.Dropout(cfg.dropout),
                )
            )

        total_hidden = sum([cfg.hidden for cfg in self.channels_cfg])
        self.head = nn.Sequential(
            nn.Linear(total_hidden, head_cfg.hidden),
            deepcopy(get_activation(head_cfg.act_)),
            nn.Dropout(head_cfg.dropout),
            nn.Linear(head_cfg.hidden, head_cfg.n_cls),
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through all channels and fusion head.

        Args:
            *inputs: Variable number of input tensors, one for each channel.
                Must match the number and order of channels defined during
                initialization.

        Returns:
            Output tensor of shape (batch_size, n_cls).

        Raises:
            AssertionError: If number of inputs doesn't match number of channels.
        """
        assert len(inputs) == len(self.channels), (
            f"Expected {len(self.channels)} inputs, got {len(inputs)}"
        )

        channel_outputs = [channel(x) for channel, x in zip(self.channels, inputs)]
        fused = torch.cat(channel_outputs, dim=1)
        return self.head(fused)
