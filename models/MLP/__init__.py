"""Multi-Layer Perceptron (MLP) module for multi-channel fusion.

This module provides a flexible multi-channel MLP architecture that can fuse
inputs from multiple sources through separate processing branches and a unified
fusion head.

Classes:
    MLPChannelCfg: Configuration dataclass for individual MLP channels.
    MLPHeadCfg: Configuration dataclass for the fusion head.
    FusionMLP: Multi-channel MLP with fusion head for multi-modal inputs.
    MLPCfg: Configuration dataclass for the simplified MLP.
    MLP: Lightweight sequential MLP defined from ``MLPCfg``.

Functions:
    get_activation: Factory function to retrieve activation modules by name.

Example:
    >>> from MLP import FusionMLP, MLPChannelCfg, MLPHeadCfg
    >>> import torch
    >>>
    >>> # Define two channels with different configurations
    >>> channels = {
    ...     "audio": (
    ...         MLPChannelCfg(
    ...             channel_name="audio",
    ...             hidden=64,
    ...             act_="relu",
    ...             dropout=0.1,
    ...         ),
    ...         torch.randn(8, 20),  # Sample tensor for dimension inference
    ...     ),
    ...     "video": (
    ...         MLPChannelCfg(
    ...             channel_name="video",
    ...             hidden=128,
    ...             act_="gelu",
    ...             dropout=0.2,
    ...         ),
    ...         torch.randn(8, 50),
    ...     ),
    ... }
    >>>
    >>> # Configure fusion head
    >>> head_cfg = MLPHeadCfg(
    ...     hidden=100,
    ...     dropout=0.15,
    ...     act_="relu",
    ...     n_cls=5,
    ... )
    >>>
    >>> # Create model and perform forward pass
    >>> model = FusionMLP(channels, head_cfg)
    >>> x_audio = torch.randn(8, 20)
    >>> x_video = torch.randn(8, 50)
    >>> output = model(x_audio, x_video)  # Shape: (8, 5)
"""

from .MLP import FusionMLP, MLPChannelCfg, MLPHeadCfg
from .simpleMLP import MLPCfg, MLP
from .utils import get_activation_factory

__all__ = [
    "FusionMLP",
    "MLPChannelCfg",
    "MLPHeadCfg",
    "MLPCfg",
    "MLP",
    "get_activation_factory",
]

__version__ = "0.1.0"
