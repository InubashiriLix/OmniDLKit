"""Echo State Network (ESN) module for reservoir computing.

This module provides a complete Echo State Network implementation with support
for multi-channel inputs, grouped reservoirs, and flexible readout mechanisms.
ESNs are a type of recurrent neural network where only the readout weights are
trained, making them computationally efficient for temporal sequence processing.

Core Components:
    ESNCell: Single reservoir cell implementing leaky-integrator dynamics.
    ESNConfig: Configuration dataclass for ESN hyperparameters.
    GroupedReservoir: Multi-group reservoir architecture for complex inputs.
    GroupConfig: Configuration for reservoir groups with fusion strategies.
    SingleReservoirCell: Alternative lightweight ESN cell implementation.
    SingleReservoirConfig: Configuration dataclass for the lightweight ESN cell.

Utilities:
    SimpleChannelEncoder: Preprocessing for multi-channel temporal data.
    AttentionPool: Attention-based temporal pooling mechanism.

Readout Functions:
    ridge_readout_fit: Closed-form ridge regression solver for readout weights.
    ridge_readout_predict: Apply trained readout to reservoir features.

Example:
    >>> from ESN import ESNCell, ESNConfig, GroupedReservoir, GroupConfig
    >>> from ESN import ridge_readout_fit, ridge_readout_predict
    >>> import torch
    >>>
    >>> # Configure single reservoir
    >>> cfg = ESNConfig(
    ...     input_dim=10,
    ...     hidden_dim=200,
    ...     leak=0.3,
    ...     spectral_radius=0.9,
    ...     nonlin="tanh",
    ... )
    >>>
    >>> # Create ESN cell and process sequence
    >>> esn = ESNCell(cfg, device=torch.device("cpu"))
    >>> U = torch.randn(100, 10)  # Time series: (T=100, D=10)
    >>> H = esn(U)  # Reservoir states: (100, 200)
    >>>
    >>> # Train readout layer
    >>> Y = torch.randn(100, 3)  # Target outputs
    >>> W, b = ridge_readout_fit(H, Y, lam=1e-3)
    >>> predictions = ridge_readout_predict(H, W, b)
    >>>
    >>> # Multi-group reservoir example
    >>> group_cfg = GroupConfig(
    ...     name="multimodal",
    ...     channels=["audio", "video"],
    ...     esn_cfgs=[cfg],
    ...     fuse="concat",
    ...     time_pool="last",
    ... )
    >>> reservoir = GroupedReservoir([group_cfg])
"""

from .esn_core import (
    ESNCell,
    ESNConfig,
    GroupedReservoir,
    GroupConfig,
    SimpleChannelEncoder,
    AttentionPool,
)
from .readout import ridge_readout_fit, ridge_readout_predict
from .single_reservoir import ESNCell as SingleReservoirCell
from .single_reservoir import ESNConfig as SingleReservoirConfig

__all__ = [
    # Core ESN components
    "ESNCell",
    "ESNConfig",
    "GroupedReservoir",
    "GroupConfig",
    "SingleReservoirCell",
    "SingleReservoirConfig",
    # Utilities
    "SimpleChannelEncoder",
    "AttentionPool",
    # Readout functions
    "ridge_readout_fit",
    "ridge_readout_predict",
]

__version__ = "0.1.0"
