"""Unit tests for FusionMLP module.

This module contains pytest-based tests for the FusionMLP class to verify
proper channel fusion, forward pass, and configuration handling.
"""

import pytest
import torch
import torch.nn as nn
from MLP import FusionMLP, MLPChannelCfg, MLPHeadCfg


class TestFusionMLP:
    """Test suite for FusionMLP."""

    @pytest.fixture
    def two_channel_config(self):
        """Creates a basic two-channel configuration.

        Returns:
            Tuple of (channels_dict, head_cfg) for testing.
        """
        channels = {
            "channel1": (
                MLPChannelCfg(
                    channel_name="channel1",
                    hidden=32,
                    act_="relu",
                    dropout=0.1,
                ),
                torch.randn(4, 10),  # Sample tensor with 10 features
            ),
            "channel2": (
                MLPChannelCfg(
                    channel_name="channel2",
                    hidden=16,
                    act_="gelu",
                    dropout=0.2,
                ),
                torch.randn(4, 5),  # Sample tensor with 5 features
            ),
        }
        head_cfg = MLPHeadCfg(
            hidden=24,
            dropout=0.15,
            act_="relu",
            n_cls=3,
        )
        return channels, head_cfg

    @pytest.fixture
    def three_channel_config(self):
        """Creates a three-channel configuration.

        Returns:
            Tuple of (channels_dict, head_cfg) for testing.
        """
        channels = {
            "audio": (
                MLPChannelCfg(
                    channel_name="audio",
                    hidden=64,
                    act_="tanh",
                    dropout=0.1,
                ),
                torch.randn(8, 20),
            ),
            "video": (
                MLPChannelCfg(
                    channel_name="video",
                    hidden=128,
                    act_="leakyrelu",
                    dropout=0.2,
                ),
                torch.randn(8, 50),
            ),
            "text": (
                MLPChannelCfg(
                    channel_name="text",
                    hidden=32,
                    act_="gelu",
                    dropout=0.15,
                ),
                torch.randn(8, 15),
            ),
        }
        head_cfg = MLPHeadCfg(
            hidden=100,
            dropout=0.2,
            act_="relu",
            n_cls=5,
        )
        return channels, head_cfg

    def test_initialization_two_channels(self, two_channel_config):
        """Tests model initialization with two channels."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        assert len(model.channels) == 2
        assert len(model.channels_cfg) == 2
        assert model.channels_cfg[0].hidden == 32
        assert model.channels_cfg[1].hidden == 16

    def test_initialization_three_channels(self, three_channel_config):
        """Tests model initialization with three channels."""
        channels, head_cfg = three_channel_config
        model = FusionMLP(channels, head_cfg)

        assert len(model.channels) == 3
        assert len(model.channels_cfg) == 3

    def test_forward_two_channels(self, two_channel_config):
        """Tests forward pass with two input channels."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 5)

        output = model(x1, x2)

        assert output.shape == (4, 3)  # batch_size=4, n_cls=3
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_three_channels(self, three_channel_config):
        """Tests forward pass with three input channels."""
        channels, head_cfg = three_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(8, 20)
        x2 = torch.randn(8, 50)
        x3 = torch.randn(8, 15)

        output = model(x1, x2, x3)

        assert output.shape == (8, 5)  # batch_size=8, n_cls=5
        assert not torch.isnan(output).any()

    def test_forward_wrong_input_count(self, two_channel_config):
        """Tests that forward raises error with wrong number of inputs."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(4, 10)

        with pytest.raises(AssertionError):
            model(x1)  # Missing second input

    def test_forward_wrong_input_dimensions(self, two_channel_config):
        """Tests forward pass with incorrect input dimensions."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(4, 15)  # Wrong: should be 10 features
        x2 = torch.randn(4, 5)

        with pytest.raises(RuntimeError):
            model(x1, x2)

    def test_channel_fusion(self, two_channel_config):
        """Tests that channels are properly fused."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 5)

        # Run forward and verify fusion dimension
        # After channel processing: 32 + 16 = 48 features
        with torch.no_grad():
            channel_outputs = [
                channel(x) for channel, x in zip(model.channels, [x1, x2])
            ]
            fused = torch.cat(channel_outputs, dim=1)
            assert fused.shape == (4, 48)  # 32 + 16

    def test_different_batch_sizes(self, two_channel_config):
        """Tests forward pass with different batch sizes."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        for batch_size in [1, 8, 16, 32]:
            x1 = torch.randn(batch_size, 10)
            x2 = torch.randn(batch_size, 5)
            output = model(x1, x2)
            assert output.shape == (batch_size, 3)

    def test_gradient_flow(self, two_channel_config):
        """Tests that gradients flow through all channels."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(4, 10, requires_grad=True)
        x2 = torch.randn(4, 5, requires_grad=True)

        output = model(x1, x2)
        loss = output.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None
        assert not torch.all(x1.grad == 0)
        assert not torch.all(x2.grad == 0)

    @pytest.mark.parametrize("act_type", [
        "relu", "sigmoid", "tanh", "leakyrelu", "elu", "gelu", "softplus", "swish"
    ])
    def test_different_activations(self, act_type):
        """Tests model with different activation functions."""
        channels = {
            "ch1": (
                MLPChannelCfg(
                    channel_name="ch1",
                    hidden=16,
                    act_=act_type,
                    dropout=0.1,
                ),
                torch.randn(4, 8),
            ),
        }
        head_cfg = MLPHeadCfg(
            hidden=12,
            dropout=0.1,
            act_=act_type,
            n_cls=2,
        )

        model = FusionMLP(channels, head_cfg)
        x = torch.randn(4, 8)
        output = model(x)

        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()

    def test_eval_mode(self, two_channel_config):
        """Tests that dropout is disabled in eval mode."""
        channels, head_cfg = two_channel_config
        model = FusionMLP(channels, head_cfg)

        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 5)

        model.eval()
        with torch.no_grad():
            output1 = model(x1, x2)
            output2 = model(x1, x2)

        # In eval mode, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_single_channel(self):
        """Tests model with only a single channel."""
        channels = {
            "solo": (
                MLPChannelCfg(
                    channel_name="solo",
                    hidden=20,
                    act_="relu",
                    dropout=0.1,
                ),
                torch.randn(4, 10),
            ),
        }
        head_cfg = MLPHeadCfg(
            hidden=15,
            dropout=0.1,
            act_="relu",
            n_cls=2,
        )

        model = FusionMLP(channels, head_cfg)
        x = torch.randn(4, 10)
        output = model(x)

        assert output.shape == (4, 2)
