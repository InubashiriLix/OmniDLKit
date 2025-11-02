import pytest
import torch

from esn_core import (
    AttentionPool,
    ESNCell,
    ESNConfig,
    GroupConfig,
    GroupedReservoir,
    SimpleChannelEncoder,
)
from readout import ridge_readout_fit, ridge_readout_predict


def test_esn_cell_forward_matches_manual_update():
    cfg = ESNConfig(
        input_dim=2,
        hidden_dim=2,
        leak=0.5,
        spectral_radius=0.0,
        bias_scale=0.0,
        seed=123,
    )
    cell = ESNCell(cfg, device=None)

    with torch.no_grad():
        cell.Win.copy_(torch.tensor([[0.2, -0.1], [0.0, 0.3]]))
        cell.W.copy_(torch.tensor([[0.5, -0.4], [0.1, 0.2]]))
        cell.bias.zero_()

    U = torch.tensor([[1.0, -0.5], [0.3, 0.2]], dtype=torch.float32)
    out = cell(U)

    leak = cfg.leak
    h = torch.zeros(cfg.hidden_dim)
    manual_states = []
    for u in U:
        pre = cell.Win @ u + cell.W @ h + cell.bias
        h = (1 - leak) * h + leak * torch.tanh(pre)
        manual_states.append(h)
    expected = torch.stack(manual_states, dim=0)

    assert torch.allclose(out, expected, atol=1e-6)


def test_esn_cell_scales_to_target_spectral_radius():
    cfg = ESNConfig(input_dim=1, hidden_dim=5, spectral_radius=1.5, seed=7)
    cell = ESNCell(cfg, device=None)

    eigvals = torch.linalg.eigvals(cell.W.detach())
    spectral_radius = eigvals.abs().max().item()

    assert spectral_radius == pytest.approx(cfg.spectral_radius, rel=0.1)


def test_esn_cell_invalid_activation_raises():
    cfg = ESNConfig(input_dim=1, nonlin="sigmoid")
    with pytest.raises(ValueError):
        ESNCell(cfg, device=None)


def test_simple_channel_encoder_shapes_and_normalisation():
    encoder = SimpleChannelEncoder(length=8, zscore=True, device=None)
    channels = {
        "matrix": torch.arange(16.0).view(4, 4),
        "vector": torch.arange(6.0),
        "exact": torch.arange(24.0).view(8, 3),
    }

    encoded = encoder.encode(channels)

    assert set(encoded.keys()) == set(channels.keys())

    matrix = encoded["matrix"]
    assert matrix.shape == (8, 1)
    assert torch.allclose(matrix.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(matrix.std(), torch.tensor(1.0), atol=1e-6)

    vector = encoded["vector"]
    assert vector.shape == (8, 1)
    assert torch.allclose(vector.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(vector.std(), torch.tensor(1.0), atol=1e-6)

    exact = encoded["exact"]
    assert exact.shape == (8, 3)
    assert torch.allclose(exact.mean(dim=0), torch.zeros(3), atol=1e-6)
    assert torch.allclose(exact.std(dim=0), torch.ones(3), atol=1e-6)


def test_simple_channel_encoder_without_zscore_preserves_values():
    encoder = SimpleChannelEncoder(length=4, zscore=False, device=None)
    channels = {"data": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}

    encoded = encoder.encode(channels)["data"]

    expected = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert encoded.shape == (4, 1)
    assert torch.allclose(encoded, expected)


def test_attention_pool_softmax_weighting():
    att = AttentionPool(hidden_dim=3)
    with torch.no_grad():
        att.q.copy_(torch.ones(3))

    H = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
        dtype=torch.float32,
    )
    scores = (H * att.q).sum(dim=1)
    weights = torch.softmax(scores, dim=0).unsqueeze(1)
    expected = (H * weights).sum(dim=0)

    pooled = att(H)

    assert torch.allclose(pooled, expected)
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_grouped_reservoir_forward_matches_manual_aggregation():
    groups = [
        GroupConfig(
            name="g_last",
            channels=["a", "b"],
            esn_cfgs=[
                ESNConfig(input_dim=2, hidden_dim=3, spectral_radius=0.9, seed=1),
                ESNConfig(input_dim=2, hidden_dim=3, spectral_radius=1.1, seed=2),
            ],
            fuse="concat",
            time_pool="last",
            washout=1,
        ),
        GroupConfig(
            name="g_mean",
            channels=["c"],
            esn_cfgs=[
                ESNConfig(input_dim=3, hidden_dim=2, spectral_radius=0.8, seed=3),
                ESNConfig(input_dim=3, hidden_dim=2, spectral_radius=1.0, seed=4),
            ],
            fuse="mean",
            time_pool="mean",
            washout=0,
        ),
        GroupConfig(
            name="g_max",
            channels=["d"],
            esn_cfgs=[
                ESNConfig(input_dim=2, hidden_dim=2, spectral_radius=0.7, seed=5)
            ],
            fuse="last",
            time_pool="max",
            washout=0,
        ),
        GroupConfig(
            name="g_att",
            channels=["e"],
            esn_cfgs=[
                ESNConfig(input_dim=2, hidden_dim=2, spectral_radius=0.6, seed=6)
            ],
            fuse="concat",
            time_pool="att",
            washout=0,
        ),
    ]
    gres = GroupedReservoir(groups, device=None)

    torch.manual_seed(0)
    encoded = {
        "a": torch.randn(5, 1),
        "b": torch.randn(5, 1),
        "c": torch.randn(5, 3),
        "d": torch.randn(5, 2),
        "e": torch.randn(5, 2),
    }

    with torch.no_grad():
        features = gres.forward_features(encoded)

        expected_parts = []
        for g, esns, att in zip(gres.groups, gres.group_esns, gres.att_pools):
            U_list = [encoded[ch] for ch in g.channels]
            L = min(u.shape[0] for u in U_list)
            U = torch.cat([u[:L] for u in U_list], dim=1)

            state_vecs = []
            for esn in esns:  # pyright: ignore[reportGeneralTypeIssues]
                H = esn(U)
                H_use = H[g.washout :] if g.washout > 0 else H

                if g.time_pool == "last":
                    state = H_use[-1]
                elif g.time_pool == "mean":
                    state = H_use.mean(dim=0)
                elif g.time_pool == "max":
                    state, _ = H_use.max(dim=0)
                elif g.time_pool == "att":
                    state = att(H_use)
                else:
                    raise AssertionError("Unexpected time_pool setting")
                state_vecs.append(state)

            if g.fuse == "concat":
                gfeat = torch.cat(state_vecs, dim=0)
            elif g.fuse == "mean":
                gfeat = torch.stack(state_vecs, dim=0).mean(dim=0)
            elif g.fuse == "last":
                gfeat = state_vecs[-1]
            else:
                raise AssertionError("Unexpected fuse setting")
            expected_parts.append(gfeat)

        expected = torch.cat(expected_parts, dim=0)

    assert torch.allclose(features, expected, atol=1e-6)


def test_grouped_reservoir_missing_channel_raises():
    groups = [
        GroupConfig(
            name="g",
            channels=["needed"],
            esn_cfgs=[
                ESNConfig(input_dim=1, hidden_dim=2, spectral_radius=0.9, seed=0)
            ],
        )
    ]
    gres = GroupedReservoir(groups, device=None)

    with pytest.raises(ValueError):
        gres.forward_features({"other": torch.zeros(5, 1)})


def test_ridge_readout_fit_and_predict_recovers_linear_map():
    torch.manual_seed(42)
    H = torch.randn(50, 3)
    W_true = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-2.0, 1.0]])
    b_true = torch.tensor([0.3, -0.7])
    Y = H @ W_true + b_true

    W, b = ridge_readout_fit(H, Y, lam=1e-6, add_bias=True)
    Y_hat = ridge_readout_predict(H, W, b)

    assert torch.allclose(W, W_true, atol=1e-5)
    assert torch.allclose(b, b_true, atol=1e-5)
    assert torch.allclose(Y_hat, Y, atol=1e-6)


def test_ridge_readout_fit_without_bias_handles_zero_mean_targets():
    torch.manual_seed(0)
    H = torch.randn(40, 4)
    W_true = torch.tensor([[1.0], [-1.0], [0.5], [2.0]])
    Y = H @ W_true

    W, b = ridge_readout_fit(H, Y, lam=1e-4, add_bias=False)
    Y_hat = ridge_readout_predict(H, W, b)

    assert torch.allclose(W, W_true, atol=1e-5)
    assert torch.allclose(b, torch.zeros_like(b))
    assert torch.allclose(Y_hat, Y, atol=1e-5)
