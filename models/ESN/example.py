import torch
import torch.nn.functional as F

from esn_core import (
    GroupedReservoir,
    ESNConfig,
    GroupConfig,
    SimpleChannelEncoder,
)

from readout import ridge_readout_fit, ridge_readout_predict


# ========== Demo：分类 ==========
def demo_classification(device="cpu"):
    torch.manual_seed(0)
    device = torch.device(device)

    # --- 通道编码器 ---
    enc = SimpleChannelEncoder(length=256, zscore=True, device=device)

    # --- 两个group，模拟“不同动力学/不同通道组合” ---
    g1_cfgs = [
        ESNConfig(
            input_dim=3,
            hidden_dim=120,
            leak=0.1,
            spectral_radius=1.4,
            input_scale=0.5,
            nonlin="tanh",
            seed=1,
        ),
        ESNConfig(
            input_dim=3,
            hidden_dim=120,
            leak=0.5,
            spectral_radius=0.9,
            input_scale=1.0,
            nonlin="satlin",
            seed=2,
        ),
    ]
    g2_cfgs = [
        ESNConfig(
            input_dim=2,
            hidden_dim=100,
            leak=0.8,
            spectral_radius=0.7,
            input_scale=2.0,
            nonlin="relu",
            seed=3,
        ),
    ]
    groups = [
        GroupConfig(
            name="texture_color",
            channels=["Y", "Edge", "Sat"],
            esn_cfgs=g1_cfgs,
            fuse="concat",
            time_pool="mean",
            washout=10,
        ),
        GroupConfig(
            name="structure",
            channels=["Grad", "Entropy"],
            esn_cfgs=g2_cfgs,
            fuse="concat",
            time_pool="last",
            washout=5,
        ),
    ]
    gres = GroupedReservoir(groups, device=device).to(device)

    # --- 造一点玩具数据：3类，每类20样本，每样本5个伪通道 ---
    N_per, C = 20, 3
    X_feats, Y_onehot = [], []
    for cls in range(C):
        for _ in range(N_per):
            Yc = torch.randn(32, 32, device=device) + cls * 0.5
            Edge = torch.abs(torch.randn(32, 32, device=device))
            Sat = torch.rand(32, 32, device=device)
            Grad = torch.abs(torch.randn(32, 32, device=device))
            Entropy = torch.rand(32, 32, device=device)

            encoded = enc.encode(
                {"Y": Yc, "Edge": Edge, "Sat": Sat, "Grad": Grad, "Entropy": Entropy}
            )
            feat = gres.forward_features(encoded)  # [F]
            X_feats.append(feat)
            oh = torch.zeros(C, device=device)
            oh[cls] = 1.0
            Y_onehot.append(oh)

    X = torch.stack(X_feats, dim=0)  # [N, F]
    Y = torch.stack(Y_onehot, dim=0)  # [N, C]

    # --- 岭回归读出 ---
    W, b = ridge_readout_fit(X, Y, lam=1e-3, add_bias=True)
    logits = ridge_readout_predict(X, W, b)
    pred = logits.argmax(dim=1)
    acc = (pred == Y.argmax(dim=1)).float().mean().item()
    print(f"[Classification] N={X.shape[0]}, F={X.shape[1]}  acc={acc:.3f}")


# ========== Demo：回归（轨迹下一步Δx,Δy） ==========
def demo_regression(device="cpu"):
    torch.manual_seed(123)
    device = torch.device(device)
    enc = SimpleChannelEncoder(length=50, zscore=True, device=device)

    g_cfgs = [
        ESNConfig(
            input_dim=2,
            hidden_dim=80,
            leak=0.2,
            spectral_radius=1.2,
            input_scale=1.0,
            seed=7,
        ),
        ESNConfig(
            input_dim=2,
            hidden_dim=80,
            leak=0.7,
            spectral_radius=0.8,
            input_scale=2.0,
            seed=8,
        ),
    ]
    groups = [
        GroupConfig(
            name="traj",
            channels=["xy"],
            esn_cfgs=g_cfgs,
            fuse="concat",
            time_pool="last",
            washout=5,
        )
    ]
    gres = GroupedReservoir(groups, device=device).to(device)

    # 合成数据：N条轨迹，三种“转向”倾向（左/直/右）
    N = 200
    X_feats, Yt = [], []
    for _ in range(N):
        mode = torch.randint(0, 3, (1,)).item()
        noise = lambda s: torch.randn(51, device=device) * s
        x = torch.cumsum(noise(0.1) + (0.2 if mode == 2 else 0.0), dim=0)  # 右倾
        y = torch.cumsum(noise(0.1) + (0.2 if mode == 0 else 0.0), dim=0)  # 左倾
        dxdy_next = torch.stack([x[50] - x[49], y[50] - y[49]])  # 预测目标

        encoded = enc.encode({"xy": torch.stack([x[:50], y[:50]], dim=1)})  # [T,2]
        feat = gres.forward_features(encoded)
        X_feats.append(feat)
        Yt.append(dxdy_next)

    X = torch.stack(X_feats, dim=0)  # [N, F]
    Y = torch.stack(Yt, dim=0)  # [N, 2]

    # 岭回归读出
    W, b = ridge_readout_fit(X, Y, lam=1e-3, add_bias=True)
    Yhat = ridge_readout_predict(X, W, b)
    mse = F.mse_loss(Yhat, Y).item()
    print(f"[Regression] N={X.shape[0]}, F={X.shape[1]}  MSE={mse:.4f}")


if __name__ == "__main__":
    # 可选：device="cuda" 或 "mps"
    demo_classification(device="cpu")
    demo_regression(device="cpu")
