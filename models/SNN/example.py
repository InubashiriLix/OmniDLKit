# example.py — MNIST + LIFConv2dSTDP（无监督 STDP 预训练 + 可选线性读出，带“点火”自修复）
# 依赖：pip install torch torchvision matplotlib

from __future__ import annotations
import math, time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 改成你的类所在文件名（保持类名 LIFConv2dSTDP 不变）===
from STDP import LIFConv2dSTDP


# ====================== 配置 ======================
@dataclass
class Cfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    data_root: str = "./data"
    batch_size: int = 64

    # ====== 时序编码（泊松） ======
    T: int = 80  # 每张图重复的时间步数
    rate_gain: float = 0.30  # 每步发放概率≈ pix * rate_gain (提高以增加输入)
    rate_bias: float = 0.05  # 给低灰度一点底噪，避免全静默

    # ====== LIF + STDP ======
    out_channels: int = 16
    ksize: int = 5
    v_th: float = 0.8  # 正常训练阈值
    warmup_v_th: float = 0.4  # 热启动阈值（点火用）降低以更容易触发
    warmup_batches: int = 10
    t_ref: float = 2.0
    tau_m: float = 10.0
    tau_s: float = 5.0
    tau_pre: float = 20.0
    tau_post: float = 20.0
    eta_plus: float = 5e-4  # 降低学习率以防止权重快速衰减
    eta_minus: float = 2e-4  # LTD 学习率要小于 LTP
    wmin: float = 0.0  # 权重下界
    wmax: float = 1.5  # 权重上界（增大以允许更强的突触）
    stdp_norm: bool = True  # 启用归一化以稳定学习
    w_scale: float = 0.5  # 增大初始权重范围

    # ====== 训练日志/时长 ======
    train_batches: int = 200
    log_every: int = 20

    # ====== 可视化卷积核 ======
    save_kernel_png: str | None = "kernels_after.png"

    # ====== 线性读出（可选，粗略） ======
    do_linear_eval: bool = True
    feats_T: int = 40
    linear_epochs: int = 5
    lr_linear: float = 1e-2
    linear_train_samples: int = 2000


# ====================== 工具函数 ======================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def poisson_encode(
    imgs: torch.Tensor, T: int, gain: float, bias: float
) -> torch.Tensor:
    """
    imgs: (B,1,28,28) ∈ [0,1]
    return: (T,B,1,28,28) 的 0/1 浮点脉冲
    """
    rate = (imgs.clamp(0, 1) * gain + bias).clamp(max=1.0)
    U = torch.rand((T, *imgs.shape), device=imgs.device, dtype=imgs.dtype)
    return (U < rate).to(imgs.dtype)


def get_loaders(cfg: Cfg):
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(cfg.data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(cfg.data_root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader


def make_model(cfg: Cfg) -> LIFConv2dSTDP:
    lif = LIFConv2dSTDP(
        in_channels=1,
        out_channels=cfg.out_channels,
        kernel_size=cfg.ksize,
        padding=cfg.ksize // 2,  # 保持空间尺寸
        v_th=cfg.v_th,
        t_ref=cfg.t_ref,
        tau_m=cfg.tau_m,
        tau_s=cfg.tau_s,
        tau_pre=cfg.tau_pre,
        tau_post=cfg.tau_post,
        eta_plus=cfg.eta_plus,
        eta_minus=cfg.eta_minus,
        wmin=cfg.wmin,
        wmax=cfg.wmax,
        stdp_norm=cfg.stdp_norm,
        w_scale=cfg.w_scale,
        device=torch.device(cfg.device),
    )

    # —— 点火前检查：强制正数初始化，避免被 wmin=0 clamp 到 0 —— #
    with torch.no_grad():
        # 初始化到较大的正数范围，确保有足够的电流产生脉冲
        lif.W.uniform_(0.3, 0.6)
        # 再打印一次确保不是 0
        wmin2, wmax2 = float(lif.W.min()), float(lif.W.max())
        print(f"[init] W range = [{wmin2:.3f}, {wmax2:.3f}]")

    return lif


def _set_thresh(lif: LIFConv2dSTDP, val: float):
    """同时更新 buffer 和 python float 缓存，确保阈值生效"""
    val = float(val)
    lif.v_th.fill_(val)
    lif._v_th_f = val


def _ignite_if_needed(
    cfg: Cfg, lif: LIFConv2dSTDP, S: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """
    尝试点火：若第一次 run 后 spikes=0，依次用 (1) 降阈值 (2) 重新正数初始化 权重。
    返回一次“保证有脉冲”的 spikes（若最终仍 0，会保持 0 并打印提示）。
    """
    B = S.size(1)
    lif.reset_states(B, (H, W), device=torch.device(cfg.device))
    spikes, _, _ = lif.run(S)
    spk_sum = int(spikes.to(torch.int).sum().item())
    if spk_sum > 0:
        return spikes

    # 尝试1：进一步降阈值
    _set_thresh(lif, 0.3)
    lif.reset_states(B, (H, W), device=torch.device(cfg.device))
    spikes, _, _ = lif.run(S)
    spk_sum = int(spikes.to(torch.int).sum().item())
    if spk_sum > 0:
        print(f"[ignite] success by lowering threshold to 0.3 (spikes={spk_sum})")
        return spikes

    # 尝试2：强制正数重初始化 + 更低阈值
    with torch.no_grad():
        lif.W.uniform_(0.4, 0.8)
    _set_thresh(lif, 0.25)
    lif.reset_states(B, (H, W), device=torch.device(cfg.device))
    spikes, _, _ = lif.run(S)
    spk_sum = int(spikes.to(torch.int).sum().item())
    if spk_sum > 0:
        print(f"[ignite] success by reinit W∈[0.4,0.8] & v_th=0.25 (spikes={spk_sum})")
        return spikes

    print("[ignite] still no spikes — 请检查 STDP 类是否被其它地方改写/覆盖为全 0 权重")
    return spikes


def train_stdp(cfg: Cfg, lif: LIFConv2dSTDP, loader: DataLoader):
    lif.train()
    H = W = 28
    t0 = time.time()

    for bi, (x, _) in enumerate(loader):
        if bi >= cfg.train_batches:
            break

        # 热启动阈值
        if bi == 0:
            _set_thresh(lif, cfg.warmup_v_th)
        elif bi == cfg.warmup_batches:
            _set_thresh(lif, cfg.v_th)

        x = x.to(cfg.device).to(lif.W.dtype)  # (B,1,28,28)
        B = x.size(0)
        S = poisson_encode(x, cfg.T, cfg.rate_gain, cfg.rate_bias)  # (T,B,1,28,28)

        if bi == 0:
            # 第一个 batch 做“点火”自检
            spikes = _ignite_if_needed(cfg, lif, S, H, W)
        else:
            lif.reset_states(B, (H, W), device=torch.device(cfg.device))
            spikes, _, _ = lif.run(S)

        if (bi + 1) % cfg.log_every == 0:
            spk_sum = int(spikes.to(torch.int).sum().item())
            wmin, wmax = float(lif.W.min()), float(lif.W.max())
            print(
                f"[{bi + 1:04d}/{cfg.train_batches}] spikes={spk_sum:>7d} | W∈[{wmin:.3f},{wmax:.3f}]"
            )

    print(
        f"STDP done in {time.time() - t0:.1f}s. weight range=({float(lif.W.min()):.3f},{float(lif.W.max()):.3f})"
    )


def maybe_save_kernels(cfg: Cfg, lif: LIFConv2dSTDP):
    if cfg.save_kernel_png is None:
        return
    try:
        W = lif.W.detach().float().cpu()  # (C_out,1,k,k)
        C, _, kH, kW = W.shape
        cols = int(math.ceil(math.sqrt(C)))
        rows = int(math.ceil(C / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        axes = axes.flatten()
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis("off")
            if i < C:
                w = W[i, 0]
                w = (w - w.min()) / (w.max() - w.min() + 1e-6)
                ax.imshow(w.numpy(), cmap="viridis", interpolation="nearest")
        plt.tight_layout()
        plt.savefig(cfg.save_kernel_png, dpi=140)
        print(f"saved kernels to {cfg.save_kernel_png}")
    except Exception as e:
        print(f"[warn] save kernels skipped: {e}")


# ============== 可选：冻结后做一个极简线性读出（粗略 sanity check） ==============
class LinearHead(torch.nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


@torch.no_grad()
def extract_features(
    cfg: Cfg, lif: LIFConv2dSTDP, loader: DataLoader, T: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lif.eval()
    H = W = 28
    old_eta_p, old_eta_m = lif.eta_plus, lif.eta_minus
    lif.eta_plus = 0.0
    lif.eta_minus = 0.0

    feats, labels = [], []
    for x, y in loader:
        x = x.to(cfg.device).to(lif.W.dtype)
        B = x.size(0)
        S = poisson_encode(x, T, cfg.rate_gain, cfg.rate_bias)
        lif.reset_states(B, (H, W), device=torch.device(cfg.device))
        spikes, _, _ = lif.run(S)
        z = spikes.to(torch.float32).sum(dim=(0, 3, 4)) / T  # (B,Cout)
        feats.append(z.cpu())
        labels.append(y)

    lif.eta_plus, lif.eta_minus = old_eta_p, old_eta_m
    return torch.cat(feats, 0), torch.cat(labels, 0)


def linear_eval(
    cfg: Cfg, lif: LIFConv2dSTDP, train_loader: DataLoader, test_loader: DataLoader
):
    base_ds = train_loader.dataset
    n = min(cfg.linear_train_samples, len(base_ds))
    small_ds = Subset(base_ds, list(range(n)))
    small_loader = DataLoader(small_ds, batch_size=cfg.batch_size, shuffle=False)

    X_tr, y_tr = extract_features(cfg, lif, small_loader, T=cfg.feats_T)
    X_te, y_te = extract_features(cfg, lif, test_loader, T=cfg.feats_T)

    head = LinearHead(X_tr.shape[1], 10).to(cfg.device)
    opt = torch.optim.SGD(head.parameters(), lr=cfg.lr_linear, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    head.train()
    ds = TensorDataset(X_tr.to(cfg.device), y_tr.to(cfg.device))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    for ep in range(cfg.linear_epochs):
        tot, nsum = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            nsum += xb.size(0)
        print(f"[linear] epoch {ep + 1}/{cfg.linear_epochs} loss={tot / nsum:.4f}")

    head.eval()
    with torch.no_grad():
        logits = head(X_te.to(cfg.device))
        pred = logits.argmax(dim=1).cpu()
        acc = (pred == y_te).float().mean().item()
    print(f"[linear] test acc ≈ {acc * 100:.2f}%（粗略，仅作 sanity check）")


# ====================== 主流程 ======================
if __name__ == "__main__":
    cfg = Cfg()
    set_seed(cfg.seed)
    train_loader, test_loader = get_loaders(cfg)
    lif = make_model(cfg)

    print("===> STDP pretrain on MNIST …")
    train_stdp(cfg, lif, train_loader)
    maybe_save_kernels(cfg, lif)

    if cfg.do_linear_eval:
        print("===> linear readout (optional)")
        linear_eval(cfg, lif, train_loader, test_loader)
