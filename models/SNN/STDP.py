"""Spike-timing-dependent plasticity utilities for convolutional LIF neurons.

This module defines `LIFConv2dSTDP`, a leaky integrate-and-fire convolutional
layer that performs unsupervised weight updates with spike-timing-dependent
plasticity (STDP). The layer keeps track of membrane dynamics, refractory
periods, and pre/post synaptic traces to update weights online during the
forward simulation.

Typical usage example:

    layer = LIFConv2dSTDP(in_channels=1, out_channels=8)
    layer.reset_states(batch_size=8, in_hw=(16, 16))
    spikes, v_traj, i_traj = layer.run(spike_train)
"""

from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFConv2dSTDP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        dt: float = 1.0,  # 离散时间步
        tau_m: float = 20.0,  # 膜时间常数（越大=漏得慢）
        tau_s: float = 5.0,  # 突触时间常数（输入低通的“记忆长度”）
        v_th: float = 1.0,  # 放电阈值
        v_reset: float = 0.0,  # 放电后重置电位
        t_ref: float = 2.0,  # 不应期（单位=步）
        # ---- STDP 超参 ----
        tau_pre: float = 20.0,  # 预突触痕迹时间常数
        tau_post: float = 20.0,  # 后突触痕迹时间常数
        eta_plus: float = 5e-4,  # LTP 学习率（post事件）
        eta_minus: float = 5e-4,  # LTD 学习率（pre事件）
        wmin: float = 0.0,  # 权重下界
        wmax: float = 1.0,  # 权重上界
        stdp_norm: bool = True,  # 是否按 (B*Hout*Wout) 归一化 dW
        w_scale: float = 0.2,  # 初始权重范围（越大=电流越强）
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # random generator
        self.g = torch.Generator().manual_seed(seed)

        # ==============================================================================================================
        # the leaking parameters:
        # beta = 0 < exp(-dt / t) < 1, the more it connects to 1, the slower it leaks, that is, remember it for longer
        beta_m = torch.exp(torch.tensor(-dt / tau_m, dtype=dtype))
        beta_s = torch.exp(torch.tensor(-dt / tau_s, dtype=dtype))
        beta_pre = torch.exp(torch.tensor(-dt / tau_pre, dtype=dtype))
        beta_post = torch.exp(torch.tensor(-dt / tau_post, dtype=dtype))
        self.register_buffer("beta_m", beta_m)
        self.register_buffer("beta_s", beta_s)
        self.register_buffer("beta_pre", beta_pre)
        self.register_buffer("beta_post", beta_post)

        # v threshold & reset
        self.register_buffer("v_th", torch.tensor(v_th, dtype=dtype))
        self.register_buffer("v_reset", torch.tensor(v_reset, dtype=dtype))

        # cache python float for type checkers and scalar ops
        self._beta_m_f: float = math.exp(-dt / tau_m)
        self._beta_s_f: float = math.exp(-dt / tau_s)
        self._beta_pre_f: float = math.exp(-dt / tau_pre)
        self._beta_post_f: float = math.exp(-dt / tau_post)
        self._v_th_f: float = float(v_th)
        self._v_reset_f: float = float(v_reset)
        self._wmin_f: float = float(wmin)
        self._wmax_f: float = float(wmax)

        # refractory steps
        self.t_ref_steps = int(round(t_ref / dt))

        # conv params
        if isinstance(kernel_size, int):
            kH, kW = kernel_size, kernel_size
        else:
            kH, kW = kernel_size
        self.kernel_size = (kH, kW)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels, self.out_channels = in_channels, out_channels
        self.stdp_norm = stdp_norm
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus

        # ======================================= WEIGHTS INITIALIZATION ==============================================
        # generate initial conv kernels uniformly in [-w_scale, w_scale]
        W0 = (
            torch.rand(
                (out_channels, in_channels, kH, kW), generator=self.g, dtype=dtype
            )
            * 2
            - 1
        ) * w_scale
        self.W = nn.Parameter(W0, requires_grad=False)  # conv kernel; no bias

        # runtime states will be allocated at first step/run
        self._state_ready = False
        if device is not None:
            self.to(device=device, dtype=dtype)

    # ------------------------------------------------ 状态分配/重置 ------------------------------------------------
    def _alloc_states(self, B: int, in_hw: tuple[int, int], device: torch.device):
        dev = device if device is not None else self.W.device
        dt = self.W.dtype
        H, W = in_hw

        # 输入低通状态
        self.x_pre = torch.zeros((B, self.in_channels, H, W), device=dev, dtype=dt)

        # 用一次零输入推断输出空间大小
        dummy = torch.zeros((B, self.in_channels, H, W), device=dev, dtype=dt)
        out = F.conv2d(
            dummy,
            self.W,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        _, C_out, H_out, W_out = out.shape

        # 膜电位 & 不应期
        self.v = torch.full_like(out, self._v_reset_f)
        self.ref_cnt = torch.zeros_like(out, dtype=torch.int32)

        # STDP 痕迹
        self.pre_trace = torch.zeros((B, self.in_channels, H, W), device=dev, dtype=dt)
        self.post_trace = torch.zeros((B, C_out, H_out, W_out), device=dev, dtype=dt)

        self._state_ready = True

    def reset_states(
        self,
        batch_size: int,
        in_hw: tuple[int, int],
        device: torch.device | None = None,
    ):
        self._alloc_states(
            batch_size, in_hw, device if device is not None else self.W.device
        )

    # ------------------------------------------------ 单步前向+学习 ------------------------------------------------
    @torch.no_grad()
    def step(
        self, s_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step in one step one time step

        Args:
            `s_in`: The input spike train, shape [batch_size, C_in, H, W]
        Returns:
            `spk`:   (B, C_out, H_out, W_out) bool
            `v_now`: (B, C_out, H_out, W_out)
            `i_syn`: (B, C_out, H_out, W_out)
        Throws:
            `ValueError`: if the input shape is not 4D
        """

        if s_in.dim() != 4 or s_in.size(1) != self.in_channels:
            raise ValueError(
                f"Invalid input size: {s_in.shape}, and it should be (B, {self.in_channels}, H, W)"
            )
        B, _, H, W = s_in.shape

        if not self._state_ready:
            self._alloc_states(B, (H, W), s_in.device)
        elif (
            self.x_pre.size(0) != B
            or self.x_pre.size(2) != H
            or self.x_pre.size(3) != W
        ):
            self._alloc_states(B, (H, W), s_in.device)

        s_in = s_in.to(dtype=self.W.dtype)

        # input -> low
        # x_pre= beat_s * x_pre + s_in
        self.x_pre.mul_(self._beta_s_f).add_(s_in)

        # the current synaptic input
        i_syn = F.conv2d(
            self.x_pre,
            self.W,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # leaky integration
        self.v.mul_(self._beta_m_f).add_(i_syn)

        # the refractory mechanism
        refractory = self.ref_cnt > 0
        self.v[refractory] = self._v_reset_f

        #
        spk = (self.v >= self._v_th_f) & ~refractory

        # fire reset and refractory counter update
        self.v[spk] = self._v_reset_f
        self.ref_cnt[spk] = self.t_ref_steps
        dec_mask = (~spk) & refractory
        self.ref_cnt[dec_mask] -= 1
        self.ref_cnt.clamp_(min=0)

        # STDP: update the trace（注意：pre 用 beta_pre，post 用 beta_post）
        self.pre_trace.mul_(self._beta_pre_f).add_(s_in)
        self.post_trace.mul_(self._beta_post_f).add_(spk.to(dtype=self.W.dtype))

        # --------------------------- STDP: 权重更新（卷积外积 via unfold+einsum） --------------------
        # unfold 出每个输出位置对应的输入补丁: (B, Cin*kH*kW, L) where L=Hout*Wout
        kH, kW = self.kernel_size
        X_pre = F.unfold(
            self.pre_trace,
            kernel_size=(kH, kW),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # (B, Cin*kH*kW, L)
        X_spk = F.unfold(
            s_in,
            kernel_size=(kH, kW),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # (B, Cin*kH*kW, L)
        # 输出侧摊平成 (B, Cout, L)
        Y_spk = spk.to(self.W.dtype).flatten(2)  # (B, Cout, L)
        Y_post = self.post_trace.flatten(2)  # (B, Cout, L)

        # 归一化系数（可提升稳定性：与 batch 和空间大小无关）
        norm = float(B * Y_spk.size(-1)) if self.stdp_norm else 1.0

        # LTP（post 事件）：ΔW += η+ * (pre_trace ⊗ post_spike) ∘ (wmax - W)
        if Y_spk.any():
            dW_plus_vec = (
                torch.einsum("bkl,bol->ok", X_pre, Y_spk) / norm
            )  # (Cout, Cin*kH*kW)
            dW_plus = dW_plus_vec.view(self.out_channels, self.in_channels, kH, kW)
            self.W.add_(self.eta_plus * dW_plus * (self._wmax_f - self.W))

        # LTD（pre 事件）：ΔW -= η- * (pre_spike ⊗ post_trace) ∘ (W - wmin)
        if X_spk.any():
            dW_minus_vec = (
                torch.einsum("bkl,bol->ok", X_spk, Y_post) / norm
            )  # (Cout, Cin*kH*kW)
            dW_minus = dW_minus_vec.view(self.out_channels, self.in_channels, kH, kW)
            self.W.sub_(self.eta_minus * dW_minus * (self.W - self._wmin_f))

        # 边界裁剪
        self.W.clamp_(self._wmin_f, self._wmax_f)

        return spk, self.v.clone(), i_syn.clone()

    # ------------------------------------------------ 运行整段序列 ------------------------------------------------
    @torch.no_grad()
    def run(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """S: (T, B, C_in, H, W) or (T, C_in, H, W)
        Args:
            `S`: The input spike train
        Returns:
            `spikes`: (T, B, C_out, H_out, W_out) bool
            `v_traj`: (T, B, C_out, H_out, W_out)
            `i_traj`: (T, B, C_out, H_out, W_out)
        """
        if S.dim() == 4:  # (T, C, H, W) -> (T, 1, C, H, W)
            S = S.unsqueeze(1)
        if S.dim() != 5:
            raise ValueError(
                f"Invalid input shape: {S.shape}, should be (T,B,C,H,W) or (T,C,H,W)"
            )

        T, B, C, H, W = S.shape
        assert C == self.in_channels, (
            f"Input channels mismatch: {C} vs {self.in_channels}"
        )

        if not self._state_ready:
            self._alloc_states(B, (H, W), S.device)

        # 先过一帧确认输出形状
        x0 = torch.zeros((B, C, H, W), device=S.device, dtype=self.W.dtype)
        o0 = F.conv2d(
            x0,
            self.W,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        OH, OW = o0.shape[-2], o0.shape[-1]

        spikes = torch.zeros(
            (T, B, self.out_channels, OH, OW), dtype=torch.bool, device=S.device
        )
        v_traj = torch.zeros_like(spikes, dtype=self.W.dtype)
        i_traj = torch.zeros_like(spikes, dtype=self.W.dtype)

        for t in range(T):
            spk, v, i = self.step(S[t])
            spikes[t], v_traj[t], i_traj[t] = spk, v, i

        return spikes, v_traj, i_traj

    @torch.no_grad()
    def set_weights(self, W: torch.Tensor):
        assert W.shape == (self.out_channels, self.in_channels, *self.kernel_size)
        self.W.copy_(W.to(self.W.dtype).to(self.W.device))


# ----------------------------- 简单测试（点燃版） -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 in-channel, 8 out-channels, 3x3 conv，padding=1 保持尺寸
    lifc = LIFConv2dSTDP(
        1,
        8,
        kernel_size=3,
        padding=1,
        v_th=0.6,  # 降低阈值，便于放电
        t_ref=2,
        tau_m=10.0,
        tau_s=5.0,
        eta_plus=2e-3,  # 稍大一些，便于看到权重变化
        eta_minus=1e-3,
        wmin=0.0,
        wmax=1.0,
        stdp_norm=False,  # 先关归一化，更新更明显
        device=device,
    )

    # —— 关键：把卷积核初始化到“小正数”区间，避免一开始就被 clamp 到 0 —— #
    with torch.no_grad():
        lifc.W.uniform_(0.08, 0.18)  # 全为正，且不太小

    # 造一段 Poisson 脉冲：上半区更活跃（作为“模式”）
    # 提高时间步 & 发放率：更容易点燃
    T, B, H, W = 120, 16, 16, 16
    rate_bg, rate_hi = 0.05, 0.20
    rate = torch.full((T, B, 1, H, W), rate_bg, device=device, dtype=lifc.W.dtype)
    rate[:, :, :, : H // 2, :] = rate_hi
    S = (torch.rand_like(rate) < rate).to(lifc.W.dtype)

    # 跑一遍无监督学习
    lifc.reset_states(B, (H, W), device=device)
    spikes, v_traj, i_traj = lifc.run(S)

    # 打印简单统计：每个输出通道的总放电次数、权重范围
    spk_sum_per_out = spikes.to(torch.int).sum(dim=(0, 1, 3, 4)).flatten().tolist()
    print("spikes sum per out-channel:", spk_sum_per_out)
    print("weight range:", float(lifc.W.min()), float(lifc.W.max()))
