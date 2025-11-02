"""
The Leaky Integrate-and-Fire (LIF) neuron layer and LIF conv2d implementation in PyTorch.

Copyright (c) 2025 InubashiriLix Author. All Rights Reserved.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFLayer(nn.Module):
    """Leaky Integrate-and-Fire (LIF) layer for fully connected architectures."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        dt: float = 1.0,
        tau_m: float = 20.0,
        tau_s: float = 5.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        t_ref: float = 2.0,
        w_scale: float = 0.2,
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the LIF neuron layer.

        Args:
            n_in (int): Number of presynaptic neurons.
            n_out (int): Number of postsynaptic neurons.
            dt (float, optional): Simulation time step in milliseconds.
            tau_m (float, optional): Membrane time constant controlling voltage decay.
            tau_s (float, optional): Synaptic time constant for presynaptic filtering.
            v_th (float, optional): Membrane threshold that triggers a spike.
            v_reset (float, optional): Membrane potential after a spike.
            t_ref (float, optional): Refractory window measured in simulation steps.
            w_scale (float, optional): Maximum absolute value for uniform weight init.
            seed (int, optional): Random seed for weight initialization.
            device (torch.device | None, optional): Target device for module buffers.
            dtype (torch.dtype, optional): Floating point dtype for computations.
        """
        super().__init__()

        # random generator
        self.g = torch.Generator().manual_seed(seed)

        # ==============================================================================================================
        # the leaking parameters:
        # beta = 0 < exp(-dt / t) < 1, the more it connects to 1, the slower it leaks, that is, remember it for longer
        beta_m = torch.exp(torch.tensor(-dt / tau_m, dtype=dtype))
        self.register_buffer("beta_m", beta_m)
        # beta_s = 0 < exp(-dt / t) < 1, the more it connects to 1, the slower it leaks, that is, remember it for longer
        beta_s = torch.exp(torch.tensor(-dt / tau_s, dtype=dtype))
        self.register_buffer("beta_s", beta_s)

        # v threshold
        self.register_buffer("v_th", torch.tensor(v_th, dtype=dtype))
        # reset potential after spike
        self.register_buffer("v_reset", torch.tensor(v_reset, dtype=dtype))
        # cache python float for type checkers and scalar ops
        self._v_reset_f: float = float(v_reset)
        self._beta_m_f: float = math.exp(-dt / tau_m)
        self._beta_s_f: float = math.exp(-dt / tau_s)
        self._v_th_f: float = float(v_th)

        # the time for refractory period in steps, that is, this neuron will not fire for these many steps after a spike
        self.t_ref_steps = int(round(t_ref / dt))
        self.n_in, self.n_out = n_in, n_out

        # ======================================= WEIGHTS INITIALIZATION ==============================================
        # geenrate initial weights uniformly in [-w_scale, w_scale]
        W0 = (
            torch.rand((n_in, n_out), generator=self.g, dtype=dtype) * 2 - 1
        ) * w_scale
        self.W = nn.Parameter(W0, requires_grad=False)

        # the runtime states. and will be allocated at the first time run `step or run` according to the batch size
        self._state_ready = False
        if device is not None:
            self.to(device=device, dtype=dtype)

    def reset_state(self, batch_size: int, device: torch.device | None = None):
        """Allocates and resets state tensors for a new sequence.

        Args:
            batch_size (int): Number of parallel samples expected in subsequent calls.
            device (torch.device | None, optional): Overrides the parameter device.
        """
        dev = device if device is not None else self.W.device
        dt = self.W.dtype

        # 膜电位初值
        self.v = torch.full(
            (batch_size, self.n_out), self._v_reset_f, device=dev, dtype=dt
        )
        # 不应期计数器：>0 表示处在不应期
        self.ref_cnt = torch.zeros(
            (batch_size, self.n_out), device=dev, dtype=torch.int32
        )
        # 突触滤波态：对输入做低通（相当于 IIR/指数滑动平均）
        self.x_pre = torch.zeros((batch_size, self.n_in), device=dev, dtype=dt)

        self._state_ready = True

    def step(
        self, s_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advances the neuron dynamics by one simulation step.

        Args:
            s_in (torch.Tensor): Input spikes of shape `(batch_size, n_in)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Boolean spike output,
                membrane potential, and synaptic current. Each tensor has shape
                `(batch_size, n_out)`.

        Raises:
            ValueError: If the input does not have shape `(batch_size, n_in)`.
        """

        if s_in.dim() != 2 or s_in.size(1) != self.n_in:
            raise ValueError(
                f"Invalid input shape: {s_in.shape}, and it should be (B, {self.n_in})"
            )

        B = s_in.size(0)
        if not self._state_ready or self.v.size(0) != B:
            self.reset_state(B, device=s_in.device)

        s_in = s_in.to(dtype=self.W.dtype)

        # the low pass filter of input
        # like if [0, 1, 0, 0, 0], the x_pre will Attenuate slowly like [0, 0.2, 0.36, 0.488, 0.5904], just like remains the short term memory
        self.x_pre.mul_(self._beta_s_f).add_(s_in)

        # then the x_pre is transformed to synaptic current by weights
        # NOTE: this step is different from the normal LIF model, cause' we use x_pre after low pass filter instead of s_in directly
        i_syn = self.x_pre @ self.W  # (B, n_out)

        # the leaky integration of membrane potential
        self.v.mul_(self._beta_m_f).add_(i_syn)

        # the refractory mechanism
        refractory = self.ref_cnt > 0
        self.v[refractory] = self._v_reset_f

        # if spike happens, then reset the membrane potential and set the refractory counter
        # wehter the spike happens
        spk = (self.v >= self._v_th_f) & (~refractory)

        self.v[spk] = self._v_reset_f
        self.ref_cnt[spk] = self.t_ref_steps

        # for those in refractory period, decrease the counter
        dec_mask = (~spk) & refractory
        self.ref_cnt[dec_mask] -= 1
        self.ref_cnt.clamp_(min=0)

        return spk, self.v.clone(), i_syn.clone()

    def run(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulates the layer over an input sequence.

        Args:
            S (torch.Tensor): Input spikes of shape `(T, B, n_in)` or `(T, n_in)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Time-major spike raster,
                membrane potential trajectory, and synaptic current trajectory.
        """

        if S.dim() == 2:  # if the current input is (T, n_in), unsqueeze to (T, 1, n_in)
            S = S.unsqueeze(1)
        T, B, Nin = S.shape
        assert Nin == self.n_in, f"Input shape mismatch: {S.shape}"

        if not self._state_ready or self.v.size(0) != 0:
            self.reset_state(B, device=S.device)

        spikes = torch.zeros((T, B, self.n_out), dtype=torch.bool, device=S.device)
        v_traj = torch.zeros((T, B, self.n_out), dtype=self.W.dtype, device=S.device)
        i_traj = torch.zeros((T, B, self.n_out), dtype=self.W.dtype, device=S.device)

        for t in range(T):
            spk, v, i = self.step(S[t])
            spikes[t], v_traj[t], i_traj[t] = spk, v, i

        return spikes, v_traj, i_traj

    @torch.no_grad()
    def set_weights(self, W: torch.Tensor):
        """Copies pre-specified weights into the layer.

        Args:
            W (torch.Tensor): Weight matrix with shape `(n_in, n_out)`.
        """
        assert W.shape == (self.n_in, self.n_out)
        self.W.copy_(W.to(self.W.dtype).to(self.W.device))


class LIFConv2d(nn.Module):
    """Leaky Integrate-and-Fire layer with convolutional connectivity."""

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
        tau_s: float = 5.0,  # 突触时间常数（输入平滑的“记忆长度”）
        v_th: float = 1.0,  # 放电阈值
        v_reset: float = 0.0,  # 放电后重置电位
        t_ref: float = 2.0,  # 不应期（单位=步）
        w_scale: float = 0.2,  # 权重初始化范围（越大=电流越强）
        seed: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the convolutional LIF layer.

        Args:
            in_channels (int): Number of presynaptic channels.
            out_channels (int): Number of postsynaptic channels.
            kernel_size (int | tuple[int, int], optional): Convolution kernel size.
            stride (int | tuple[int, int], optional): Convolution stride.
            padding (int | tuple[int, int], optional): Input padding.
            dilation (int | tuple[int, int], optional): Dilation factor.
            dt (float, optional): Simulation time step in milliseconds.
            tau_m (float, optional): Membrane time constant controlling voltage decay.
            tau_s (float, optional): Synaptic time constant for presynaptic filtering.
            v_th (float, optional): Spike threshold voltage.
            v_reset (float, optional): Reset voltage after a spike.
            t_ref (float, optional): Refractory window measured in simulation steps.
            w_scale (float, optional): Maximum absolute value for uniform weight init.
            seed (int, optional): Random seed for weight initialization.
            device (torch.device | None, optional): Target device for module buffers.
            dtype (torch.dtype, optional): Floating point dtype for computations.
        """
        super().__init__()

        # random generator
        self.g = torch.Generator().manual_seed(seed)

        # ==============================================================================================================
        # the leaking parameters:
        # beta = 0 < exp(-dt / t) < 1, the more it connects to 1, the slower it leaks, that is, remember it for longer
        beta_m = torch.exp(torch.tensor(-dt / tau_m, dtype=dtype))
        self.register_buffer("beta_m", beta_m)
        # beta_s = 0 < exp(-dt / t) < 1, the more it connects to 1, the slower it leaks, that is, remember it for longer
        beta_s = torch.exp(torch.tensor(-dt / tau_s, dtype=dtype))
        self.register_buffer("beta_s", beta_s)

        # v threshold
        self.register_buffer("v_th", torch.tensor(v_th, dtype=dtype))
        # reset potential after spike
        self.register_buffer("v_reset", torch.tensor(v_reset, dtype=dtype))
        # cache python float for type checkers and scalar ops
        self._v_reset_f: float = float(v_reset)
        self._beta_m_f: float = math.exp(-dt / tau_m)
        self._beta_s_f: float = math.exp(-dt / tau_s)
        self._v_th_f: float = float(v_th)

        # the time for refractory peroid in steps, that is, this neuron will not fire for these many steps after a spike
        self.t_ref_steps = int(round(t_ref / dt))

        # conv params (record for forward)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if isinstance(kernel_size, int):
            kH, kW = kernel_size, kernel_size
        else:
            kH, kW = kernel_size
        self.kernel_size = (kH, kW)
        self.in_channels, self.out_channels = in_channels, out_channels

        W0 = (
            torch.rand(
                (out_channels, in_channels, kH, kW), generator=self.g, dtype=dtype
            )
            * 2
            - 1
        ) * w_scale
        self.W = nn.Parameter(W0, requires_grad=False)

        self._state_ready = False
        if device is not None:
            self.to(device=device, dtype=dtype)

    def reset_state(
        self,
        batch_size: int,
        in_hw: tuple[int, int],
        device: torch.device | None = None,
    ):
        """Allocates and resets state tensors for convolutional simulation.

        Args:
            batch_size (int): Number of parallel samples expected in subsequent calls.
            in_hw (tuple[int, int]): Input spatial dimensions `(height, width)`.
            device (torch.device | None, optional): Overrides the parameter device.
        """
        dev = device if device is not None else self.W.device
        dt = self.W.dtype
        H, W = in_hw

        self.x_pre = torch.zeros(
            (batch_size, self.in_channels, H, W), device=dev, dtype=dt
        )

        dummy = torch.zeros((batch_size, self.in_channels, H, W), device=dev, dtype=dt)
        out = F.conv2d(
            dummy,
            self.W,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        self.v = torch.full_like(out, self._v_reset_f)
        self.ref_cnt = torch.zeros_like(out, dtype=torch.int32)

        self._sate_ready = True

    def step(
        self, s_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advances the convolutional layer by one simulation step.

        Args:
            s_in (torch.Tensor): Input spikes of shape `(batch_size, C_in, H, W)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Boolean spike output,
                membrane potential, and synaptic current. Each tensor has shape
                `(batch_size, C_out, H_out, W_out)`.

        Raises:
            ValueError: If the input is not four-dimensional or channel count differs.
        """
        if s_in.dim() != 4 or s_in.size(1) != self.in_channels:
            raise ValueError(
                f"Invalid input shape: {s_in.shape}, and it should be (B, {self.in_channels}, H, W)"
            )

        B, _, H, W = s_in.shape
        if not self._state_ready:
            self.reset_state(B, in_hw=(H, W), device=s_in.device)
        elif (
            self.x_pre.size(0) != B
            or self.x_pre.size(2) != H
            or self.x_pre.size(3) != W
        ):
            # 输入空间/批次变了，重新分配
            self.reset_state(B, in_hw=(H, W), device=s_in.device)

        s_in = s_in.to(dtype=self.W.dtype)

        # the low pass filter of input
        # like if a pixel fired once, the x_pre will attenuate slowly, just like short-term memory
        self.x_pre.mul_(self._beta_s_f).add_(s_in)

        # then the x_pre is transformed to synaptic current by conv kernels
        # NOTE: this step is different from the normal LIF model, cause' we use x_pre after low pass filter instead of s_in directly
        i_syn = F.conv2d(
            self.x_pre,
            self.W,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # the leaky integration of membrane potential
        self.v.mul_(self._beta_m_f).add_(i_syn)

        # the refractory mechanism
        refractory = self.ref_cnt > 0
        self.v[refractory] = self._v_reset_f

        # wehter the spike happens
        spk = (self.v >= self._v_th_f) & (~refractory)

        # reset & set refractory
        self.v[spk] = self._v_reset_f
        self.ref_cnt[spk] = self.t_ref_steps

        # decrease refractory counter for those still in refractory but not spiking
        dec_mask = (~spk) & refractory
        self.ref_cnt[dec_mask] -= 1
        self.ref_cnt.clamp_(min=0)

        return spk, self.v.clone(), i_syn.clone()

    def run(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulates the convolutional layer over an input sequence.

        Args:
            S (torch.Tensor): Input spikes of shape `(T, B, C_in, H, W)` or
                `(T, C_in, H, W)`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Time-major spike raster,
                membrane potential trajectory, and synaptic current trajectory.
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
            self.reset_state(B, in_hw=(H, W), device=S.device)

        # 先做一帧，确定输出空间大小（更稳妥）
        with torch.no_grad():
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
        """Copies pre-specified convolutional weights into the layer.

        Args:
            W (torch.Tensor): Weight tensor with shape
                `(out_channels, in_channels, kernel_height, kernel_width)`.
        """
        assert W.shape == (self.out_channels, self.in_channels, *self.kernel_size)
        self.W.copy_(W.to(self.W.dtype).to(self.W.device))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lif = LIFLayer(
        1,
        1,
        dt=1.0,
        tau_m=10.0,
        tau_s=5.0,
        v_th=1.0,
        v_reset=0.0,
        t_ref=2.0,
        w_scale=0.0,
        device=device,
    )
    lif.set_weights(torch.tensor([[0.25]]))  # 单输入-单输出，权重=0.25

    T = 30
    S = torch.zeros((T, 1, 1), device=device)
    S[:20, 0, 0] = 1.0
    lif.reset_state(batch_size=1, device=device)
    spikes, v_traj, i_traj = lif.run(S)

    print("t  |x_pre≈(隐含) | I_syn | v | spike")
    for t in range(20):
        i_val = float(i_traj[t, 0, 0].cpu())
        v_val = float(v_traj[t, 0, 0].cpu())
        spk = int(spikes[t, 0, 0].cpu())
        print(f"{t:2d} |    -        | {i_val:4.2f} | {v_val:4.2f} | {spk}")
