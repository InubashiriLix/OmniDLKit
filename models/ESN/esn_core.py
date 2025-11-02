"""Echo State Network core components. Echo State Network 核心组件。

This module provides a light-weight configuration dataclass and a single
reservoir cell implementation that together can be used to build Echo State
Networks (ESNs). The cell follows the classic leaky-integrator formulation and
supports configurable non-linearities, sparsity, and spectral radius scaling.
该模块提供轻量级配置数据类与单个储层单元实现，可组合构建 Echo State Network。
储层单元遵循经典的泄漏积分形式，并支持配置非线性函数、稀疏度与谱半径缩放。
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from numpy import generic
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ESNConfig:
    """Hyper-parameters describing ESN structure and dynamics. 定义 ESN 结构与动力学的超参数。

    Attributes:
        input_dim (int): Number of input features per timestep. 每个时间步的输入特征数。
        hidden_dim (int): Reservoir width (neurons). 储层神经元数量。
        leak (float): Leak rate in ``(0, 1]`` controlling state refresh. 泄漏率，控制状态更新强度。
        spectral_radius (float): Target spectral radius for ``W``. 递归权重 ``W`` 的目标谱半径。
        input_scale (float): Scaling factor for input weights. 输入权重的缩放系数。
        bias_scale (float): Standard deviation for bias sampling. 偏置采样的标准差。
        nonlin (str): Name of activation (``tanh``/``relu``/``satlin``). 激活函数名称（``tanh``、``relu``、``satlin``）。
        sparsity (float): Probability that entries in ``W`` are zeroed. ``W`` 中元素被置零的概率。
        seed (int): Random seed for reproducible init. 初始化的随机种子。
    """

    input_dim: int
    hidden_dim: int = 200
    leak: float = 0.3
    spectral_radius: float = 0.9
    input_scale: float = 1.0
    bias_scale: float = 0.0
    nonlin: str = "tanh"  # tanh | relu | satlin
    sparsity: float = 0.0  # 0~1，>0时对W做稀疏
    seed: int = 42


class ESNCell(nn.Module):
    """Single ESN reservoir cell responsible for state updates. 负责状态更新的单个 ESN 储层单元。

    Attributes:
        cfg (ESNConfig): Configuration object storing hyper-parameters. 存放超参数的配置对象。
        Win (torch.nn.Parameter): Input weight matrix ``(hidden_dim, input_dim)``. 输入权重矩阵。
        W (torch.nn.Parameter): Recurrent weight matrix ``(hidden_dim, hidden_dim)``. 递归权重矩阵。
        bias (torch.nn.Parameter): Bias vector ``(hidden_dim,)`` applied pre-activation. 预激活前添加的偏置向量。
        _act (Callable): Activation function applied inside the reservoir. 储层内部使用的激活函数。
    """

    def __init__(self, cfg: ESNConfig, device: Optional[torch.device]):
        """Initialise the reservoir cell and parameters. 初始化储层单元与参数。

        Args:
            cfg (ESNConfig): ESN configuration defining dimensions and dynamics. ESN 配置，定义尺度与动力学。
            device (Optional[torch.device]): Target device for tensors, e.g. ``cpu``/``cuda``. 张量应放置的设备，例如 ``cpu`` 或 ``cuda``。
        """

        super().__init__()
        self.cfg = cfg
        gen = torch.Generator(device=device).manual_seed(cfg.seed)

        # the input layer
        self.Win = torch.nn.Parameter(
            torch.randn(
                cfg.hidden_dim,
                cfg.input_dim,
                generator=gen,
                device=device,
                requires_grad=False,
            )
        )

        # the temp layer -> self.W
        W = torch.randn(
            cfg.hidden_dim,
            cfg.hidden_dim,
            generator=gen,
            device=device,
            requires_grad=True,
        )

        # make it sparse
        if cfg.sparsity > 0:
            mask = (
                torch.rand(W.shape, device=W.device, generator=gen) > cfg.sparsity
            ).float()
            W = W * mask

        self.W = nn.Parameter(W, requires_grad=True)

        self.bias = torch.nn.Parameter(
            torch.randn(cfg.hidden_dim, generator=gen, device=device) * cfg.bias_scale,
            requires_grad=False,
        )

        self._spectral_scale_(self.W, cfg.spectral_radius, iters=20)

        if cfg.nonlin == "tanh":
            self._act = torch.tanh
        elif cfg.nonlin == "relu":
            self._act = F.relu
        elif cfg.nonlin == "satlin":
            self._act = lambda x: torch.clamp(x, 0, 1)
        else:
            raise ValueError(
                f"nonlin {cfg.nonlin} not supported, use tanh, relu and stalin instead"
            )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """Iteratively compute hidden states for one sequence. 迭代计算单条序列的隐藏状态。

        Args:
            U (torch.Tensor): Sequence of shape ``(T, input_dim)``. ``T`` 表示时间步数的输入序列。

        Returns:
            torch.Tensor: Hidden trajectory ``(T, hidden_dim)`` capturing reservoir states. 返回储层隐藏状态轨迹 ``(T, hidden_dim)``。
        """

        # Track the number of timesteps in the incoming sequence.
        T, _ = U.shape
        # Pre-allocate the hidden trajectory tensor for efficient writes.
        H = torch.zeros(T, self.cfg.hidden_dim, device=U.device)
        # Initialise the reservoir state to zeros before processing the sequence.
        h = torch.zeros(self.cfg.hidden_dim, device=U.device)

        # Cache the leak rate to avoid repeated attribute lookups in the loop.
        leak = self.cfg.leak

        # Iterate through each timestep and propagate the reservoir dynamics.
        for t in range(T):
            # Combine the new input, previous state, and bias to form the pre-activation.
            pre = self.Win @ U[t] + self.W @ h + self.bias
            # Apply the leaky integrator update with the configured activation.
            h = (1 - leak) * h + leak * self._act(pre)
            # Persist the current hidden state in the trajectory tensor.
            H[t] = h

        return H

    @staticmethod
    def _spectral_scale_(W: torch.Tensor, target_radius: float, iters: int = 20):
        """Normalise ``W`` to the desired spectral radius. 将 ``W`` 归一化到指定谱半径。

        Args:
            W (torch.Tensor): Recurrent weight matrix to scale. 需要缩放的递归权重矩阵。
            target_radius (float): Desired spectral radius; ``<=0`` skips scaling. 目标谱半径，``<=0`` 时跳过缩放。
            iters (int): Power iteration count used to estimate the largest eigenvalue. 幂迭代次数，用于估计最大特征值。
        """

        # if the target_radius <= then return
        if target_radius <= 0:
            return
        with torch.no_grad():
            # randomlize a vector called v
            # throught multiple iteration, we can transform v as the main feature vector of W
            v = torch.randn(W.shape[0], 1, device=W.device)
            for _ in range(iters):
                v = F.normalize(torch.matmul(W, v), dim=0, eps=1e-8)

            s = torch.norm(W @ v)
            if s > 0:
                W.mul_(target_radius / s)


class SimpleChannelEncoder:
    """Flatten, crop, and normalise channels to uniform length. 将通道展开、裁剪并归一化到统一长度。

    Attributes:
        L (int): Target temporal length ``L`` for each channel. 每个通道的目标时间长度 ``L``。
        z (bool): Whether to apply z-score normalisation. 是否执行 z-score 归一化。
        device (Optional[torch.device]): Device to host encoded tensors. 编码张量所在设备。
    """

    def __init__(
        self,
        length: int = 256,
        zscore: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.L = length
        self.z = zscore
        self.device = device

    def encode(self, channels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode raw channel tensors into uniform sequences. 将原始通道张量编码为统一序列。

        Args:
            channels (Dict[str, torch.Tensor]): Mapping of channel names to tensors. 通道名称到张量的映射。

        Returns:
            Dict[str, torch.Tensor]: Processed channels with shape ``(L, D)``. 处理后的通道张量，形状为 ``(L, D)``。
        """
        out = {}
        for name, arr in channels.items():
            arr = arr.detach().clone().to(self.device)
            if arr.dim() == 2 and arr.shape[0] != self.L:
                v = arr.reshape(-1).float()  # flatten
                if v.numel() < self.L:
                    # WARNING: what the fuck? will this cause duplicated features
                    rep = math.ceil(self.L / v.numel())
                    v = v.repeat(rep)[: self.L]  # make sure this will larger than L
                else:
                    v = v[: self.L]
                if self.z:
                    v = (v - v.mean()) / (v.std() + 1e-6)

                out[name] = v.view(self.L, 1)  # [T, 1]
            elif arr.dim() == 2 and arr.shape[0] == self.L:
                v = arr.float()
                if self.z:
                    m = v.mean(dim=0, keepdim=True)
                    s = v.std(dim=0, keepdim=True)
                    v = (v - m) / (s + 1e-6)
                out[name] = v
            else:
                # for the other dimensions, we flatten it into [T, 1]
                v = arr.reshape(-1).float()
                if v.numel() < self.L:
                    rep = math.ceil(self.L / v.numel())
                    v = v.repeat(rep)[: self.L]
                else:
                    v = v[: self.L]
                if self.z:
                    v = (v - v.mean()) / (v.std() + 1e-6)
                out[name] = v.view(self.L, 1)
        return out


@dataclass
class GroupConfig:
    """Configuration for a reservoir group. 储层分组的配置。

    Attributes:
        name (str): Group identifier for logging or inspection. 分组标识，用于日志或调试。
        channels (List[str]): Channel names consumed by the group. 分组使用的通道名称列表。
        esn_cfgs (List[ESNConfig]): ESN configurations within the group. 分组内每个 ESN 的配置。
        fuse (str): Feature fusion strategy (``concat``/``mean``/``last``). 特征融合策略。
        time_pool (str): Temporal pooling method (``last``/``mean``/``max``/``att``). 时间汇聚方式。
        washout (int): Number of initial timesteps to discard. 需要丢弃的初始时间步数量。
    """

    name: str
    channels: List[str]  # 这个group吃哪些channel
    esn_cfgs: List[ESNConfig]  # 这个group里放几个ESN（参数各异）
    fuse: str = "concat"  # concat | mean | last
    time_pool: str = "last"  # last | mean | max | att
    washout: int = 0


class AttentionPool(nn.Module):
    """Attention pooling over temporal states. 对时间序列隐藏状态执行注意力汇聚。

    Attributes:
        q (torch.nn.Parameter): Trainable query vector for attention weights. 可训练的查询向量，用于计算注意力权重。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Pool hidden states via attention weights. 通过注意力权重汇聚隐藏状态。

        Args:
            H (torch.Tensor): Hidden trajectory ``(T, hidden_dim)``. 每个时间步的隐藏状态序列。

        Returns:
            torch.Tensor: Aggregated state vector ``(hidden_dim,)``. 汇聚后的隐藏状态向量。
        """
        # calcuate the attention score
        # H: the hidden state of each time step
        # NOTE: IDK how it works
        score = (H * self.q).sum(dim=1)  # [T]
        w = torch.softmax(score, dim=0).unsqueeze(1)  # [T, 1]
        return (H * w).sum(dim=0)  # [H]


class GroupedReservoir(nn.Module):
    """Collection of ESN groups with shared pooling rules. 多个 ESN 分组及其汇聚策略的集合。

    Attributes:
        groups (List[GroupConfig]): Definitions for each reservoir group. 每个储层分组的配置。
        device (Optional[torch.device]): Device hosting the reservoir modules. 储层模块所在设备。
        group_esns (nn.ModuleList): Reservoir cells organised per group. 每组对应的 ESN 单元列表。
        att_pools (nn.ModuleList): Temporal pooling modules aligned to groups. 与分组对应的时间汇聚模块。
    """

    def __init__(
        self, groups: List[GroupConfig], device: Optional[torch.device] = None
    ):
        """Construct grouped reservoirs from configurations. 根据配置创建分组储层。

        Args:
            groups (List[GroupConfig]): Reservoir group definitions. 储层分组配置列表。
            device (Optional[torch.device]): Target device for modules. 模块放置的设备。
        """
        super().__init__()
        self.groups: List[GroupConfig] = groups
        self.device = device
        # the groups of ESNCell
        self.group_esns = nn.ModuleList()  # 存放 ESNCell 实例
        # the attention pools for each group
        self.att_pools = nn.ModuleList()  # 存放 AttentionPool 实例

        for g in groups:
            esns = nn.ModuleList([ESNCell(cfg, device=device) for cfg in g.esn_cfgs])
            self.group_esns.append(esns)
            # if any of ESN need attention pool, then prepare the first attention
            if g.time_pool == "att":
                self.att_pools.append(
                    AttentionPool(g.esn_cfgs[0].hidden_dim).to(device)
                )
            else:
                self.att_pools.append(nn.Identity())

    @torch.no_grad()
    def forward_features(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate fused features from encoded channels. 从编码后的通道生成融合特征。

        Args:
            encoded (Dict[str, torch.Tensor]): Mapping from channel name to tensor ``(T, D)``. 通道名称到 ``(T, D)`` 张量的映射。

        Returns:
            torch.Tensor: Concatenated feature vector ``(F,)`` across groups. 跨分组拼接后的特征向量 ``(F,)``。
        """
        feats = []
        for g, esns, att in zip(self.groups, self.group_esns, self.att_pools):
            # 组内把对应channels对齐并拼D
            U_list = []
            for ch in g.channels:
                if ch not in encoded:
                    raise ValueError(
                        f"Channel '{ch}' not provided for group '{g.name}'"
                    )
                U_list.append(encoded[ch])
            L = min(u.shape[0] for u in U_list)
            U = torch.cat([u[:L] for u in U_list], dim=1)  # [T, sumD]

            # 多个ESN分别过一遍
            states_vecs = []
            # WARNING: module esn is not iterable
            for esn in esns:  # pyright: ignore[reportGeneralTypeIssues]
                H = esn(U)  # [T, H]
                H_use = H[g.washout :] if g.washout > 0 else H

                if g.time_pool == "last":
                    s = H_use[-1]
                elif g.time_pool == "mean":
                    s = H_use.mean(dim=0)
                elif g.time_pool == "max":
                    s, _ = H_use.max(dim=0)
                elif g.time_pool == "att":
                    s = att(H_use)
                else:
                    raise ValueError("Unknown time_pool")
                states_vecs.append(s)

            if g.fuse == "concat":
                gfeat = torch.cat(states_vecs, dim=0)
            elif g.fuse == "mean":
                gfeat = torch.stack(states_vecs, dim=0).mean(dim=0)
            elif g.fuse == "last":
                gfeat = states_vecs[-1]
            else:
                raise ValueError("Unknown fuse")
            feats.append(gfeat)

        return torch.cat(feats, dim=0)  # [F]
