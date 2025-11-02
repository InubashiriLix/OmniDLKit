import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable, Callable


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
