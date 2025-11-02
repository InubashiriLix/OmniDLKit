from typing import Tuple
import torch
import torch.nn as nn


# Ridge Regression Readout (Closed-form Solution)
@torch.no_grad()
def ridge_readout_fit(
    H: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3, add_bias: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve ridge regression readout weights. 求解岭回归读出权重。

    Args:
        H (torch.Tensor): Feature matrix ``(N, F)`` extracted from reservoir states. 储层特征矩阵 ``(N, F)``。
        Y (torch.Tensor): Target matrix ``(N, C)`` for supervised fitting. 监督学习目标矩阵 ``(N, C)``。
        lam (float): L2 regularisation strength ``λ``. L2 正则化系数 ``λ``。
        add_bias (bool): Whether to append a bias column during fitting. 是否在拟合时添加偏置列。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Weight matrix ``(F, C)`` and bias vector ``(C,)``. 返回权重矩阵 ``(F, C)`` 与偏置向量 ``(C,)``。
    """
    if add_bias:
        Hb = torch.cat(
            [H, torch.ones(H.shape[0], 1, device=H.device)], dim=1
        )  # [N, F+1]
    else:
        Hb = H  # [N, F]

    I = torch.eye(Hb.shape[1], device=H.device)  # [F+1, F+1] or [F, F]
    Wfull = torch.linalg.solve(Hb.T @ Hb + lam * I, Hb.T @ Y)  # [F+1, C]
    if add_bias:
        W, b = Wfull[:-1, :], Wfull[-1:, :].view(-1)
    else:
        W, b = Wfull, torch.zeros(Y.shape[1], device=H.device)
    return W, b


@torch.no_grad()
def ridge_readout_predict(
    H: torch.Tensor, W: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Apply linear readout to features. 将线性读出应用于特征。

    Args:
        H (torch.Tensor): Feature matrix ``(N, F)``. 特征矩阵 ``(N, F)``。
        W (torch.Tensor): Weight matrix from ridge regression ``(F, C)``. 岭回归得到的权重矩阵 ``(F, C)``。
        b (torch.Tensor): Bias vector ``(C,)``. 偏置向量 ``(C,)``。

    Returns:
        torch.Tensor: Prediction matrix ``(N, C)``. 预测结果矩阵 ``(N, C)``。
    """

    return H @ W + b
