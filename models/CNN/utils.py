import torch
import torch.nn as nn


def get_activation(act: str):
    """
    根据字符串名称返回对应的PyTorch 激活函数的工厂实例模块

    ---

    支持的激活函数类型（不区分大小写）：
        - `"relu"`:      `nn.ReLU`
            线性整流单元，公式 `f(x) = max(0, x)`，简单高效，常用于深度网络，但可能出现神经元死亡。
        - `"sigmoid"`:   `nn.Sigmoid`
            S型激活函数，公式 `f(x) = 1 / (1 + exp(-x))`，输出范围 (0, 1)，适合二分类，易梯度消失。
        - `"tanh"`:      `nn.Tanh`
            双曲正切函数，公式 `f(x) = tanh(x)`，输出范围 (-1, 1)，比 Sigmoid 更常用，仍有梯度消失。
        - `"leakyrelu"`: `nn.LeakyReLU`
            带泄漏的 ReLU，公式 `f(x) = x if x > 0 else αx`（α通常为0.01），解决 ReLU 神经元死亡问题。
        - `"elu"`:       `nn.ELU`
            指数线性单元，公式 `f(x) = x if x > 0 else α*(exp(x)-1)`，负区间缓慢增长，提升鲁棒性。
        - `"gelu"`:      `nn.GELU`
            高斯误差线性单元，公式 `f(x) = x * Φ(x)`，Φ(x)为标准正态分布累积分布函数，Transformer常用。
        - `"softplus"`:  `nn.Softplus`
            软正值函数，公式 `f(x) = log(1 + exp(x))`，ReLU 的平滑版本，输出始终为正。
        - `"swish"`:     自定义 `Swish (x * sigmoid(x))`
            Google提出，公式 `f(x) = x * sigmoid(x)`，性能优于 ReLU，具有自门控特性。
    ---

    参数:
        act (str): 激活函数名称

    返回:
        nn.Module: 对应的激活函数模块

    异常:
        ValueError: 如果输入名称不在支持列表中
    """

    act = act.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "leakyrelu":
        return nn.LeakyReLU()
    elif act == "elu":
        return nn.ELU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "softplus":
        return nn.Softplus()
    elif act == "swish":

        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)

        return Swish()
    else:
        raise ValueError(f"Unknown activation: {act}")


def get_activation_factory(act: str):
    """
    根据字符串名称返回对应的PyTorch 激活函数的工厂实例模块

    ---

    支持的激活函数类型（不区分大小写）：
        - `"relu"`:      `nn.ReLU`
            线性整流单元，公式 `f(x) = max(0, x)`，简单高效，常用于深度网络，但可能出现神经元死亡。
        - `"sigmoid"`:   `nn.Sigmoid`
            S型激活函数，公式 `f(x) = 1 / (1 + exp(-x))`，输出范围 (0, 1)，适合二分类，易梯度消失。
        - `"tanh"`:      `nn.Tanh`
            双曲正切函数，公式 `f(x) = tanh(x)`，输出范围 (-1, 1)，比 Sigmoid 更常用，仍有梯度消失。
        - `"leakyrelu"`: `nn.LeakyReLU`
            带泄漏的 ReLU，公式 `f(x) = x if x > 0 else αx`（α通常为0.01），解决 ReLU 神经元死亡问题。
        - `"elu"`:       `nn.ELU`
            指数线性单元，公式 `f(x) = x if x > 0 else α*(exp(x)-1)`，负区间缓慢增长，提升鲁棒性。
        - `"gelu"`:      `nn.GELU`
            高斯误差线性单元，公式 `f(x) = x * Φ(x)`，Φ(x)为标准正态分布累积分布函数，Transformer常用。
        - `"softplus"`:  `nn.Softplus`
            软正值函数，公式 `f(x) = log(1 + exp(x))`，ReLU 的平滑版本，输出始终为正。
        - `"swish"`:     自定义 `Swish (x * sigmoid(x))`
            Google提出，公式 `f(x) = x * sigmoid(x)`，性能优于 ReLU，具有自门控特性。
    ---

    参数:
        act (str): 激活函数名称

    返回:
        nn.Module: 对应的激活函数模块

    异常:
        ValueError: 如果输入名称不在支持列表中
    """

    name = act.lower()
    if name == "relu":
        return lambda: nn.ReLU()
    if name == "sigmoid":
        return lambda: nn.Sigmoid()
    if name == "tanh":
        return lambda: nn.Tanh()
    if name == "leakyrelu":
        return lambda: nn.LeakyReLU()
    if name == "elu":
        return lambda: nn.ELU()
    if name == "gelu":
        return lambda: nn.GELU()
    if name == "softplus":
        return lambda: nn.Softplus()
    if name == "swish":

        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)

        return lambda: Swish()
    raise ValueError(f"Unknown activation: {name}")
