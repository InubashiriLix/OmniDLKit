# Copyright (c) 2025 InubashiriLix Author. All Rights Reserved.

import math
from typing import Optional, Literal


import torch
import torch.nn as nn
import torch.nn.functional as F


_ActName = Optional[Literal["relu", "gelu", "silu", "tanh", "sigmoid", "none"]]
_NormName = Optional[Literal["ln", "bn", "none"]]


def _get_activation(name: _ActName) -> nn.Module:
    if not isinstance(name, str) and name is not None:
        raise TypeError("Expected _ActName string or None.")
    name = (name or "none").lower()  # pyright: ignore[reportAssignmentType]
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.Identity()  # "none" / None


def _init_linear(linear: nn.Linear, activation: _ActName) -> None:
    if activation in ("relu", "silu"):
        nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
    elif activation in ("gelu",):
        nn.init.xavier_normal_(linear.weight, gain=math.sqrt(2.0))
    else:
        nn.init.xavier_normal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class ANNALayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: _ActName = "gelu",
        norm: _NormName = "ln",
        dropout: float = 0.0,
        residual: bool = False,
        pre_norm: bool = False,
        bias: bool = True,
        residual_scale: float = 1.0,
    ):
        """the constructor function

        Args:
            `in_dim`: int, input dim
            `out_dim`: int, output dim
            `activation`: _ActName, the activation function name, supporting list
                `"relu"`, `"gelu"`, `"silu"`, `"tanh"`, `"sigmoid"`, "`none`" for none -> we use torch.identity in code
            `norm`: _NormName, the normalization layer name, supporting list:
                `ln`: layer norm, `bn`: batch norm, `none`: no normalization
            `dropout`: float, the dropout rate, default is 0.0
            `residual`: bool, whether to use residual connection, default is False
            `pre_norm`: bool, whether to apply normalization before the main layer, default is False
            `bias`: bool, whether to use bias in the linear layer, default is True
            `residual_scale`: the scale for the residual connection, default is 1.0

        Raises:
            TypeError: when the norm name is not str or none
        """
        super(ANNALayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.preact = pre_norm
        self.use_res = residual
        self.res_scale = float(residual_scale)

        # the main linear layer
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        _init_linear(self.fc, activation)

        self.act = _get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 归一化
        norm = (norm or "none").lower()  # pyright: ignore[reportAssignmentType]
        if not isinstance(norm, str):
            raise TypeError("Expected _NormName string or None.")
        if norm == "ln":
            # LayerNorm 直接对最后一维做归一化，可作用于任意形状
            self.norm_type = "ln"
            self.norm = nn.LayerNorm(in_dim if pre_norm else out_dim)
        elif norm == "bn":
            # BatchNorm1d 需要 (N,C) 或 (N,C,L)
            self.norm_type = "bn"
            ch = in_dim if pre_norm else out_dim
            self.norm = nn.BatchNorm1d(ch)
        else:
            self.norm_type = "none"
            self.norm = nn.Identity()

        # the residual projection
        if self.use_res and in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
            nn.init.xavier_normal_(self.proj.weight)
        else:
            self.proj = None

    @staticmethod
    def _apply_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        """Apply linear layer to the input tensor, supporting both 2D and 3D inputs
        Args:
            x: the input tensor
            linear: the linear layerdescription]
        Returns:
            torch.Tensor: the output tensor after applying the linear layer
        """
        if x.ndim == 2:
            return linear(x)
        # (N, L, C) -> (-1, C) -> (N, L, out)
        n, l, c = x.shape[0], x.shape[1], x.shape[2]
        y = linear(x.reshape(n * l, c))
        return y.reshape(n, l, -1)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """apply noramlization to the input tensor
        Args:
            `x`: torch.Tensor the input tensor
        Returns:
            the output tensor after applying normalization
        Raises:
            ValueError: when the input dim is not supported for batch norm
        """
        # if the current norm is "none" -> Identity, return x directly
        if isinstance(self.norm, nn.Identity):
            return x

        if self.norm_type == "bn":
            # supporting the datalike (N, C) or (N, L, C)
            if x.ndim == 2:
                return self.norm(x)
            elif x.ndim == 3:
                x_perm = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
                y = self.norm(x_perm)
                return y.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
            raise ValueError(
                "Batch Norm currently only suppports input dim == 2 / 3 or (N, L, C) or (N, C)"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """the forward function, will be automatically called when using model(input)
        Args:
            x: the input tensor
        Returns:
            updated tensor after applying the ANN layer
        """
        res = x

        if self.preact:  # Norm -> Linear -> Act -> Dropout
            x = self._apply_norm(x)
            x = self._apply_linear(x, self.fc)
            x = self.act(x)
            x = self.dropout(x)
        else:  #
            x = self._apply_linear(x, self.fc)
            x = self._apply_norm(x)
            x = self.act(x)
            x = self.dropout(x)
        if self.use_res:
            if self.proj is not None:
                res = self._apply_linear(res, self.proj)
            x = res + self.res_scale * x
        return x


if __name__ == "__main__":
    import torchvision
    import torchvision.datasets as datasets

    def make_mnist_dataloader(batch_size: int = 32):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        return train_loader

    class Modle(nn.Module):
        def __init__(self):
            super(Modle, self).__init__()
            self.ann1 = ANNALayer(
                in_dim=28 * 28,
                out_dim=512,
                activation="gelu",
                norm="ln",
                dropout=0.1,
                residual=True,
                pre_norm=True,
            )
            self.ann2 = ANNALayer(
                in_dim=512,
                out_dim=1024,
                activation="relu",
                norm="bn",
                dropout=0.1,
                residual=True,
                pre_norm=True,
            )

            self.fc = nn.Linear(1024, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.ann1(x)
            x = self.ann2(x)
            x = self.fc(x)
            return x

    model = Modle()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data = make_mnist_dataloader(batch_size=64)

    def train_epochs(epochs: int, dataloader):
        for epoch in range(epochs):
            epoch_loss = 0
            for images, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")

    train_epochs(10, data)
