from multiprocessing.spawn import _main
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

_ActName = Optional[Literal["relu", "gelu", "silu", "tanh", "sigmoid", "none"]]
_NormName = Optional[Literal["ln", "bn", "none"]]


def _get_activation(name: _ActName) -> nn.Module:
    name = (name or "none").lower()  # pyright: ignore[reportAssignmentType]
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.Identity()  # "none" / None


def _init_conv(conv: nn.Conv2d, act: _ActName):
    if act in ("relu", "silu"):
        init.kaiming_normal_(conv.weight, nonlinearity="relu")
    elif act in ("gelu",):
        init.xavier_normal_(conv.weight, gain=2.0**0.5)
    else:
        init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        init.zeros_(conv.bias)


class ANNConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
            `in_channels`: int, input dim
            `out_channels`: int, output dim
            `kernel_size`: int, kernel size
            `stride`: int, stride
            `padding`: int, padding
            `activation`: _ActName, activation function
                `"relu"`, `"gelu"`, `"silu"`, `"tanh"`, `"sigmoid"`, "`none`" for none -> we use torch.identity in code
            `norm`: _NormName, normalization layer, including
                `ln`: layer norm, `bn`: batch norm, `none`: no normalization
            `dropout`: float, dropout rate
            `residual`: bool, whether to use residual connections
            `pre_norm`: bool, whether to apply normalization before the main operation
            `bias`: bool, whether to use bias in the convolution
            `residual_scale`: float, scaling factor for the residual connection
        """
        super(ANNConvLayer, self).__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.preact = pre_norm
        self.use_res = residual
        self.res_scale = float(residual_scale)

        # the main convolutional layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        _init_conv(self.conv, activation)

        self.act = _get_activation(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        # normalization
        norm = (norm or "none").lower()  # pyright: ignore[reportAssignmentType]
        if norm == "ln":
            self.norm_type = "gn-ln"
            ch = in_channels if pre_norm else out_channels
            self.norm = nn.GroupNorm(1, ch)
        elif norm == "bn":
            self.norm_type = "bn"
            ch = in_channels if pre_norm else out_channels
            self.norm = nn.BatchNorm2d(ch)
        else:
            self.norm_type = "none"
            self.norm = nn.Identity()

        need_proj = residual and (in_channels != out_channels or stride != 1)
        if need_proj:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=bias)
            _init_conv(self.proj, activation)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self.preact:
            x = self.norm(x)
            x = self.conv(x)
            x = self.act(x)
            x = self.drop(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)

        if self.use_res:
            if self.proj is not None:
                res = self.proj(res)
            x = res + self.res_scale * x

        return x


if __name__ == "__main__":
    import torchvision
    import torchvision.datasets as datasets
    import torch.optim as optim

    def make_mnist_dataloader(batch_size: int = 32):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        return train_loader

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.ann1 = ANNConvLayer(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                activation="gelu",
                norm="ln",
                dropout=0.1,
                residual=True,
                pre_norm=True,
            )

            self.ann2 = ANNConvLayer(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                activation="relu",
                norm="bn",
                dropout=0.1,
                residual=True,
                pre_norm=True,
            )

            self.fc = nn.Linear(64 * 14 * 14, 10)

        def forward(self, x):
            x = self.ann1(x)
            x = self.ann2(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def train_epochs():
        train_loader = make_mnist_dataloader()
        model = Model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            total_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                with torch.no_grad():
                    total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/10], Loss: {total_loss}")

        # Save the trained model
        torch.save(model.state_dict(), "model.pth")
        print("Model saved as model.pth")

    train_epochs()
