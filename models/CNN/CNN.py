from typing import List, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_activation_factory  # 你现有项目里的工厂函数


# ----------------------- 配置 -----------------------
@dataclass
class LayerConfig:
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    activation: str = "gelu"
    norm: str | None = (
        "ln"  # "ln" = GroupNorm(1,C) 作为 LN-Channel；"bn" = BatchNorm2d；None/"none" = 无
    )
    dropout: float = 0.0
    residual: bool = False
    pre_norm: bool = False  # True: Norm -> Conv -> Act；False: Conv -> Norm -> Act
    bias: bool = True
    residual_scale: float = 1.0  # 残差缩放（稳定训练时可<1）


@dataclass
class CNNConfig:
    layer_confs: List[LayerConfig]


# ----------------------- 校验 -----------------------
def _check_cnn_config(cfg: CNNConfig, input_data_channel_len: int):
    assert len(cfg.layer_confs) > 0, "cfg.layer_confs 不能为空"
    assert cfg.layer_confs[0].in_channels == input_data_channel_len, (
        "input_channel_len 与首层 in_channels 不一致，请检查数据或配置"
    )
    last_c = cfg.layer_confs[0].out_channels
    for i, layer_conf in enumerate(cfg.layer_confs[1:], start=1):
        assert layer_conf.in_channels == last_c, (
            f"第 {i} 个层的 in_channels({layer_conf.in_channels}) "
            f"与前一层 out_channels({last_c}) 不一致"
        )
        last_c = layer_conf.out_channels


# ----------------------- 组件 -----------------------
class _Identity(nn.Module):
    def forward(self, x):
        return x


def _make_norm2d(name: Optional[str], num_channels: int) -> nn.Module:
    """
    "ln": 用 GroupNorm(1, C) 实现“LayerNorm-Channel”（对 NCHW 稳定且与 H/W 无关）
    "bn": BatchNorm2d
    None/"none": 不使用归一化
    """
    if name is None:
        return _Identity()
    name = name.lower()
    if name == "ln":
        return nn.GroupNorm(1, num_channels)
    if name == "bn":
        return nn.BatchNorm2d(num_channels)
    return _Identity()


def _init_conv_weight(conv: nn.Conv2d, act_name: Optional[str]):
    if act_name is not None:
        a = act_name.lower()
    else:
        a = "none"
    if a in ("relu", "silu"):
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    elif a in ("gelu",):
        nn.init.xavier_normal_(conv.weight, gain=2.0**0.5)
    else:
        nn.init.xavier_normal_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class ResidualConvBlock(nn.Module):
    """
    按 LayerConfig 构建的卷积 Block：
      - pre-norm:  Norm(in) -> Conv -> Act -> Dropout -> (+ Residual)
      - post-norm: Conv -> Norm(out) -> Act -> Dropout -> (+ Residual)
    当 residual=True 且 (通道不等或 stride>1) 时，残差分支使用 1×1 Conv(stride 同主分支) 做投影。
    输入/输出均为 NCHW。
    """

    def __init__(self, cfg: LayerConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = cfg.pre_norm
        self.use_res = cfg.residual
        self.res_scale = float(cfg.residual_scale)

        # 主卷积
        self.conv = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            bias=cfg.bias,
        )
        _init_conv_weight(self.conv, cfg.activation)

        # 归一化（pre 使用 in_channels；post 使用 out_channels）
        ch_for_norm = cfg.in_channels if cfg.pre_norm else cfg.out_channels
        self.norm = _make_norm2d(cfg.norm, ch_for_norm)

        # 激活（使用你的工厂函数；既兼容返回 nn.Module，也兼容返回可调用的 factory）
        act_factory = get_activation_factory(cfg.activation)
        self.act = act_factory() if callable(act_factory) else act_factory

        # Dropout
        self.drop = (
            nn.Dropout2d(cfg.dropout)
            if (cfg.dropout and cfg.dropout > 0)
            else _Identity()
        )

        # 残差投影
        need_proj = self.use_res and (
            cfg.in_channels != cfg.out_channels or cfg.stride != 1
        )
        if need_proj:
            self.proj = nn.Conv2d(
                cfg.in_channels,
                cfg.out_channels,
                kernel_size=1,
                stride=cfg.stride,
                bias=False,
            )
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        if self.pre_norm:
            x = self.norm(x)
            x = self.conv(x)
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


# ----------------------- 组网 -----------------------
def makeCnnPipeline(cfg: CNNConfig, input_data_channel_len: int) -> nn.Module:
    """
    根据 CNNConfig 生成 nn.Sequential，由若干 ResidualConvBlock 顺序堆叠。
    """
    _check_cnn_config(cfg=cfg, input_data_channel_len=input_data_channel_len)
    blocks: List[nn.Module] = [ResidualConvBlock(lc) for lc in cfg.layer_confs]
    return nn.Sequential(*blocks)


if __name__ == "__main__":
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    def make_mnist_dataloader(batch_size: int = 64) -> DataLoader:
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    def build_mnist_model() -> nn.Module:
        cfg = CNNConfig(
            layer_confs=[
                LayerConfig(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation="relu",
                    norm="bn",
                ),
                LayerConfig(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation="relu",
                    norm="bn",
                    residual=True,
                ),
                LayerConfig(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation="relu",
                    norm="bn",
                ),
            ]
        )
        backbone = makeCnnPipeline(cfg=cfg, input_data_channel_len=1)
        num_classes = 10
        head_in_features = cfg.layer_confs[-1].out_channels
        return nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_in_features, num_classes),
        )

    def train_epochs(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        epochs: int = 2,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)

            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mnist_model().to(device)
    train_loader = make_mnist_dataloader()
    train_epochs(model, train_loader, device, epochs=10)
