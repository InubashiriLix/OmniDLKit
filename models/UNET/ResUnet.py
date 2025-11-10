from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    """
    编码块（ResUNet）:
    [可选 MaxPool] -> (3x3 SAME Conv -> BN -> ReLU -> 3x3 SAME Conv -> BN)
    + 残差 (Identity 或 1x1 Conv) -> ReLU
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        head_max_pool: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()
        self.export_config: Dict[str, int] = {
            "input_channel": input_channel,
            "output_channel": output_channel,
            "head_max_pool": int(head_max_pool),
            "use_bn": int(use_bn),
        }

        self.pool = nn.MaxPool2d(2, 2) if head_max_pool else nn.Identity()

        bias = not use_bn
        self.conv1 = nn.Conv2d(
            input_channel,
            output_channel,
            3,
            padding=1,
            padding_mode="reflect",
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(output_channel) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            output_channel,
            output_channel,
            3,
            padding=1,
            padding_mode="reflect",
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(output_channel) if use_bn else nn.Identity()

        self.proj = (
            nn.Conv2d(input_channel, output_channel, 1, bias=True)
            if input_channel != output_channel
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        s = self.proj(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.relu(y + s)  # 残差相加后激活
        return y

    def get_info(self) -> Dict[str, int]:
        return self.export_config


class DecBlock(nn.Module):
    """
    解码块（ResUNet）:
    输入 = concat([上一路特征, 对应skip])  →  (3x3 SAME -> BN -> ReLU -> 3x3 SAME -> BN)
    + 残差(把concat投影到 mid 通道) → ReLU → [可选] 上采样×2（nearest + 1x1 conv 调整通道）
    说明：残差相加发生在 **middle_channel** 维度，保证 y+s 通道一致。
    """

    def __init__(
        self,
        input_channel: int,
        middle_channel: int,
        output_channel: int,
        tail_deconv: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()
        self.tail_deconv = tail_deconv

        bias = not use_bn
        # 主分支：in -> mid -> mid
        self.conv1 = nn.Conv2d(
            input_channel,
            middle_channel,
            3,
            padding=1,
            padding_mode="reflect",
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(middle_channel) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            middle_channel,
            middle_channel,
            3,
            padding=1,
            padding_mode="reflect",
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(middle_channel) if use_bn else nn.Identity()

        # 残差分支：concat 后直接投影到 middle_channel
        self.proj = (
            nn.Conv2d(input_channel, middle_channel, 1, bias=True)
            if input_channel != middle_channel
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

        # 块尾变换：
        #   - 若需要上采样：先上采样×2，再用 1x1 把 mid -> out（不改变空间尺寸）
        #   - 否则：直接 1x1 把 mid -> out（作为输出头）
        if tail_deconv:
            self.tail = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    middle_channel,
                    output_channel,
                    kernel_size=2,
                    stride=1,
                    bias=True,
                    padding="same",
                ),
            )
        else:
            self.tail = nn.Conv2d(
                middle_channel, output_channel, kernel_size=1, bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.proj(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.relu(y + s)  # 残差相加
        y = self.tail(y)  # 上采样或输出投影
        return y


class EncPart(nn.Module):
    def __init__(self, input_channels: int, layers: int = 4):
        super().__init__()
        # 64,128,256,512 ...
        ch = 64
        blocks: List[nn.Module] = []
        # 第一个块不池化
        blocks.append(EncBlock(input_channels, ch, head_max_pool=False))
        for _ in range(1, layers):
            blocks.append(EncBlock(ch, ch * 2, head_max_pool=True))
            ch *= 2
        self.enc_layers = nn.ModuleList(blocks)  # ✅ 注册

        self._btlneck_in = ch

    def get_btlneck_in_chan(self) -> int:
        return self._btlneck_in

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for m in self.enc_layers:
            x = m(x)
            feats.insert(0, x)  # 存成 [x4, x3, x2, x1]
        return feats


class BtlneckPart(nn.Module):
    """
    底部：再做一次 Res 编码块（含池化），然后上采样×2，把通道从 2*C 回到 C。
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.block = EncBlock(in_ch, in_ch * 2, head_max_pool=True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_ch * 2, in_ch, kernel_size=2, bias=True, stride=1, padding="same"
            ),
        )
        self._dec_in = in_ch * 2  # 用于“concat后输入通道”的计算

    def get_dec_in_chan(self) -> int:
        # 解码第一层的输入 = skip(C) + bottleneck上采样后(C) = 2*C
        return self._dec_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)  # -> 2C
        x = self.up(x)  # -> C（空间×2）
        return x


class DecPart(nn.Module):
    """
    解码端：每层都先 concat([x, skip_i])，再进 DecBlock。
    通道规则：假设 encoder 顶层通道为 C，
      第一层输入 2C -> mid=C -> out=C/2 (再上采样)；
      下一层输入  (out + 下一skip) = C/2 + C/2 = C -> mid=C/2 -> out=C/4 ...
    """

    def __init__(self, input_channels: int, output_channel: int, layers: int = 4):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = input_channels  # 2C
        for _ in range(layers - 1):
            blocks.append(
                DecBlock(
                    input_channel=ch,
                    middle_channel=ch // 2,
                    output_channel=ch // 4,
                    tail_deconv=True,
                )
            )
            ch //= (
                2  # 下一次 concat 前，特征通道降为 ch/2，与下一 skip (ch/2) 合并 -> ch
            )

        # 最后一层：不再上采样，直接输出到类别数
        blocks.append(
            DecBlock(
                input_channel=ch,
                middle_channel=ch // 2,
                output_channel=output_channel,
                tail_deconv=False,
            )
        )
        self.dec_layers = nn.ModuleList(blocks)  # ✅ 注册

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i, m in enumerate(self.dec_layers):
            x = torch.cat([x, skips[i]], dim=1)  # concat along channel
            x = m(x)
        return x


class ResUnet(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, layers: int = 4):
        super().__init__()
        self.enc_part = EncPart(input_channels=input_channel, layers=layers)
        self.btlneck_part = BtlneckPart(self.enc_part.get_btlneck_in_chan())
        self.dec_part = DecPart(
            input_channels=self.btlneck_part.get_dec_in_chan(),
            output_channel=output_channel,
            layers=layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.enc_part(x)  # [x4, x3, x2, x1]
        x = self.btlneck_part(skips[0])  # 上采样到与 x4 一样大，通道 C
        x = self.dec_part(x, skips)  # 逐层 concat + 解码
        return x
