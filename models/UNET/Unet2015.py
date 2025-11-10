# unet2015_blocks.py
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def crop_like_fast(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    根据参考特征图对源特征图做中心裁剪或截取，以便让 skip 连接对齐。

    Args:
        `src` (torch.Tensor): 形状为 ``[N, C, H_s, W_s]`` 的源特征图。
        `ref` (torch.Tensor): 形状为 ``[N, C, H_r, W_r]`` 的参考特征图。

    Returns:
        torch.Tensor: 与 ``ref`` 在空间尺度一致的裁剪结果，仍共享 ``src`` 的内存视图。
    """
    _, _, Hs, Ws = src.shape
    _, _, Hr, Wr = ref.shape
    y0 = (Hs - Hr) // 2 if Hs >= Hr else 0
    x0 = (Ws - Wr) // 2 if Ws >= Wr else 0
    y1 = y0 + min(Hr, Hs)
    x1 = x0 + min(Wr, Ws)
    return src[:, :, y0:y1, x0:x1]  # view，不额外占显存


class EncBlockV(nn.Module):
    """
    编码端基础块，遵循 valid 卷积语义：
      ``[可选 MaxPool] -> Conv3x3 + ReLU -> Conv3x3 + ReLU``。

    Attributes:
        `pool` (nn.Module): 可选的 2×2 最大池化层或恒等映射。
        `body` (nn.Sequential): 两层卷积-激活叠加，用于提取特征。
    """

    def __init__(self, in_ch: int, out_ch: int, head_max_pool: bool = True):
        """
        初始化编码块。

        Args:
            `in_ch` (int): 输入特征通道数。
            `out_ch` (int): 输出特征通道数。
            `head_max_pool` (bool, optional): 是否在块首添加 2×2 最大池化。默认为 ``True``。
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2) if head_max_pool else nn.Identity()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行编码块前向传播。

        Args:
            `x` (torch.Tensor): 输入特征，形状为 ``[N, in_ch, H, W]``。

        Returns:
            torch.Tensor: 经过可选池化和两层卷积后的特征。
        """
        x = self.pool(x)
        return self.body(x)


class BtlneckBlockV(nn.Module):
    """
    U-Net 瓶颈块，采用 valid 卷积并在末尾进行两倍上采样：
      ``MaxPool -> Conv3x3×2 -> Upsample×2 + Conv1x1``。

    Attributes:
        `pool` (nn.MaxPool2d): 将输入尺度减半的池化层。
        `conv` (nn.Sequential): 两层 3×3 valid 卷积堆叠。
        `up` (nn.Sequential): 最近邻上采样与 1×1 卷积，用于还原通道数。
    """

    def __init__(self, in_ch: int, mid_ch: int):
        """
        构建瓶颈块。

        Args:
            `in_ch` (int): 输入通道数，同时也是上采样后输出的通道数。
            `mid_ch` (int): 中间卷积层使用的通道数。
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行瓶颈块前向传播。

        Args:
            `x` (torch.Tensor): 输入特征，形状 ``[N, in_ch, H, W]``。

        Returns:
            torch.Tensor: 先压缩后上采样得到的特征，通道数恢复到 ``in_ch``。
        """
        x = self.pool(x)
        x = self.conv(x)
        x = self.up(x)  # 通道回到 in_ch；空间×2，但仍会比 skip 小—解码时用 crop 对齐
        return x


class DecBlockV(nn.Module):
    """
    解码端基础块，处理 concat 后的特征并可选执行上采样：
      ``Conv3x3 + ReLU -> Conv3x3 + ReLU -> [可选] Upsample×2 + Conv1x1``。

    Attributes:
        `tail_up` (bool): 标识是否需要末尾上采样。
        `body` (nn.Sequential): 两层 3×3 valid 卷积堆叠。
        `tail` (nn.Module): 上采样 + 1×1 卷积或仅 1×1 卷积。
    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, tail_up: bool = True):
        """
        初始化解码块。

        Args:
            `in_ch` (int): concat 后输入的通道数。
            `mid_ch` (int): 中间卷积通道数。
            `out_ch` (int): 输出通道数；若 ``tail_up`` 为真，上采样后再通过 1×1 卷积得到该通道数。
            `tail_up` (bool, optional): 是否在末尾执行最近邻上采样。默认为 ``True``。
        """
        super().__init__()
        self.tail_up = tail_up
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )
        if tail_up:
            self.tail = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True),
            )
        else:
            self.tail = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行解码块前向传播。

        Args:
            `x` (torch.Tensor): 输入特征，形状 ``[N, in_ch, H, W]``。

        Returns:
            torch.Tensor: 解码块输出，若 ``tail_up`` 为真其空间尺寸会扩大两倍。
        """
        x = self.body(x)
        x = self.tail(x)
        return x


class EncPartV(nn.Module):
    """
    U-Net 编码分支，默认包含 4 层 valid 卷积块，通道数依次翻倍。

    Attributes:
        `enc_layers` (nn.ModuleList): 顺序堆叠的编码块。
        `_btl_in` (int): 编码末端输出的通道数，亦是瓶颈输入通道。
    """

    def __init__(self, in_ch: int, layers: int = 4):
        """
        构建编码分支。

        Args:
            `in_ch` (int): 输入图像的通道数。
            `layers` (int, optional): 编码层数，至少为 1。默认为 ``4``。
        """
        super().__init__()
        ch = 64
        blocks: List[nn.Module] = [EncBlockV(in_ch, ch, head_max_pool=False)]
        for _ in range(1, layers):
            blocks.append(EncBlockV(ch, ch * 2, head_max_pool=True))
            ch *= 2
        self.enc_layers = nn.ModuleList(blocks)
        self._btl_in = ch  # 最深层通道

    def get_btlneck_in_chan(self) -> int:
        """
        获取瓶颈期望的输入通道数。

        Returns:
            int: 最深层编码输出通道数。
        """
        return self._btl_in

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        逐层编码并收集 skip 连接。

        Args:
            `x` (torch.Tensor): 输入图像或特征，形状 ``[N, in_ch, H, W]``。

        Returns:
            List[torch.Tensor]: 从深到浅的特征列表 ``[x_L, …, x_1]``，便于解码阶段逐层使用。
        """
        feats: List[torch.Tensor] = []
        for m in self.enc_layers:
            x = m(x)
            feats.insert(0, x)  # [x4, x3, x2, x1]
        return feats


class BtlneckPartV(nn.Module):
    """
    瓶颈部分，负责连接编码与解码，内部复用 :class:`BtlneckBlockV`。

    Attributes:
        `block` (BtlneckBlockV): 具体的瓶颈块实例。
        `_dec_in` (int): 解码阶段首层 concat 后的通道数。
    """

    def __init__(self, in_ch: int):
        """
        构建瓶颈部分。

        Args:
            `in_ch` (int): 输入通道数，通常等于编码末端的通道数。
        """
        super().__init__()
        self.block = BtlneckBlockV(in_ch, mid_ch=in_ch * 2)
        self._dec_in = in_ch * 2  # 解码首层的 concat 通道（z: in_ch + skip: in_ch）

    def get_dec_in_chan(self) -> int:
        """
        获取解码端第一层的输入通道数。

        Returns:
            int: concat 后的通道数（瓶颈输出 + 最深层 skip）。
        """
        return self._dec_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行瓶颈前向计算。

        Args:
            `x` (torch.Tensor): 最深层编码特征 ``[N, C, H, W]``。

        Returns:
            torch.Tensor: 上采样后的瓶颈输出，通道数为 ``C``。
        """
        return self.block(x)


class DecPartV(nn.Module):
    """
    解码分支，逐层与 skip 特征对齐、拼接，再通过 :class:`DecBlockV` 进行上采样。

    Attributes:
        `dec_layers` (nn.ModuleList): 由若干解码块构成的顺序模块列表。
    """

    def __init__(self, in_ch_concat: int, out_classes: int, layers: int = 4):
        """
        构建解码分支。

        Args:
            `in_ch_concat` (int): 第一层 concat 后的通道数（通常为 ``2C``）。
            `out_classes` (int): 输出类别或最终通道数。
            `layers` (int, optional): 解码层数，应与编码层数一致。默认为 ``4``。
        """
        super().__init__()
        blocks: List[nn.Module] = []
        ch = in_ch_concat  # 2C
        for _ in range(layers - 1):
            blocks.append(
                DecBlockV(in_ch=ch, mid_ch=ch // 2, out_ch=ch // 4, tail_up=True)
            )
            ch //= 2
        blocks.append(
            DecBlockV(in_ch=ch, mid_ch=ch // 2, out_ch=out_classes, tail_up=False)
        )
        self.dec_layers = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        结合 skip 特征执行解码。

        Args:
            `x` (torch.Tensor): 瓶颈输出特征，形状 ``[N, C, H, W]``。
            `skips` (List[torch.Tensor]): 从深到浅排列的 skip 特征列表。

        Returns:
            torch.Tensor: 解码后的语义图或 logits。
        """
        for i, m in enumerate(self.dec_layers):
            # 把 skip 裁剪到 “当前 x” 的大小再拼
            s = crop_like_fast(skips[i], x)
            x = torch.cat([x, s], dim=1)
            x = m(x)
        return x


class Unet2015(nn.Module):
    """
    经典 U-Net（2015）结构，采用 valid 卷积与显式裁剪的 skip 连接实现。

    Attributes:
        `enc` (EncPartV): 编码子网络。
        `btl` (BtlneckPartV): 瓶颈子网络。
        `dec` (DecPartV): 解码子网络。
    """

    def __init__(
        self,
        pic_size: Tuple[int, int],
        in_channels: int = 1,
        num_classes: int = 2,
        layers: int = 4,
    ):
        """
        初始化 U-Net 结构。

        Args:
            `pic_size` (Tuple[int, int]): 训练或推理时的输入图像尺寸 ``(H, W)``，用于记录但不强制检查。
            `in_channels` (int, optional): 输入图像通道数。默认为 ``1``。
            `num_classes` (int, optional): 输出类别或通道数。默认为 ``2``。
            `layers` (int, optional): 编解码层数。默认为 ``4``。
        """
        super().__init__()
        self.pic_size = pic_size
        self.enc = EncPartV(in_ch=in_channels, layers=layers)
        self.btl = BtlneckPartV(self.enc.get_btlneck_in_chan())
        self.dec = DecPartV(
            in_ch_concat=self.btl.get_dec_in_chan(),
            out_classes=num_classes,
            layers=layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行 U-Net 的端到端前向传播。

        Args:
            `x` (torch.Tensor): 输入图像张量，形状 ``[N, in_channels, H, W]``。

        Returns:
            torch.Tensor: 输出 logits 或特征图，通道数为 ``num_classes``。
        """
        skips = self.enc(x)  # [x4, x3, x2, x1]
        z = self.btl(skips[0])  # bottleneck 上采样输出（通道 C）
        y = self.dec(z, skips)  # 解码（每层先 crop skip 再 concat）
        return y


def enable_runtime_optim(
    model: nn.Module, use_channels_last: bool = True, try_compile: bool = True
) -> nn.Module:
    """
    在推理或训练前对模型施加简易运行期优化，例如 ``channels_last`` 和 ``torch.compile``。

    Args:
        model (nn.Module): 需要优化的模型。
        use_channels_last (bool, optional): 是否切换为 ``channels_last`` 内存格式。默认为 ``True``。
        try_compile (bool, optional): 是否尝试调用 ``torch.compile`` 以获得图级优化。默认为 ``True``。

    Returns:
        nn.Module: 经过优化后可直接使用的模型实例。
    """
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    if try_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")  # pyright: ignore[reportAssignmentType]
        except Exception:
            pass
    return model


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    H, W = 572, 572  # 经典 U-Net demo 尺寸
    x = torch.randn(2, 1, H, W, device="cuda" if torch.cuda.is_available() else "cpu")

    net = Unet2015(pic_size=(H, W), in_channels=1, num_classes=2).to(x.device)
    net = enable_runtime_optim(net, use_channels_last=True, try_compile=True)

    with (
        torch.inference_mode(),
        torch.cuda.amp.autocast(enabled=x.is_cuda, dtype=torch.float16),
    ):
        y = net(x.to(memory_format=torch.channels_last))
    print("in :", x.shape, x.dtype, x.device)
    print("out:", y.shape, y.dtype, next(net.parameters()).device)
