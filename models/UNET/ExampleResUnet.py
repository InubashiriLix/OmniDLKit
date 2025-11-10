import os, random, time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResUnet import ResUnet
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class MNISTSeg(Dataset):
    def __init__(self, root="./data", size=(256, 256), train=True):
        self.ds = datasets.MNIST(
            root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.size = size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, _ = self.ds[idx]  # img: (1, 28, 28), [0,1]
        # 前景=数字：阈值生成 mask，(H,W) 的 Long
        mask = (img[0] > 0.1).to(torch.long)  # 0:背景  1:前景

        # Resize 到 256x256（或者你模型要求的，需能被 2^层数 整除）
        img = F.interpolate(
            img.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
        ).squeeze(0)
        mask = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=self.size,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )

        return img, mask  # img:(1,H,W), mask:(H,W)


# ====== 指标 ======
@torch.no_grad()
def metrics(pred_logits, target):
    """
    pred_logits: (B, 2, H, W)  target: (B, H, W)
    """
    pred = pred_logits.argmax(1)  # (B,H,W)
    correct = (pred == target).sum().item()
    total = target.numel()

    # IoU / Dice（对2类求均值）
    miou, mdice = 0.0, 0.0
    for cls in [0, 1]:
        pred_c = pred == cls
        tgt_c = target == cls
        inter = (pred_c & tgt_c).sum().item()
        union = (pred_c | tgt_c).sum().item()
        iou = inter / (union + 1e-8)
        dice = (2 * inter) / (pred_c.sum().item() + tgt_c.sum().item() + 1e-8)
        miou += iou
        mdice += dice
    miou /= 2.0
    mdice /= 2.0
    return correct / total, miou, mdice


# ====== 训练 ======
@dataclass
class Cfg:
    size = (256, 256)
    batch = 16
    epochs = 2  # 快速验机；想更稳可以加到 5~10
    lr = 1e-3
    num_workers = 2
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


cfg = Cfg()
set_seed(cfg.seed)
print("Device:", cfg.device)

# 数据
full_train = MNISTSeg(train=True, size=cfg.size)
full_test = MNISTSeg(train=False, size=cfg.size)

# 也可以切一部分做快速训练
train_len = min(len(full_train), 8000)  # 8k 样本，小点更快
val_len = min(len(full_test), 1000)
train_ds, _ = random_split(
    full_train,
    [train_len, len(full_train) - train_len],
    generator=torch.Generator().manual_seed(cfg.seed),
)
val_ds, _ = random_split(
    full_test,
    [val_len, len(full_test) - val_len],
    generator=torch.Generator().manual_seed(cfg.seed),
)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

# 模型
model = ResUnet(input_channel=1, output_channel=2, layers=4).to(cfg.device)
# 小优化：channels_last + AMP
model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

best_miou = 0.0
for ep in range(1, cfg.epochs + 1):
    model.train()
    t0 = time.time()
    loss_sum = 0.0
    for img, mask in train_loader:
        img = img.to(cfg.device, memory_format=torch.channels_last, non_blocking=True)
        mask = mask.to(cfg.device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=(cfg.device == "cuda"), dtype=torch.float16
        ):
            logits = model(img)  # (B,2,H,W)
            loss = criterion(logits, mask)  # CE 需要 mask 为 Long
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * img.size(0)
    train_loss = loss_sum / len(train_loader.dataset)  # pyright: ignore[reportArgumentType]

    # 验证
    model.eval()
    acc_sum, miou_sum, mdice_sum, n = 0.0, 0.0, 0.0, 0
    with (
        torch.no_grad(),
        torch.cuda.amp.autocast(enabled=(cfg.device == "cuda"), dtype=torch.float16),
    ):
        for img, mask in val_loader:
            img = img.to(
                cfg.device, memory_format=torch.channels_last, non_blocking=True
            )
            mask = mask.to(cfg.device, non_blocking=True)
            logits = model(img)
            acc, miou, mdice = metrics(logits, mask)
            bs = img.size(0)
            acc_sum += acc * bs
            miou_sum += miou * bs
            mdice_sum += mdice * bs
            n += bs
    val_acc = acc_sum / n
    val_miou = miou_sum / n
    val_dice = mdice_sum / n

    dt = time.time() - t0
    print(
        f"[Epoch {ep}/{cfg.epochs}] "
        f"train_loss={train_loss:.4f}  "
        f"val_acc={val_acc:.4f}  mIoU={val_miou:.4f}  Dice={val_dice:.4f}  "
        f"time={dt:.1f}s"
    )

    best_miou = max(best_miou, val_miou)

print("Done. Best mIoU:", round(best_miou, 4))

# [Epoch 1/2] train_loss=0.0186  val_acc=0.9980  mIoU=0.9932  Dice=0.9966  time=265.2s
