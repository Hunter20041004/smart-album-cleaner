"""
MobileNetV2 遷移學習訓練 — Good / Bad 二元表情分類器。

預期資料結構（前處理腳本 prepare_dataset.py 的輸出）：
    datasets/processed/Good/*.jpg
    datasets/processed/Bad/*.jpg

訓練策略：
    - 載入 ImageNet 預訓練 MobileNetV2
    - 凍結所有特徵層（features.* 的 requires_grad = False）
    - classifier 換成 Linear(in_features → 2)
    - Train/Val 自動 80/20 隨機切分（固定 seed 確保可重現）
    - 訓練增強：水平翻轉 / ±10° 旋轉 / 亮度微調
    - 驗證：僅做 resize + normalize
    - 紀錄最佳 val accuracy 的權重，存到 models/mobilenet_face.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    MobileNet_V2_Weights, mobilenet_v2,
    MobileNet_V3_Large_Weights, mobilenet_v3_large,
)

NUM_CLASSES = 2
IMG_SIZE = 224
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ARCH_CHOICES = ("mobilenet_v2", "mobilenet_v3_large")
DEFAULT_ARCH = "mobilenet_v3_large"  # 實驗:換更強的 backbone

# V2: 19 個 features blocks  → 解凍最後 3 個 = features[16:]
# V3-Large: 17 個 features blocks → 解凍最後 3 個 = features[14:]
_DEFAULT_UNFREEZE_FROM = {
    "mobilenet_v2": 16,
    "mobilenet_v3_large": 14,
}


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """強化版資料增強：對抗小資料集容易過擬合的問題。"""
    train_tf = transforms.Compose([
        # 隨機裁切 + 縮放：模擬不同人臉框大小，學「以五官為中心」的特徵
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # 色彩擾動：模擬不同光線、白平衡、膚色差異
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # 隨機遮擋：模擬手、頭髮、配件遮住部分臉
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.5, 2.0)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_model(arch: str = DEFAULT_ARCH) -> nn.Module:
    """載入 ImageNet 預訓 backbone,凍結 features,把 classifier 換成 2 類 head。"""
    if arch == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
        in_features = 1280  # V2 features 池化後維度
    elif arch == "mobilenet_v3_large":
        # V2 較新更強、含 SE attention,但參數更多有過擬合風險
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=weights)
        in_features = 960  # V3-Large features 池化後維度
    else:
        raise ValueError(f"未知架構:{arch} (允許 {ARCH_CHOICES})")

    for p in model.features.parameters():
        p.requires_grad = False
    # 統一替換 classifier 為 Dropout + 2 類 Linear,讓 V2 / V3 fine-tune 流程對稱
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def make_loaders(
    data_root: Path,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    test_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """切 train / val / test 三段。val 用來選超參數;test 只在最後評估一次。"""
    train_tf, val_tf = build_transforms()
    full_train = ImageFolder(str(data_root), transform=train_tf)
    full_val = ImageFolder(str(data_root), transform=val_tf)
    if full_train.classes != ["Bad", "Good"]:
        print(f"[warn] 類別順序:{full_train.classes}(ImageFolder 依字母排序)")

    n_total = len(full_train)
    if n_total == 0:
        raise RuntimeError(f"{data_root} 中沒有任何圖片,先跑 prepare_dataset.py")
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise RuntimeError(
            f"資料太少:total={n_total} val={n_val} test={n_test} → train={n_train}"
        )

    g = torch.Generator().manual_seed(SEED)
    train_idx, val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test], generator=g,
    )

    train_set = torch.utils.data.Subset(full_train, list(train_idx))
    val_set = torch.utils.data.Subset(full_val, list(val_idx))
    test_set = torch.utils.data.Subset(full_val, list(test_idx))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader, test_loader, full_train.classes


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_best_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> tuple[float, float]:
    """掃描閾值 0.30~0.70，找出讓 val accuracy 最高的 P(Bad) 切點。

    回傳 (best_threshold, accuracy_at_best_threshold)。
    """
    model.eval()
    bad_idx = classes.index("Bad")
    all_p_bad: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            all_p_bad.append(probs[:, bad_idx])
            all_labels.append(labels.numpy())
    p_bad = np.concatenate(all_p_bad)
    labels = np.concatenate(all_labels)

    best_thr = 0.5
    best_acc = -1.0
    for thr in np.arange(0.30, 0.71, 0.01):
        preds = np.where(p_bad >= thr, bad_idx, 1 - bad_idx)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
) -> tuple[float, float]:
    """回傳 (avg_loss, accuracy)。optimizer=None 表示驗證模式。"""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def train(
    data_root: Path,
    out_path: Path,
    arch: str = DEFAULT_ARCH,
    epochs: int | None = None,
    batch_size: int = 32,
    lr: float | None = None,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    finetune: bool = False,
    unfreeze_from: int | None = None,
) -> None:
    # 不同模式有不同的合理預設
    if finetune:
        epochs = epochs if epochs is not None else 5
        lr = lr if lr is not None else 1e-5
    else:
        epochs = epochs if epochs is not None else 15
        lr = lr if lr is not None else 1e-3
    if unfreeze_from is None:
        unfreeze_from = _DEFAULT_UNFREEZE_FROM[arch]

    torch.manual_seed(SEED)
    device = _pick_device()
    print(f"[info] arch = {arch}  device = {device}  "
          f"mode = {'FINETUNE' if finetune else 'STAGE-1'}")

    train_loader, val_loader, test_loader, classes = make_loaders(
        data_root, batch_size, val_ratio, num_workers,
    )
    print(f"[info] classes = {classes}  "
          f"train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    model = build_model(arch).to(device)
    prior_best = 0.0
    if finetune:
        if not out_path.is_file():
            raise RuntimeError(
                f"--finetune 需要先存在的 checkpoint:{out_path}\n"
                "請先跑一次 stage-1 訓練(不加 --finetune)"
            )
        ckpt = torch.load(out_path, map_location=device, weights_only=False)
        # 防呆:checkpoint 的 arch 跟現在指定的不一致 → 拒絕載入避免錯誤覆寫
        ckpt_arch = ckpt.get("arch", "mobilenet_v2")  # 舊版 checkpoint 沒寫 arch,預設 v2
        if ckpt_arch != arch:
            raise RuntimeError(
                f"checkpoint 的架構是 {ckpt_arch},但現在指定 arch={arch}。\n"
                f"請先跑 stage-1(不加 --finetune)用 {arch} 重新訓練。"
            )
        model.load_state_dict(ckpt["state_dict"])
        prior_best = float(ckpt.get("val_acc", 0.0))
        # 解凍 features[unfreeze_from:]，讓底層特徵也能微調
        for i, block in enumerate(model.features):
            if i >= unfreeze_from:
                for p in block.parameters():
                    p.requires_grad = True
        print(f"[finetune] 載入 checkpoint val_acc={prior_best:.3f}，"
              f"解凍 features[{unfreeze_from}:] (共 {len(model.features) - unfreeze_from} 個 blocks)")

    # label_smoothing 讓模型不過度自信，對小資料集很有幫助
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    # weight_decay 是 L2 正則，壓制過擬合（finetune 模式更需要）
    weight_decay = 1e-4 if finetune else 1e-5
    print(f"[info] trainable params: {n_trainable:,}  lr={lr}  "
          f"epochs={epochs}  weight_decay={weight_decay}")
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # finetune 模式以「贏過 prior_best 才存」為準，不會被前幾個 epoch 的小退步覆寫
    best_acc = prior_best
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, device, criterion, None)
        marker = ""
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "arch": arch,
                "classes": classes,
                "img_size": IMG_SIZE,
                "epoch": epoch,
                "val_acc": va_acc,
            }, out_path)
            marker = "  ⭐ saved"
        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  |  "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}{marker}"
        )

    if best_epoch == -1:
        print(f"\n[done] 這次訓練沒有超過先前最佳 val_acc={prior_best:.3f}，"
              f"checkpoint 保持原狀。")
    else:
        print(f"\n[done] best val_acc={best_acc:.3f} @ epoch {best_epoch} → {out_path}")

    # 用「最佳那次」的權重再做一次決策閾值調校
    best_ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    best_thr, thr_acc = _find_best_threshold(model, val_loader, device, classes)
    print(f"[threshold] 掃描 0.30~0.70 → 最佳 P(Bad) 閾值 = {best_thr:.2f}  "
          f"(此閾值下 val_acc = {thr_acc:.3f})")
    best_ckpt["decision_threshold"] = best_thr
    best_ckpt["threshold_val_acc"] = thr_acc

    # 用 test set 做最終評估(這個分數才是真正的泛化能力)
    _, test_acc = _run_epoch(model, test_loader, device, criterion, None)
    print(f"[test] 獨立 test set acc = {test_acc:.3f}  "
          f"(n={len(test_loader.dataset)},此分數最能反映真實能力)")
    best_ckpt["test_acc"] = test_acc

    torch.save(best_ckpt, out_path)
    print(f"[done] 已將決策閾值 + test_acc 寫入 {out_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MobileNetV2 遷移學習：表情 Good / Bad 二元分類。",
    )
    p.add_argument(
        "--data", type=Path, default=Path("datasets/processed"),
        help="處理後的資料夾，內含 Good/ 與 Bad/（預設 datasets/processed）",
    )
    p.add_argument(
        "--out", type=Path, default=Path("models/mobilenet_face.pth"),
        help="最佳模型輸出路徑（預設 models/mobilenet_face.pth）",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="訓練輪數（stage-1 預設 15、finetune 預設 5）",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--lr", type=float, default=None,
        help="學習率（stage-1 預設 1e-3、finetune 預設 1e-5）",
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--arch", choices=ARCH_CHOICES, default=DEFAULT_ARCH,
        help=f"backbone 架構(預設 {DEFAULT_ARCH})",
    )
    p.add_argument(
        "--finetune", action="store_true",
        help="從現有 checkpoint 繼續訓練,並解凍部分特徵層",
    )
    p.add_argument(
        "--unfreeze-from", type=int, default=None,
        help="finetune 模式時解凍 features[N:] 起的 blocks "
             "(V2 預設 16、V3-Large 預設 14,皆代表「最後 3 個」)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        train(
            args.data, args.out,
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            finetune=args.finetune,
            unfreeze_from=args.unfreeze_from,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
