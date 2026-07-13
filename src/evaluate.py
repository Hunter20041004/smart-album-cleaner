"""
獨立評估腳本 — 不需要重訓,直接吃現有 checkpoint。

輸出:
    - Per-class precision / recall / F1
    - Confusion matrix (文字 + PNG)
    - 錯誤案例 top-N(信心最高的誤判)

用法:
    python -m src.evaluate
    python -m src.evaluate --model models/mobilenet_face.pth --data datasets/processed
    python -m src.evaluate --output-dir reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不開視窗,只存檔
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.train_mobilenet import (
    build_model,
    make_loaders,
)

DEFAULT_MODEL = Path("models/mobilenet_face.pth")
DEFAULT_DATA = Path("datasets/processed")
DEFAULT_OUTDIR = Path("reports")


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _gather_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """回傳 (y_true, y_pred, p_bad_per_sample, file_paths)。"""
    model.eval()
    bad_idx = classes.index("Bad")
    y_true: list[int] = []
    p_bad_all: list[float] = []
    paths: list[str] = []

    # 從 dataset 反向找出 file path(透過 Subset.indices → ImageFolder.samples)
    ds = loader.dataset
    base_ds = ds.dataset if hasattr(ds, "dataset") else ds  # Subset → ImageFolder
    indices = ds.indices if hasattr(ds, "indices") else list(range(len(ds)))
    samples = base_ds.samples

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            p_bad_all.extend(probs[:, bad_idx].tolist())
            y_true.extend(labels.numpy().tolist())

    # 填回 paths(順序對齊 indices)
    for idx in indices:
        paths.append(samples[idx][0])

    p_bad_arr = np.array(p_bad_all)
    y_true_arr = np.array(y_true)
    # 用 threshold 決定預測
    y_pred = np.where(p_bad_arr >= threshold, bad_idx, 1 - bad_idx)
    return y_true_arr, y_pred, p_bad_arr, paths


def _plot_confusion_matrix(
    cm: np.ndarray, classes: list[str], out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix (test set)")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # 在格子內標數字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _top_n_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_bad: np.ndarray,
    paths: list[str],
    classes: list[str],
    n: int = 10,
) -> list[dict]:
    """挑出「模型最自信卻錯了」的 top-N 樣本。"""
    bad_idx = classes.index("Bad")
    errors: list[dict] = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue
        # 預測機率(對該預測類別的信心)
        conf = p_bad[i] if y_pred[i] == bad_idx else (1 - p_bad[i])
        errors.append({
            "path": paths[i],
            "true": classes[int(y_true[i])],
            "pred": classes[int(y_pred[i])],
            "confidence": float(conf),
            "p_bad": float(p_bad[i]),
        })
    # 按信心倒序(越「自信地錯」越前面)
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    return errors[:n]


def evaluate(
    model_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    top_n_errors: int = 10,
) -> dict:
    """主流程,回傳 metrics dict。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型:{model_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"找不到資料目錄:{data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device()
    print(f"[info] device = {device}")
    print(f"[info] model  = {model_path}")
    print(f"[info] data   = {data_root}")
    print(f"[info] output = {output_dir}")

    # 載入 checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", "mobilenet_v2")
    classes = ckpt["classes"]
    threshold = float(ckpt.get("decision_threshold", 0.5))
    print(f"[info] arch={arch}  classes={classes}  threshold={threshold:.3f}")

    model = build_model(arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 用「跟訓練同樣的 SEED + val/test 切分」拿到一致的 test set
    _, val_loader, test_loader, _ = make_loaders(
        data_root, batch_size, val_ratio, num_workers=0, test_ratio=test_ratio,
    )
    print(f"[info] val n={len(val_loader.dataset)}  test n={len(test_loader.dataset)}")

    # ── Test set 評估 ────────────────────────────────────────────────
    print("\n=== TEST SET ===")
    y_true, y_pred, p_bad, paths = _gather_predictions(
        model, test_loader, device, classes, threshold,
    )

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix(列=真實,欄=預測):")
    print(f"             pred {classes[0]:<5}  pred {classes[1]:<5}")
    for i, cls in enumerate(classes):
        print(f"true {cls:<5}   {cm[i, 0]:5d}        {cm[i, 1]:5d}")

    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3, zero_division=0,
    )
    print(f"\n{report}")

    # 存 confusion matrix PNG
    cm_path = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(cm, classes, cm_path)
    print(f"[ok] confusion matrix → {cm_path}")

    # 存 classification report TXT
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(
        f"Model: {model_path}\nArch:  {arch}\n"
        f"Threshold (P(Bad)): {threshold:.3f}\n"
        f"Test n: {len(y_true)}\n\n{report}\n",
        encoding="utf-8",
    )
    print(f"[ok] classification report → {report_path}")

    # ── Top-N 錯誤案例 ──────────────────────────────────────────────
    errors = _top_n_errors(y_true, y_pred, p_bad, paths, classes, n=top_n_errors)
    err_lines = [
        f"Top {top_n_errors} 「最自信卻錯了」的測試樣本 — 看這些可以理解模型瓶頸\n",
        "─" * 80,
    ]
    for i, e in enumerate(errors, 1):
        err_lines.append(
            f"{i:2d}. [真實:{e['true']:<4} → 預測:{e['pred']:<4}] "
            f"信心 {e['confidence']*100:5.1f}%  P(Bad)={e['p_bad']:.3f}\n"
            f"    {e['path']}"
        )
    errors_path = output_dir / "top_errors.txt"
    errors_path.write_text("\n".join(err_lines), encoding="utf-8")
    print(f"[ok] top errors → {errors_path}")
    print("\n看一下這幾張最值得復盤的:")
    for e in errors[:5]:
        print(f"  [{e['true']}→{e['pred']}, {e['confidence']*100:.1f}%]  {Path(e['path']).name}")

    return {
        "test_acc": float((y_true == y_pred).mean()),
        "test_n": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "threshold": threshold,
        "top_errors": errors,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="模型評估:不重訓,直接報告 test set 表現")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--top-n-errors", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        evaluate(
            args.model, args.data, args.output_dir,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            top_n_errors=args.top_n_errors,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
