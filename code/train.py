"""RM 第四次培训作业 —— YOLO26n 训练脚本（可复现、带超参数记录）。

用法示例：
    # 冒烟测试：5% 数据、3 epoch，10 分钟验证整条链路
    python train.py --name smoke --epochs 3 --fraction 0.05

    # baseline：全量数据
    python train.py --name baseline_n_640 --epochs 100 --imgsz 640 --batch 16

    # 调参实验：只改一个变量
    python train.py --name exp_imgsz960 --epochs 100 --imgsz 960 --batch 8

脚本会自动：
- 把 codebase 里的本地 ultralytics(8.4.7) 插到 sys.path 最前面，保证用的是作业给的 YOLO26 源码；
- 训练结束后打印最佳权重路径 + 最后一轮的 mAP，并给出可直接抄进作业表格的一行。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODEBASE = ROOT / "codebase" / "ultralytics-YOLO26"
sys.path.insert(0, str(CODEBASE))  # 必须放在 import ultralytics 之前

# ---- 整机保护：限制每个进程的数学库线程数 ----
# dataloader 的每个 worker 都会复制一份 OpenMP 线程池。32 核机器上 workers=8 时，
# 8 x 32 = 256 个线程互相抢 CPU，加上 14G 内存无 swap，桌面会直接卡死。
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
# 8G 显存下减少显存碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLO  # noqa: E402

DEFAULT_DATA = ROOT / "datasets" / "rm" / "data.yaml"
DEFAULT_WEIGHTS = CODEBASE / "yolo26n.pt"
DEFAULT_PROJECT = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO26 on the RM dataset")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="预训练权重；用 --cfg 可从零初始化")
    p.add_argument("--cfg", type=Path, default=CODEBASE / "ultralytics/cfg/models/26/yolo26.yaml")
    p.add_argument("--from-scratch", action="store_true", help="不用预训练权重（不推荐）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="8GB 显存：8 最稳，16 更快但更容易吃满")
    p.add_argument("--lr0", type=float, default=None, help="初始学习率；不填用 ultralytics 自动值")
    p.add_argument("--device", default="0", help="0 = 第一块 GPU，cpu = 用 CPU")
    p.add_argument("--workers", type=int, default=2, help="dataloader 进程数；8G/14G 的笔记本建议 2")
    p.add_argument("--fraction", type=float, default=1.0, help="只使用部分训练集（调试用）")
    p.add_argument("--patience", type=int, default=50, help="早停：连续多少轮没提升就停")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    p.add_argument("--name", default="baseline_n_640")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度（默认开启，省显存提速）")
    p.add_argument("--cache", action="store_true", help="把图片缓存进内存，训练更快但吃内存")
    p.add_argument("--resume", action="store_true", help="从上次中断处继续")
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    if not opt.data.exists():
        raise SystemExit(f"找不到 data.yaml: {opt.data}")

    if opt.resume:
        # 续训必须从该次训练的 last.pt 恢复：里面有 epoch / optimizer / lr 调度等完整状态
        ckpt = opt.project / opt.name / "weights" / "last.pt"
        if not ckpt.exists():
            raise SystemExit(f"找不到断点文件 {ckpt}，无法续训（检查 --project/--name 是否写对）")

        # 关键：ultralytics 允许续训时覆盖 imgsz，一旦不一致，模型尺度会被悄悄换掉，
        # 曲线会出现无法解释的掉点。这里强制与 checkpoint 对齐，绝不静默改尺寸。
        import torch

        ck_args = torch.load(ckpt, map_location="cpu", weights_only=False).get("train_args", {})
        explicit = {a.split("=")[0].lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
        ck_imgsz = ck_args.get("imgsz")
        if "imgsz" in explicit and ck_imgsz and opt.imgsz != ck_imgsz:
            raise SystemExit(
                f"拒绝执行：续训 --imgsz={opt.imgsz} 与断点 imgsz={ck_imgsz} 不一致。\n"
                f"改输入尺寸等于换任务，请用新的 --name 从头训练，不要在续训里改。"
            )
        if ck_imgsz:
            opt.imgsz = ck_imgsz
        opt.epochs = ck_args.get("epochs", opt.epochs)
        opt.weights = ckpt
        print(f"[resume] 从 {ckpt} 继续训练（imgsz={opt.imgsz}, epochs={opt.epochs}）")

    source = str(opt.cfg) if opt.from_scratch else str(opt.weights)
    model = YOLO(source)
    model.info()

    model.train(
        data=str(opt.data),
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        lr0=opt.lr0,
        device=opt.device,
        workers=opt.workers,
        fraction=opt.fraction,
        patience=opt.patience,
        seed=opt.seed,
        amp=not opt.no_amp,
        cache=opt.cache,
        resume=opt.resume,
        project=str(opt.project),
        name=opt.name,
        exist_ok=True,
        plots=True,  # 生成 PR 曲线、混淆矩阵等，写文档要用
    )

    run_dir = opt.project / opt.name
    results_csv = run_dir / "results.csv"
    print("\n" + "=" * 70)
    print(f"最佳权重: {run_dir / 'weights' / 'best.pt'}")
    print(f"训练曲线/PR曲线/混淆矩阵: {run_dir}")
    if results_csv.exists():
        with results_csv.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))
            print("\n可直接抄进作业表格的一行：")
            print(
                f"| {opt.name} | yolo26n | {opt.imgsz} | {opt.epochs} | {opt.batch} | "
                f"{opt.lr0 or 'auto'} | {'无' if opt.fraction == 1.0 else f'fraction={opt.fraction}'} | "
                f"{'off' if opt.no_amp else 'on'} | {best.get('epoch', '').strip()} | "
                f"{float(best.get('metrics/mAP50(B)', 0)):.4f} | "
                f"{float(best.get('metrics/mAP50-95(B)', 0)):.4f} | | |"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
