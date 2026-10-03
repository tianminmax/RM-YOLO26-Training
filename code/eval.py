"""在指定 split 上评估权重，并打印可直接抄进作业文档的指标行。

用法：
    # 训练完立刻用 test split 拿一个"诚实"的数字（写文档用）
    python eval.py --weights runs/baseline_n_640/weights/best.pt --split test

    # 换不同 imgsz 评估同一个权重，看推理尺寸对精度的影响
    python eval.py --weights runs/baseline_n_640/weights/best.pt --split test --imgsz 960 --batch 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codebase" / "ultralytics-YOLO26"))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained YOLO weight")
    p.add_argument("--weights", required=True, help="要评估的权重，例如 runs/xxx/weights/best.pt")
    p.add_argument("--data", type=Path, default=ROOT / "datasets" / "rm" / "data.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default=None)
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    weights = Path(opt.weights)
    run_name = opt.name or f"{weights.parent.parent.name}_{opt.split}_{opt.imgsz}"

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(opt.data),
        split=opt.split,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        project=str(ROOT / "runs" / "val"),
        name=run_name,
        exist_ok=True,
        plots=True,
    )
    b = metrics.box
    out_dir = ROOT / "runs" / "val" / run_name

    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(weights),
        "run": weights.parent.parent.name,
        "split": opt.split,
        "imgsz": opt.imgsz,
        "batch": opt.batch,
        "precision": round(b.mp, 4),
        "recall": round(b.mr, 4),
        "mAP50": round(b.map50, 4),
        "mAP75": round(b.map75, 4),
        "mAP50-95": round(b.map, 4),
    }

    # ① 本次评估的完整指标落盘，永不丢失
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))

    # ② 追加到总表，直接就是作业文档里的训练表格
    summary = ROOT / "runs" / "summary.csv"
    new_file = not summary.exists()
    if not new_file:
        # 指标列变化（例如后来加了 mAP75）时，旧表头不兼容，自动归档重建，避免写错列
        with summary.open() as f:
            old_header = f.readline().strip().split(",")
        if old_header != list(record.keys()):
            backup = summary.with_name("summary_old.csv")
            summary.replace(backup)
            print(f"[提示] 指标列已变化，旧表已归档为 {backup}，本次重新建表。")
            new_file = True
    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(record)

    print("\n" + "=" * 70)
    print(f"权重       : {weights}")
    print(f"评估集     : {opt.split}  (imgsz={opt.imgsz})")
    print(f"P / R      : {b.mp:.4f} / {b.mr:.4f}")
    print(f"mAP50      : {b.map50:.4f}")
    print(f"mAP75      : {b.map75:.4f}")
    print(f"mAP50-95   : {b.map:.4f}")
    print(f"PR曲线/混淆矩阵: {out_dir}")
    print(f"指标已落盘 : {out_dir / 'metrics.json'}")
    print(f"总表已追加 : {summary}")
    print("\n可直接抄进作业文档的一行：")
    print(f"| 编号 | {weights.parent.parent.name} | epochs - | lr - | 优化器 auto | imgsz {opt.imgsz} | "
          f"mAP50 {b.map50:.4f} | mAP75 {b.map75:.4f} |")
    print("=" * 70)


if __name__ == "__main__":
    main()
