"""用已训练模型给未标注数据生成「预标注」（YOLO 格式 txt），供人工核对后使用。

重要：预标注 ≠ 标注。直接把预测当标签会引入错误监督，必须人工过一遍再用于训练。

用法：
    # 先小样本试跑，看置信度分布是否健康
    python autolabel.py --limit 12 --conf 0.25

    # 全量预标注（302 张，GPU 约 30 秒）
    python autolabel.py --conf 0.25

输出：
    datasets/rm_addon/labels/*.txt        预标注标签
    datasets/rm_addon/autolabel_stats.csv 每张图的框数与最高置信度
    datasets/rm_addon/review/             抽样的红框检查图（可直接肉眼过）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codebase" / "ultralytics-YOLO26"))

# 8G 显存下减少碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-label unlabeled images")
    p.add_argument("--weights", default=str(ROOT / "runs" / "exp_imgsz960_v2" / "weights" / "best.pt"))
    p.add_argument("--src", type=Path, default=ROOT / "data" / "unlabeled" / "images")
    p.add_argument("--out", type=Path, default=ROOT / "datasets" / "rm_addon")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25, help="低于此置信度的框不写入标签")
    p.add_argument("--batch", type=int, default=4, help="显存不足时降到 2 或 1")
    p.add_argument("--device", default="0")
    p.add_argument("--no-half", action="store_true", help="关闭半精度（默认 GPU 上自动开启，省一半显存）")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部），用于试跑")
    p.add_argument("--offset", type=int, default=0, help="从第 N 张开始处理（配合 --limit 分块跑）")
    p.add_argument("--sample", type=int, default=0, help="均匀抽样 N 张（0=不抽样），用于快速看效果")
    p.add_argument("--review", type=int, default=24, help="生成多少张可视化检查图")
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    images = sorted(opt.src.glob("*.jpg"))
    if opt.offset:
        images = images[opt.offset :]
    if opt.limit:
        images = images[: opt.limit]
    if opt.sample and opt.sample < len(images):
        step = len(images) / opt.sample
        images = [images[int(i * step)] for i in range(opt.sample)]
    if not images:
        raise SystemExit(f"{opt.src} 下没有图片")

    lab_dir = opt.out / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    review_dir = opt.out / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(opt.weights)
    results = model.predict(
        [str(p) for p in images], imgsz=opt.imgsz, conf=min(opt.conf, 0.05),
        device=opt.device, batch=opt.batch, verbose=False, stream=True,
        half=(opt.device != "cpu" and not opt.no_half),
    )

    rows, all_confs, empty = [], [], []
    for img_path, r in zip(images, results):
        W, H = r.orig_shape[1], r.orig_shape[0]
        boxes = [(b, float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        kept = [(b, c) for b, c in boxes if c >= opt.conf]

        lines = []
        for (x1, y1, x2, y2), _ in kept:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lab_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        confs = [c for _, c in kept]
        all_confs.extend(confs)
        if not kept:
            empty.append(img_path.name)
        rows.append({
            "image": img_path.name,
            "boxes": len(kept),
            "min_conf": round(min(confs), 3) if confs else 0.0,
            "max_conf": round(max(confs), 3) if confs else 0.0,
            "dropped_low_conf": len(boxes) - len(kept),
        })

    stats = opt.out / "autolabel_stats.csv"
    # 分块多次运行时不能覆盖已有统计：按图片名合并，避免丢数据
    if stats.exists():
        old = {r["image"]: r for r in csv.DictReader(stats.open())}
        for r in rows:
            old[r["image"]] = r
        rows = [old[k] for k in sorted(old)]
    with stats.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "boxes", "min_conf", "max_conf", "dropped_low_conf"])
        w.writeheader()
        w.writerows(rows)

    # 抽样生成检查图：优先挑"框少 / 置信度低 / 零检出"的可疑样本
    order = sorted(rows, key=lambda r: (r["boxes"], r["max_conf"]))[: opt.review]
    for row in order:
        p = opt.src / row["image"]
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        for line in (lab_dir / f"{Path(row['image']).stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = [float(v) for v in line.split()]
            W, H = im.size
            d.rectangle([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
                        outline=(255, 0, 0), width=6)
        im.thumbnail((1400, 1400))
        im.save(review_dir / f"chk_{Path(row['image']).stem}.jpg", quality=85)

    confs_sorted = sorted(all_confs)
    print("\n" + "=" * 70)
    print(f"权重 {opt.weights} | imgsz={opt.imgsz} | 写入阈值 conf>={opt.conf}")
    print(f"处理 {len(images)} 张，共写入 {len(all_confs)} 个框，平均 {len(all_confs) / max(len(images), 1):.2f} 框/图")
    if confs_sorted:
        q = lambda p: confs_sorted[min(len(confs_sorted) - 1, int(len(confs_sorted) * p))]
        print(f"置信度分位: min {confs_sorted[0]:.2f} | 25% {q(.25):.2f} | 中位 {q(.5):.2f} | 75% {q(.75):.2f} | max {confs_sorted[-1]:.2f}")
    print(f"零检出图片 {len(empty)} 张：{empty[:8]}{' ...' if len(empty) > 8 else ''}")
    print(f"\n标签: {lab_dir}")
    print(f"统计: {stats}")
    print(f"人工检查图: {review_dir}（优先给出的是最可疑的 {len(order)} 张）")
    print("=" * 70)
    print("下一步：先看 review/ 里的检查图，确认框得对不对；再决定是否把它们并入训练集。")


if __name__ == "__main__":
    main()
