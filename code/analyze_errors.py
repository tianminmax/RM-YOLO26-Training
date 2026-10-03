"""逐张分析检测误差：到底哪些图漏检、哪些图误检，并画出对比图。

不依赖 COCO 格式，直接读 YOLO 标签做 IoU 匹配，输出：
  runs/error_analysis/<split>_errors.csv   每张图的 TP/FN/FP 统计
  runs/error_analysis/<split>_<名称>.jpg   漏检/误检图 (红=标注, 蓝=预测)

用法：
    python analyze_errors.py --weights runs/exp_imgsz960_v2/weights/best.pt --split test --imgsz 960
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codebase" / "ultralytics-YOLO26"))

from ultralytics import YOLO  # noqa: E402


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-image detection error analysis")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25, help="与提交推理保持一致")
    p.add_argument("--iou", type=float, default=0.5, help="匹配阈值")
    p.add_argument("--save-plots", type=int, default=12, help="最多为几张问题图生成对比图")
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    data_root = ROOT / "datasets" / "rm"
    img_dir, lab_dir = data_root / "images" / opt.split, data_root / "labels" / opt.split
    images = sorted(img_dir.glob("*.jpg"))
    out_dir = ROOT / "runs" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(opt.weights))
    results = model.predict(
        [str(p) for p in images], imgsz=opt.imgsz, conf=opt.conf, device=opt.device,
        batch=opt.batch, verbose=False, stream=True,
    )

    rows, problems = [], []
    tot_gt = tot_tp = tot_fn = tot_fp = 0
    for img_path, r in zip(images, results):
        W, H = r.orig_shape[1], r.orig_shape[0]
        gts = []
        lab = lab_dir / f"{img_path.stem}.txt"
        if lab.exists():
            for line in lab.read_text().splitlines():
                if line.strip():
                    _, x, y, w, h = [float(v) for v in line.split()[:5]]
                    gts.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
        preds = [list(map(float, b)) for b in r.boxes.xyxy.tolist()]
        confs = [float(c) for c in r.boxes.conf.tolist()]

        # 贪心匹配：每个 GT 找与其 IoU 最大的未使用预测框
        used, tp = set(), 0
        for g in gts:
            best, bi = opt.iou, None
            for i, pb in enumerate(preds):
                if i in used:
                    continue
                v = iou(g, pb)
                if v >= best:
                    best, bi = v, i
            if bi is not None:
                used.add(bi)
                tp += 1
        fn, fp = len(gts) - tp, len(preds) - len(used)
        tot_gt, tot_tp, tot_fn, tot_fp = tot_gt + len(gts), tot_tp + tp, tot_fn + fn, tot_fp + fp
        rows.append({"image": img_path.name, "gt": len(gts), "tp": tp, "fn": fn, "fp": fp,
                     "max_conf": round(max(confs), 3) if confs else 0.0})
        if fn or fp:
            problems.append((img_path.name, fn, fp, gts, preds, confs))

    csv_path = out_dir / f"{opt.split}_errors.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "gt", "tp", "fn", "fp", "max_conf"])
        w.writeheader()
        w.writerows(rows)

    fn_images = [p for p in problems if p[1] > 0]
    fn_images.sort(key=lambda p: -p[1])
    for name, fn, fp, gts, preds, confs in fn_images[: opt.save_plots]:
        im = Image.open(img_dir / name).convert("RGB")
        d = ImageDraw.Draw(im)
        for g in gts:
            d.rectangle(g, outline=(255, 0, 0), width=6)
        for pb, cf in zip(preds, confs):
            d.rectangle(pb, outline=(0, 120, 255), width=6)
            d.text((pb[0] + 8, pb[1] + 8), f"{cf:.2f}", fill=(0, 120, 255))
        im.thumbnail((1400, 1400))
        im.save(out_dir / f"{opt.split}_{Path(name).stem}.jpg", quality=85)

    print("\n" + "=" * 70)
    print(f"权重 {opt.weights} | {opt.split} 集 | imgsz={opt.imgsz} conf={opt.conf} 匹配IoU={opt.iou}")
    print(f"GT 目标 {tot_gt} 个 -> TP {tot_tp} | 漏检 FN {tot_fn} ({tot_fn / max(tot_gt, 1):.1%}) | 误检 FP {tot_fp}")
    print(f"含漏检的图片 {len(fn_images)} 张，漏检最多的 5 张：")
    for name, fn, fp, *_ in fn_images[:5]:
        print(f"   {name}: 漏检 {fn} 个, 误检 {fp} 个")
    print(f"\n明细表: {csv_path}")
    print(f"对比图(红=标注,蓝=预测): {out_dir}/{opt.split}_*.jpg")
    print("=" * 70)


if __name__ == "__main__":
    main()
