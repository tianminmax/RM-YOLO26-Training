"""用「原数据集 + 新增(预)标注数据」构建一个新数据集目录（原数据集保持不动）。

用法：
    # 先看会构建成什么样（不写任何文件）
    python merge_addon.py --dry-run

    # 真正构建数据集（生成 datasets/rm_plus/ 及其 data.yaml）
    python merge_addon.py --min-conf 0.4

可选过滤：
    --min-conf 0.4    丢弃最高置信度低于该值的整张图（太不确定，不值得冒险）
    --min-boxes 1     至少要有几个框才要
    --min-boxes 0     连"零检出"的图也接受（当纯背景负样本用，务必先人工确认确实没有目标）
    --drop-minimap    丢弃小地图区域(x>0.85,y>0.72)的框（应对标注规范不一致）

为什么构建新数据集而不是直接改 datasets/rm：
  原数据集不被修改，val/test 永远干净；不满意就删掉新目录，零风险回滚。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a new dataset = original + pseudo-labeled extra images")
    p.add_argument("--addon", type=Path, default=ROOT / "datasets" / "rm_addon", help="autolabel.py 的输出目录")
    p.add_argument("--src-images", type=Path, default=ROOT / "data" / "unlabeled" / "images")
    p.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "rm", help="原始数据集（只读，不修改）")
    p.add_argument("--target", type=Path, default=ROOT / "datasets" / "rm_plus", help="构建出的新数据集目录")
    p.add_argument("--min-conf", type=float, default=0.4)
    p.add_argument("--min-boxes", type=int, default=1)
    p.add_argument("--drop-minimap", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    opt = parse_args()
    stats_csv = opt.addon / "autolabel_stats.csv"
    if not stats_csv.exists():
        raise SystemExit(f"找不到 {stats_csv}，请先运行 autolabel.py")

    rows = list(csv.DictReader(stats_csv.open()))
    accepted, rejected = [], []
    for r in rows:
        ok = int(r["boxes"]) >= opt.min_boxes and float(r["max_conf"]) >= opt.min_conf
        (accepted if ok else rejected).append(r["image"])

    src_lab = opt.addon / "labels"
    boxes_before = boxes_after = 0
    final_labels: dict[str, str] = {}
    for name in accepted:
        txt = (src_lab / f"{Path(name).stem}.txt").read_text()
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        boxes_before += len(lines)
        if opt.drop_minimap:
            lines = [ln for ln in lines
                     if not (float(ln.split()[1]) > 0.85 and float(ln.split()[2]) > 0.72)]
        boxes_after += len(lines)
        if lines or opt.min_boxes == 0:
            final_labels[name] = "\n".join(lines) + ("\n" if lines else "")

    accepted = list(final_labels)
    base_train = sorted((opt.dataset / "images" / "train").glob("*.jpg"))
    print(f"新增图片 {len(accepted)} 张（拒绝 {len(rejected)} 张：置信度<{opt.min_conf} 或框数<{opt.min_boxes}）")
    print(f"新增框数 {boxes_after}（过滤前 {boxes_before}）")
    print(f"train 原始 {len(base_train)} 张 -> 新数据集 {len(base_train) + len(accepted)} 张")
    if rejected[:5]:
        print(f"被拒样本示例: {rejected[:5]}")
    print(f"输出目录: {opt.target}（原始 {opt.dataset} 不改动）")

    if opt.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return

    # 1) 把原数据集的 train/val/test 全部软链到新目录（val/test 原样保留）
    for split in ("train", "val", "test"):
        (opt.target / "images" / split).mkdir(parents=True, exist_ok=True)
        (opt.target / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted((opt.dataset / "images" / split).glob("*.jpg")):
            lab = opt.dataset / "labels" / split / f"{img.stem}.txt"
            for dst, src in ((opt.target / "images" / split / img.name, img),
                             (opt.target / "labels" / split / lab.name, lab)):
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src)

    # 2) 新增图片只写进 train
    for name in accepted:
        img_dst = opt.target / "images" / "train" / name
        lab_dst = opt.target / "labels" / "train" / f"{Path(name).stem}.txt"
        if img_dst.is_symlink() or img_dst.exists():
            img_dst.unlink()
        img_dst.symlink_to(opt.src_images / name)
        lab_dst.write_text(final_labels[name])

    # 3) 写 data.yaml
    yaml_path = opt.target / "data.yaml"
    yaml_path.write_text(
        "# RM 作业 - 原数据 + 新增(预)标注数据\n"
        f"path: {opt.target}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n  0: robot\n"
    )

    counts = {s: len(list((opt.target / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test")}
    print(f"\n完成：{counts}")
    print(f"data.yaml: {yaml_path}")
    print(f"训练命令：python train.py --name exp_960_addon --data {yaml_path} "
          f"--epochs 150 --imgsz 960 --batch 4 --workers 2 --device 0")
    print(f"回滚：删除 {opt.target} 即可，原始数据集未受影响。")


if __name__ == "__main__":
    main()
