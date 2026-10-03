"""划分 RM 数据集：data/labeled -> datasets/rm/{train,val,test}，并生成 data.yaml。

特点：
- 固定随机种子，结果可复现；
- 按「每图框数」分层抽样，保证 HUD 截图（框多）与实拍照片（框少）在三个集合里比例一致；
- 图片/标签用软链接，不复制文件、不占额外磁盘。

用法：
    python prepare_data.py --dry-run     # 只打印统计，不写文件
    python prepare_data.py               # 真正划分
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "labeled"
DST = ROOT / "datasets" / "rm"
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
CLASS_NAMES = {0: "robot"}  # 数据只有 1 类（标签第一列全是 0）


def box_count(label_path: Path) -> int:
    """一张图里标了多少个框。"""
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def bucket(n: int) -> str:
    """按框数分桶，用于分层抽样。"""
    if n <= 2:
        return "1-2"
    if n <= 6:
        return "3-6"
    if n <= 10:
        return "7-10"
    return "11+"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split the RM dataset")
    p.add_argument("--src", type=Path, default=SRC, help="已标注数据目录（含 images/ 与 labels/）")
    p.add_argument("--dst", type=Path, default=DST, help="输出目录")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    images = sorted((args.src / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"没有在 {args.src / 'images'} 找到图片")

    # 1) 收集每张图的框数，检查图片/标签是否成对
    records: list[tuple[Path, int]] = []
    missing = []
    for img in images:
        lab = args.src / "labels" / f"{img.stem}.txt"
        if not lab.exists():
            missing.append(img.name)
            continue
        records.append((img, box_count(lab)))
    if missing:
        raise SystemExit(f"{len(missing)} 张图缺少标签，例如 {missing[:5]}")

    # 2) 按框数分层，每层内按比例切分
    strata: dict[str, list[tuple[Path, int]]] = {}
    for rec in records:
        strata.setdefault(bucket(rec[1]), []).append(rec)

    assign: dict[str, list[Path]] = {s: [] for s in SPLITS}
    for name, group in sorted(strata.items()):
        group = group[:]
        random.shuffle(group)
        n = len(group)
        n_train = int(round(n * RATIOS["train"]))
        n_val = int(round(n * RATIOS["val"]))
        # 防止四舍五入把最后一张漏掉
        n_val = min(n_val, n - n_train)
        chunks = {
            "train": group[:n_train],
            "val": group[n_train : n_train + n_val],
            "test": group[n_train + n_val :],
        }
        print(f"  分层 {name:>4}: 共 {n:>4} 张 -> " + " ".join(f"{s} {len(chunks[s]):>4}" for s in SPLITS))
        for s in SPLITS:
            assign[s].extend(p for p, _ in chunks[s])

    total = sum(len(v) for v in assign.values())
    print(f"\n合计 {total} 张：", {s: len(assign[s]) for s in SPLITS})
    print("平均框数：", {s: round(sum(box_count(ROOT / 'data' / 'labeled' / 'labels' / f'{p.stem}.txt') for p in assign[s]) / max(len(assign[s]), 1), 2) for s in SPLITS})

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return

    # 3) 落地：建目录 + 软链接 + 写 data.yaml
    for s in SPLITS:
        (args.dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / s).mkdir(parents=True, exist_ok=True)

    for s in SPLITS:
        for img in assign[s]:
            lab = args.src / "labels" / f"{img.stem}.txt"
            dst_img = args.dst / "images" / s / img.name
            dst_lab = args.dst / "labels" / s / lab.name
            for link, target in ((dst_img, img), (dst_lab, lab)):
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target)

    yaml_path = args.dst / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
    yaml_path.write_text(
        f"# RM 第四次培训作业 - YOLO26n 数据集配置\n"
        f"path: {args.dst}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    print(f"\n已写入 {yaml_path}")
    print("下一步：python -c \"from ultralytics.data.utils import check_det_dataset; check_det_dataset(r'%s')\"" % yaml_path)


if __name__ == "__main__":
    main()
