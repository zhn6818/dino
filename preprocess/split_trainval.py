#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train/val 划分 (按原图分组)
============================
将 origin_data_split 中的平衡小图，按【原图名】分组后划分为 train/val，
输出为 ImageFolder 结构（train/<cls>/, val/<cls>/），可直接喂给 eval_linear.py / finetune。

防数据泄漏：同一原图的所有小块（命名 {原图名}_{x}_{y}.jpg）必定进入同一集合，
train 与 val 的原图组互不相交。

默认复制（保留 origin_data_split 平衡池），加 --move 改为移动。

用法：
  python3 split_trainval.py                  # 默认 8:2，复制
  python3 split_trainval.py --ratio 0.9      # 9:1
  python3 split_trainval.py --move           # 移动（清空源目录，省磁盘）
"""
import os
import shutil
import argparse
import random
from collections import defaultdict

DEFAULT_SRC = "/data1/zhn/jinxiang/origin_data_split"
DEFAULT_DST = "/data1/zhn/jinxiang/finetune_dataset"
ALL_CLASSES = ["JLD", "JZ", "TT", "ZZ"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_origin(fname):
    """从小图文件名解析原图名：去扩展名后，去掉最后两段 _x_y。

    '20230723154032495_1_500_1000.jpg' -> '20230723154032495_1'
    'IMG001_0_0.jpg'                    -> 'IMG001'
    """
    stem = fname.rsplit(".", 1)[0]
    return stem.rsplit("_", 2)[0]


def main():
    parser = argparse.ArgumentParser(description="按原图分组划分 train/val")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"源平衡小图目录 (默认 {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"输出目录 (默认 {DEFAULT_DST})")
    parser.add_argument("--classes", nargs="+", default=ALL_CLASSES, help="要处理的类别")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集原图组比例 (默认 0.8)")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--move", action="store_true", help="移动而非复制（清空源目录，省磁盘）")
    args = parser.parse_args()

    op = shutil.move if args.move else shutil.copy2
    random.seed(args.seed)

    print("=" * 60)
    print(f"train/val 划分: {args.src}")
    print(f"  -> {args.dst}")
    print(f"  训练比例={args.ratio}, 操作={'移动' if args.move else '复制'}, seed={args.seed}")
    print("=" * 60, flush=True)

    total_train = total_val = 0
    summary = []  # 每类 (cls, n_origins, n_train_origins, n_train, n_val, train_origins, val_origins)
    for cls in args.classes:
        cls_dir = os.path.join(args.src, cls)
        if not os.path.isdir(cls_dir):
            print(f"[跳过] 源目录不存在: {cls_dir}")
            continue

        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS))

        # 按原图名分组
        groups = defaultdict(list)
        for f in files:
            groups[parse_origin(f)].append(f)
        origins = sorted(groups.keys())
        random.shuffle(origins)
        n_train_origins = round(len(origins) * args.ratio)
        train_origins = set(origins[:n_train_origins])
        val_origins = set(origins[n_train_origins:])

        train_dir = os.path.join(args.dst, "train", cls)
        val_dir = os.path.join(args.dst, "val", cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        n_tr = n_va = 0
        for origin, fs in groups.items():
            target_dir = train_dir if origin in train_origins else val_dir
            for f in fs:
                op(os.path.join(cls_dir, f), os.path.join(target_dir, f))
            if origin in train_origins:
                n_tr += len(fs)
            else:
                n_va += len(fs)

        total_train += n_tr
        total_val += n_va
        summary.append((cls, len(origins), n_train_origins, n_tr, n_va, train_origins, val_origins))
        print(f"  [{cls}] 原图组 {len(origins)} -> train {n_train_origins} / val {len(origins)-n_train_origins}, "
              f"小图 train {n_tr} / val {n_va}", flush=True)

    grand = total_train + total_val
    print("=" * 60)
    print(f"完成: train {total_train} / val {total_val} (共 {grand})")
    if grand > 0:
        print(f"  训练占比 {total_train / grand * 100:.1f}%")
    print("=" * 60)

    # 无泄漏自检：train 与 val 原图组必须不相交
    leaked = False
    for cls, _, _, _, _, tr_o, va_o in summary:
        inter = tr_o & va_o
        if inter:
            leaked = True
            print(f"[警告] {cls} train/val 原图组有交集 {len(inter)} 个！")
    if not leaked:
        print("[自检] train/val 原图组无交集，无数据泄漏 ✓")


if __name__ == "__main__":
    main()
