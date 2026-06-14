#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别平衡 (Class Balancing)
==========================
对 origin_data_split 中各类别的小图进行随机下采样，使每类小图数量对齐到最少类别，
消除类别不平衡。直接在 origin_data_split 目录上删除多余小图
（源数据 origin_data 保留，可随时重切）。

重要：小图命名含原图名（{原图名}_{x}_{y}.jpg），后续做 train/val 划分时
必须按原图名分组，否则同一原图的小块分属 train/val 会造成数据泄漏。

用法：
  python3 balance_patches.py                 # 每类对齐到最少类（ZZ）
  python3 balance_patches.py --target 12000  # 每类下采样到 12000
  python3 balance_patches.py --dry-run       # 只预览不删除
"""
import os
import argparse
import random

DEFAULT_DIR = "/data1/zhn/jinxiang/origin_data_split"
ALL_CLASSES = ["JLD", "JZ", "TT", "ZZ"]


def list_patches(cls_dir):
    """列出某类别目录下的所有图片文件（已排序，保证采样可复现）。"""
    return sorted(f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def main():
    parser = argparse.ArgumentParser(description="类别平衡：随机下采样各类小图")
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"小图目录 (默认 {DEFAULT_DIR})")
    parser.add_argument("--classes", nargs="+", default=ALL_CLASSES, help="要处理的类别")
    parser.add_argument("--target", type=int, default=0, help="每类目标数量；0=对齐最少类")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（保证可复现）")
    parser.add_argument("--dry-run", action="store_true", help="只预览不实际删除")
    args = parser.parse_args()

    # 先统计各类数量，确定目标
    counts = {}
    for cls in args.classes:
        cls_dir = os.path.join(args.dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[跳过] 目录不存在: {cls_dir}")
            continue
        counts[cls] = len(list_patches(cls_dir))

    if not counts:
        print("无可处理类别，退出。")
        return

    target = args.target if args.target > 0 else min(counts.values())

    print("=" * 60)
    print(f"类别平衡: 目标每类 {target} 块  ({'预览模式' if args.dry_run else '实际删除'})")
    print(f"当前各类数量: {counts}")
    print("=" * 60, flush=True)

    total_removed = 0
    for cls in args.classes:
        if cls not in counts:
            continue
        cls_dir = os.path.join(args.dir, cls)
        files = list_patches(cls_dir)
        n = len(files)
        if n <= target:
            print(f"  [{cls}] {n} <= {target}, 保持不变")
            continue
        random.seed(args.seed)
        keep = set(random.sample(files, target))
        removed = 0
        for f in files:
            if f not in keep:
                if not args.dry_run:
                    os.remove(os.path.join(cls_dir, f))
                removed += 1
        total_removed += removed
        print(f"  [{cls}] {n} -> {target}  (删除 {removed})", flush=True)

    print("=" * 60)
    if args.dry_run:
        print(f"[预览] 计划删除 {total_removed} 块, 每类保留 {target}")
    else:
        print(f"完成: 共删除 {total_removed} 块, 每类对齐到 {target}")
    print("=" * 60)


if __name__ == "__main__":
    main()
