#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑窗分割 (Sliding Window Patch Splitting)
==========================================
将 origin_data 中的金相大图按滑动窗口切成固定尺寸的小图块，保存到 origin_data_split。

特性：
  - 步长 stride 可调（默认 500，重叠 = 窗口 - 步长）
  - 边缘回卷补全：当原图尺寸不是 stride 的整数倍时，最后一块从右/下边缘往回取，
    保证整张图都被覆盖（最后一块与前一邻块有少量重叠），不丢弃任何区域
  - 多进程并行 + 实时进度打印（不依赖 tqdm）
  - 命名规则：{原图名}_{x起点}_{y起点}.jpg，便于后续按原图分组做 train/val 划分
    （划分时必须按原图分组，否则同一原图的小块同时进入 train/val 会造成数据泄漏）

用法：
  python3 split_patches.py                     # 处理全部 4 个类别
  python3 split_patches.py --classes ZZ        # 只处理 ZZ（用于先小规模验证）
  python3 split_patches.py --stride 448 --window 512
"""
import os
import sys
import time
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

from PIL import Image

# 关闭超大图 DecompressionBomb 警告（金相图分辨率较高）
Image.MAX_IMAGE_PIXELS = None

DEFAULT_SRC = "/data1/zhn/jinxiang/origin_data"
DEFAULT_DST = "/data1/zhn/jinxiang/origin_data_split"
ALL_CLASSES = ["JLD", "JZ", "TT", "ZZ"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_starts(length, window, stride):
    """计算某一维度上的滑动窗口起点列表（回卷补全覆盖到边缘）。

    例: length=2448, window=512, stride=384
        -> [0, 384, 768, 1152, 1536, 1920, 1936]  (最后 1936 = 2448-512 为回卷补全)

    length <= window 时只返回 [0]（整张图取一块，从左上角开始）。
    """
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    # 若最后一个起点未恰好贴到边缘，补一个贴边的起点（回卷补全）
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def split_image(fname, src_dir, dst_dir, window, stride, quality):
    """处理单张原图：滑窗切块并保存。返回切出的小图数量。"""
    src_path = os.path.join(src_dir, fname)
    stem = os.path.splitext(fname)[0]
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")  # 统一为 RGB，避免 RGBA / 灰度图保存 jpg 出错
            W, H = im.size
            xs = get_starts(W, window, stride)
            ys = get_starts(H, window, stride)
            n = 0
            for x in xs:
                for y in ys:
                    patch = im.crop((x, y, x + window, y + window))
                    out_path = os.path.join(dst_dir, f"{stem}_{x}_{y}.jpg")
                    patch.save(out_path, quality=quality)
                    n += 1
            return n
    except Exception as e:
        print(f"[ERROR] {src_path}: {e}", file=sys.stderr)
        return 0


def process_class(cls, src_base, dst_base, window, stride, quality, workers):
    """处理单个类别目录。"""
    src_dir = os.path.join(src_base, cls)
    dst_dir = os.path.join(dst_base, cls)
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        print(f"[跳过] 源目录不存在: {src_dir}")
        return 0

    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        print(f"[跳过] {cls}: 无图片")
        return 0

    func = partial(
        split_image,
        src_dir=src_dir, dst_dir=dst_dir,
        window=window, stride=stride, quality=quality,
    )

    total = 0
    done = 0
    start = time.time()
    # 进度打印频率：约 20 次更新
    step = max(1, len(files) // 20)
    print(f"[{cls}] 开始处理 {len(files)} 张原图 ...", flush=True)
    with Pool(processes=workers) as pool:
        for n in pool.imap_unordered(func, files, chunksize=8):
            total += n
            done += 1
            if done % step == 0 or done == len(files):
                pct = done * 100 // len(files)
                print(f"  [{cls}] {done}/{len(files)} 原图 ({pct}%), 已切出 {total} 块", flush=True)
    elapsed = time.time() - start
    print(f"[{cls}] 完成: {len(files)} 张原图 -> {total} 块小图, 用时 {elapsed:.1f}s", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser(description="滑窗分割大图为 512x512 小图块")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"源目录 (默认 {DEFAULT_SRC})")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"目标目录 (默认 {DEFAULT_DST})")
    parser.add_argument("--classes", nargs="+", default=ALL_CLASSES, help="要处理的类别 (默认全部)")
    parser.add_argument("--window", type=int, default=512, help="窗口尺寸 (默认 512)")
    parser.add_argument("--stride", type=int, default=500, help="滑动步长 (默认 500, 重叠=窗口-步长)")
    parser.add_argument("--quality", type=int, default=95, help="jpg 保存质量 1-100 (默认 95)")
    parser.add_argument("--workers", type=int, default=min(cpu_count(), 16), help="进程数")
    args = parser.parse_args()

    if not (0 < args.stride <= args.window):
        raise SystemExit(f"stride 须满足 0 < stride <= window，当前 stride={args.stride}, window={args.window}")

    overlap = args.window - args.stride

    print("=" * 60)
    print(f"滑窗分割: {args.src}")
    print(f"  -> {args.dst}")
    print(f"  窗口={args.window}, 步长={args.stride}, 重叠={overlap}px ({overlap * 100 // args.window}%)")
    print(f"  类别={args.classes}, 进程数={args.workers}, 质量={args.quality}")
    print("=" * 60, flush=True)

    grand_total = 0
    for cls in args.classes:
        grand_total += process_class(
            cls, args.src, args.dst,
            args.window, args.stride, args.quality, args.workers,
        )

    print("=" * 60)
    print(f"全部完成: 共切出 {grand_total} 块小图 -> {args.dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()
