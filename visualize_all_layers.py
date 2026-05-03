# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DINO 全层可视化：Attention + PCA + CLS Similarity
  1. 每层每个 head 的 CLS-to-Patch 注意力热力图 + 阈值分割 + 轮廓叠加
  2. PCA 降维特征可视化（语义分割效果）
  3. CLS token 与 patch token 的余弦相似度
  4. 单层综合图（5 列：原图 / 注意力均值 / CLS 相似度 / PCA / 分割）
  5. 全层总览图
"""
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from pathlib import Path

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

import utils
import vision_transformer as vits


# ── 工具函数 ────────────────────────────────────────────


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    mask = mask[None, :, :]
    colors = random_colors(1)

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    color = colors[0]
    _mask = mask[0]
    if blur:
        _mask = cv2.blur(_mask, (10, 10))
    masked_image = apply_mask(masked_image, _mask, color, alpha)
    if contour:
        padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
        padded_mask[1:-1, 1:-1] = _mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    plt.close(fig)


def threshold_attention(attn, threshold):
    """保留 top threshold% 的注意力质量，返回二值 mask。"""
    nh, n_patches = attn.shape
    val, idx = torch.sort(attn, dim=1)
    val = val / val.sum(dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx, dim=1)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    return th_attn.float()


def pca_to_rgb(features, h_feat, w_feat):
    """PCA 降维到 3 维作为 RGB，带符号校正。"""
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(features)
    for j in range(3):
        if rgb[:, j].mean() < 0:
            rgb[:, j] = -rgb[:, j]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.reshape(h_feat, w_feat, 3)


# ── 特征提取 ────────────────────────────────────────────


def extract_attention_and_features(model, x):
    """
    通过 hook 提取每层的注意力权重和特征，一次前向传播完成。
    返回:
      attentions: list of (1, num_heads, num_patches) 每层 CLS-to-Patch 注意力
      features:   list of (1, num_tokens, embed_dim) 每层经过 norm 的特征
    """
    attentions = []
    features = []
    hooks = []

    def make_hook(idx):
        def attn_hook(module, inp, _out):
            B, N, C = inp[0].shape
            qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            attentions.append(attn[:, :, 0, 1:].detach())

        def feat_hook(module, inp, _out):
            features.append(model.norm(inp[0]).detach())

        if idx == 'attn':
            return attn_hook
        return feat_hook

    attn_hooks = []
    feat_hooks = []
    for blk in model.blocks:
        attn_hooks.append(blk.attn.register_forward_hook(make_hook('attn')))
        feat_hooks.append(blk.register_forward_hook(make_hook('feat')))

    with torch.no_grad():
        model(x)

    for h in attn_hooks + feat_hooks:
        h.remove()

    return attentions, features


# ── 可视化 ──────────────────────────────────────────────


def save_per_head_attention(orig_img_np, attn, patch_size, layer_idx, threshold, output_dir):
    """每层每个 head 的独立热力图 + 阈值分割 + 轮廓叠加。"""
    nh = attn.shape[0]
    n_patches = attn.shape[1]
    h_feat = w_feat = int(n_patches ** 0.5)

    attn_up = F.interpolate(
        attn.reshape(nh, h_feat, w_feat).unsqueeze(0),
        scale_factor=patch_size, mode="nearest",
    )[0].cpu().numpy()

    head_dir = Path(output_dir) / f"layer{layer_idx:02d}_heads"
    head_dir.mkdir(parents=True, exist_ok=True)

    for j in range(nh):
        plt.imsave(head_dir / f"attn-head{j:02d}.png", attn_up[j], format="png")

    th_attn = threshold_attention(attn, threshold)
    th_up = F.interpolate(
        th_attn.reshape(nh, h_feat, w_feat).unsqueeze(0),
        scale_factor=patch_size, mode="nearest",
    )[0].cpu().numpy()

    for j in range(nh):
        display_instances(
            orig_img_np, th_up[j],
            str(head_dir / f"mask_th{threshold}_head{j:02d}.png"),
        )


def save_single_layer(orig_img_np, orig_img, attn, pca_rgb, cls_sim,
                      layer_idx, patch_size, threshold, output_dir):
    """单层综合图：原图 / 注意力均值 / CLS 相似度 / PCA / 分割。"""
    nh = attn.shape[0]
    n_patches = attn.shape[1]
    h_feat = w_feat = int(n_patches ** 0.5)

    attn_up = F.interpolate(
        attn.reshape(1, nh, h_feat, w_feat),
        scale_factor=patch_size, mode="nearest",
    )[0].cpu().numpy()
    mean_attn = attn_up.mean(axis=0)

    th_attn = threshold_attention(attn, threshold)
    th_up = F.interpolate(
        th_attn.reshape(nh, h_feat, w_feat).unsqueeze(0),
        scale_factor=patch_size, mode="nearest",
    )[0].cpu().numpy()
    combined_mask = th_up.max(axis=0)

    masked = orig_img_np.astype(np.uint32).copy()
    colors = random_colors(1)
    masked = apply_mask(masked, combined_mask, colors[0], alpha=0.5)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mean_attn)
    axes[1].set_title(f"Layer {layer_idx} Attention (mean)")
    axes[1].axis("off")

    im2 = axes[2].imshow(cls_sim, cmap="inferno")
    axes[2].set_title(f"Layer {layer_idx} CLS Similarity")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(pca_rgb)
    axes[3].set_title(f"Layer {layer_idx} PCA")
    axes[3].axis("off")

    axes[4].imshow(masked.astype(np.uint8))
    axes[4].set_title(f"Layer {layer_idx} Segmentation")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"layer{layer_idx:02d}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_overview(orig_img_np, orig_img, all_attn, all_pca, all_cls_sim,
                  n_layers, patch_size, threshold, output_dir):
    """全层总览：注意力 + PCA + CLS 相似度 + 分割。"""
    fig, axes = plt.subplots(n_layers, 4, figsize=(20, 5 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    for i in range(n_layers):
        nh = all_attn[i].shape[0]
        n_patches = all_attn[i].shape[1]
        h_feat = w_feat = int(n_patches ** 0.5)

        attn_up = F.interpolate(
            all_attn[i].reshape(1, nh, h_feat, w_feat),
            scale_factor=patch_size, mode="nearest",
        )[0].cpu().numpy()
        mean_attn = attn_up.mean(axis=0)

        th_attn = threshold_attention(all_attn[i], threshold)
        th_up = F.interpolate(
            th_attn.reshape(nh, h_feat, w_feat).unsqueeze(0),
            scale_factor=patch_size, mode="nearest",
        )[0].cpu().numpy()
        combined_mask = th_up.max(axis=0)

        masked = orig_img_np.astype(np.uint32).copy()
        colors = random_colors(1)
        masked = apply_mask(masked, combined_mask, colors[0], alpha=0.5)

        axes[i, 0].imshow(mean_attn)
        axes[i, 0].set_title(f"Layer {i} Attention")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(all_cls_sim[i], cmap="inferno")
        axes[i, 1].set_title(f"Layer {i} CLS Similarity")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(all_pca[i])
        axes[i, 2].set_title(f"Layer {i} PCA")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(masked.astype(np.uint8))
        axes[i, 3].set_title(f"Layer {i} Segmentation")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "overview_all_layers.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 主函数 ──────────────────────────────────────────────


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps and PCA for all layers')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture.')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="Attention threshold, keep top X% mass.")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存归一化后的输入图
    img_save_path = str(output_dir / "img.png")
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        img_save_path,
    )
    orig_img_np = skimage.io.imread(img_save_path)
    orig_pil = Image.fromarray(orig_img_np)

    h_feat = w_feat = img.shape[2] // args.patch_size
    img = img.to(device)

    # ── 一次性提取 attention + features ──
    print("\n=== Extracting attention and features for all layers ===")
    attentions, features = extract_attention_and_features(model, img)
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    print(f"Model: {n_layers} layers, {n_heads} heads per layer")

    # ── 逐层可视化 ──
    all_attn, all_pca, all_cls_sim = [], [], []

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx][0].cpu()  # (nh, N)

        # PCA
        patch_tokens = features[layer_idx][0, 1:].cpu().numpy()  # skip CLS
        pca_rgb = pca_to_rgb(patch_tokens, h_feat, w_feat)
        # 上采样 PCA 到原图尺寸
        pca_up = F.interpolate(
            torch.from_numpy(pca_rgb).permute(2, 0, 1).unsqueeze(0),
            scale_factor=args.patch_size, mode="nearest",
        )[0].permute(1, 2, 0).numpy()

        # CLS 相似度
        all_tokens = features[layer_idx][0]  # (num_tokens, embed_dim)
        pt_norm = F.normalize(all_tokens[1:], dim=-1)
        ct_norm = F.normalize(all_tokens[0:1], dim=-1)
        sim = (pt_norm * ct_norm).sum(dim=-1)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        cls_sim = sim.cpu().numpy().reshape(h_feat, w_feat)

        all_attn.append(attn)
        all_pca.append(pca_up)
        all_cls_sim.append(cls_sim)

        # 每个 head 独立热力图 + mask
        save_per_head_attention(
            orig_img_np, attn, args.patch_size, layer_idx, args.threshold, output_dir,
        )
        # 单层综合图
        save_single_layer(
            orig_img_np, orig_pil, attn, pca_up, cls_sim,
            layer_idx, args.patch_size, args.threshold, output_dir,
        )
        print(f"  Layer {layer_idx} done")

    # ── 全层总览 ──
    save_overview(
        orig_img_np, orig_pil, all_attn, all_pca, all_cls_sim,
        n_layers, args.patch_size, args.threshold, output_dir,
    )

    print(f"\nDone! Results saved to {output_dir}/")
    print(f"  layer00_heads/ ~ layer{n_layers - 1:02d}_heads/")
    print(f"    attn-head00.png ~ head{n_heads - 1:02d}.png    (attention heatmaps)")
    print(f"    mask_th*_head00~{n_heads - 1:02d}.png          (threshold masks)")
    print(f"  layer00_summary.png ~ layer{n_layers - 1:02d}_summary.png  (per-layer summary)")
    print(f"  overview_all_layers.png                          (all layers overview)")
