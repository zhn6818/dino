"""
Grad-CAM visualization for DINO-trained ResNet50.
Usage:
    python visualize_resnet.py --image_path img/test.jpg
    python visualize_resnet.py --image_path img/test.jpg --checkpoint_key teacher
"""
import argparse
import math
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gradcam_with_channels(model, img_tensor, target_layer, top_k=100):
    """Compute averaged Grad-CAM and per-channel CAMs in one forward/backward pass."""
    features, grads = [], []

    def fwd_hook(module, input, output):
        features.append(output)

    def bwd_hook(module, grad_input, grad_output):
        grads.append(grad_output[0])

    h = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    output = model(img_tensor)
    output[:, output.argmax(dim=1).item()].backward()

    h.remove()
    h2.remove()

    feat = features[0]  # [1, C, Hf, Wf]
    grad = grads[0]     # [1, C, Hf, Wf]
    weights = grad.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

    # Averaged Grad-CAM
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    avg_heatmap = cam.squeeze().detach().cpu().numpy()

    # Per-channel CAMs
    C = feat.shape[1]
    k = min(top_k, C) if top_k > 0 else C
    per_channel_cam = F.relu(weights * feat)  # [1, C, Hf, Wf]
    topk_vals, topk_idx = weights[0, :, 0, 0].abs().topk(k)
    selected_cam = per_channel_cam[0, topk_idx]  # [K, Hf, Wf]
    topk_weights = weights[0, topk_idx, 0, 0]    # [K]

    selected_cam = F.interpolate(
        selected_cam.unsqueeze(0),
        size=img_tensor.shape[2:],
        mode="bilinear",
        align_corners=False,
    )[0]  # [K, Hi, Wi]

    cam_min = selected_cam.flatten(1).min(dim=1).values[:, None, None]
    cam_max = selected_cam.flatten(1).max(dim=1).values[:, None, None]
    selected_cam = (selected_cam - cam_min) / (cam_max - cam_min + 1e-8)

    return (
        avg_heatmap,
        selected_cam.detach().cpu().numpy(),
        topk_idx.detach().cpu().numpy(),
        topk_weights.detach().cpu().numpy(),
    )


def save_per_channel_grid(selected_cam, topk_idx, topk_weights, img_np,
                          layer_name, output_dir, name, ncols=10):
    """Save per-channel CAMs as a grid image."""
    K = selected_cam.shape[0]
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    fig.suptitle(f"Per-Channel Grad-CAM: {layer_name} (top {K} by |weight|)", fontsize=14)

    for i in range(K):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        heatmap_color = cv2.applyColorMap(np.uint8(255 * selected_cam[i]), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"ch{topk_idx[i]} w={topk_weights[i]:.6f}", fontsize=6)
        ax.axis("off")

    for i in range(K, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis("off")

    plt.tight_layout()
    save_path = f"{output_dir}/{name}_gradcam_{layer_name}_channels.jpg"
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {layer_name} channel grid -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--pretrained_weights", default="dino_output/resnet50/checkpoint0190.pth")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--image_path", default="img/test4.jpg")
    parser.add_argument("--output_dir", default="gradcam_maps")
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--top_k", default=100, type=int,
                        help="Number of top channels to show in per-channel grid (by |weight|). -1 for all.")
    parser.add_argument("--no_grid", action="store_true",
                        help="Skip per-channel grid generation (only save averaged Grad-CAM).")
    args = parser.parse_args()

    # Load model
    model = models.__dict__[args.arch]()
    model.fc = torch.nn.Identity()

    if args.pretrained_weights:
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key and args.checkpoint_key in state_dict:
            print(f"Using key '{args.checkpoint_key}' from checkpoint")
            state_dict = state_dict[args.checkpoint_key]
        # Remove DINOHead keys, keep backbone only
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    print(f"Loaded {args.arch}")

    # 对 ResNet50 的 4 个 stage 分别取最后一个卷积层进行 Grad-CAM 可视化
    target_layers = {
        "layer1": model.layer1[-1].conv2,  # stage1 末尾卷积，捕获低级特征（边缘、纹理）
        "layer2": model.layer2[-1].conv3,  # stage2 末尾卷积，捕获中级特征（局部结构）
        "layer3": model.layer3[-1].conv3,  # stage3 末尾卷积，捕获中高级特征（部件、模式）
        "layer4": model.layer4[-1].conv3,  # stage4 末尾卷积，捕获高级语义特征（物体类别）
    }

    tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = Image.open(args.image_path).convert("RGB")
    img_np = np.array(img.resize((args.image_size, args.image_size)))

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image_path))[0]

    for layer_name, target_layer in target_layers.items():
        img_tensor = tfm(img).unsqueeze(0).requires_grad_(True)
        avg_heatmap, selected_cam, topk_idx, topk_weights = gradcam_with_channels(
            model, img_tensor, target_layer, top_k=args.top_k
        )

        # 将平均热力图叠加到原图上
        heatmap_color = cv2.applyColorMap(np.uint8(255 * avg_heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)

        save_path = f"{args.output_dir}/{name}_gradcam_{layer_name}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved {layer_name} -> {save_path}")

        # 逐通道网格图
        if not args.no_grid:
            save_per_channel_grid(
                selected_cam, topk_idx, topk_weights,
                img_np, layer_name, args.output_dir, name
            )


if __name__ == "__main__":
    main()
