"""
Feature-map visualization for DINO-trained ResNet50.
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


def load_backbone_weights(model, pretrained_weights, checkpoint_key):
    """Load only backbone weights from a DINO checkpoint."""
    if not pretrained_weights:
        return

    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key and checkpoint_key in state_dict:
        print(f"Using key '{checkpoint_key}' from checkpoint")
        state_dict = state_dict[checkpoint_key]

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        if key.startswith("head."):
            continue
        key = key.replace("backbone.", "")
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loaded backbone weights: missing={len(missing)}, unexpected={len(unexpected)}")


def activation_with_channels(model, img_tensor, target_layer, top_k=100, score_mode="mean"):
    """Collect a layer output and return top-k channel activation maps."""
    features = []

    def fwd_hook(module, input, output):
        features.append(output.detach())

    h = target_layer.register_forward_hook(fwd_hook)
    with torch.no_grad():
        model(img_tensor)

    h.remove()

    feat = F.relu(features[0][0])  # [C, Hf, Wf]
    flat_feat = feat.flatten(1)
    if score_mode == "max":
        scores = flat_feat.max(dim=1).values
    elif score_mode == "l2":
        scores = flat_feat.pow(2).mean(dim=1).sqrt()
    else:
        scores = flat_feat.mean(dim=1)

    k = feat.shape[0] if top_k < 0 else min(top_k, feat.shape[0])
    topk_scores, topk_idx = scores.topk(k)
    selected_maps = feat[topk_idx]

    selected_maps = F.interpolate(
        selected_maps.unsqueeze(0),
        size=img_tensor.shape[2:],
        mode="bilinear",
        align_corners=False,
    )[0]

    map_min = selected_maps.flatten(1).min(dim=1).values[:, None, None]
    map_max = selected_maps.flatten(1).max(dim=1).values[:, None, None]
    selected_maps = (selected_maps - map_min) / (map_max - map_min + 1e-8)

    return (
        selected_maps.cpu().numpy(),
        topk_idx.detach().cpu().numpy(),
        topk_scores.detach().cpu().numpy(),
    )


def save_per_channel_grid(selected_maps, topk_idx, topk_scores, img_np,
                          layer_name, output_dir, name, ncols=10):
    """Save per-channel activation maps as a grid image."""
    K = selected_maps.shape[0]
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes[None, :] if nrows == 1 else axes[:, None]
    fig.suptitle(f"Feature Maps: {layer_name} (top {K} by activation score)", fontsize=14)

    for i in range(K):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        heatmap_color = cv2.applyColorMap(np.uint8(255 * selected_maps[i]), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"ch{topk_idx[i]} score={topk_scores[i]:.6f}", fontsize=6)
        ax.axis("off")

    for i in range(K, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].axis("off")

    plt.tight_layout()
    save_path = f"{output_dir}/{name}_activation_{layer_name}_channels.jpg"
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {layer_name} feature-map grid -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--pretrained_weights", default="dino_output/resnet50/checkpoint0190.pth")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--image_path", default="img/test4.jpg")
    parser.add_argument("--output_dir", default="gradcam_maps")
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--top_k", default=100, type=int,
                        help="Number of top channels to show in per-channel grid. -1 for all.")
    parser.add_argument("--score_mode", default="mean", choices=["mean", "max", "l2"],
                        help="How to rank channels before visualization.")
    parser.add_argument("--ncols", default=10, type=int,
                        help="Number of columns in each feature-map grid.")
    args = parser.parse_args()

    # Load model
    model = models.__dict__[args.arch]()
    model.fc = torch.nn.Identity()
    load_backbone_weights(model, args.pretrained_weights, args.checkpoint_key)

    model.eval()
    print(f"Loaded {args.arch}")

    # Visualize the output feature maps of the four complete ResNet stages.
    target_layers = {
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
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
        img_tensor = tfm(img).unsqueeze(0)
        selected_maps, topk_idx, topk_scores = activation_with_channels(
            model, img_tensor, target_layer, top_k=args.top_k, score_mode=args.score_mode
        )
        save_per_channel_grid(
            selected_maps, topk_idx, topk_scores,
            img_np, layer_name, args.output_dir, name, ncols=args.ncols
        )


if __name__ == "__main__":
    main()
