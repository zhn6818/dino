"""
Grad-CAM visualization for DINO-trained ResNet50.
Usage:
    python visualize_resnet.py --image_path img/test.jpg
    python visualize_resnet.py --image_path img/test.jpg --checkpoint_key teacher
"""
import argparse
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image


def gradcam(model, img_tensor, target_layer):
    """Compute Grad-CAM heatmap."""
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

    feat = features[0]
    grad = grads[0]
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.squeeze().detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--pretrained_weights", default="dino_output/resnet50/checkpoint.pth")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--image_path", default="img/test4.jpg")
    parser.add_argument("--output_dir", default="gradcam_maps")
    parser.add_argument("--image_size", default=224, type=int)
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

    # Grad-CAM on last conv layer
    target_layer = model.layer4[-1].conv3

    tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = Image.open(args.image_path).convert("RGB")
    img_np = np.array(img.resize((args.image_size, args.image_size)))
    img_tensor = tfm(img).unsqueeze(0).requires_grad_(True)

    heatmap = gradcam(model, img_tensor, target_layer)

    # Overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image_path))[0]
    cv2.imwrite(f"{args.output_dir}/{name}_gradcam.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved to {args.output_dir}/{name}_gradcam.jpg")


if __name__ == "__main__":
    main()
