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
    features, grads = [], []  # 用于存储前向传播的特征图和反向传播的梯度

    def fwd_hook(module, input, output):  # 前向 hook：目标层前向传播时捕获并保存该层的输出（特征图）
        features.append(output)

    def bwd_hook(module, grad_input, grad_output):  # 反向 hook：目标层反向传播时捕获并保存该层输出的梯度
        grads.append(grad_output[0])

    h = target_layer.register_forward_hook(fwd_hook)  # 在目标卷积层上注册前向 hook，用于捕获特征图
    h2 = target_layer.register_full_backward_hook(bwd_hook)  # 在目标卷积层上注册反向 hook，用于捕获梯度

    output = model(img_tensor)  # 前向传播：将图像输入模型，得到分类 logits
    output[:, output.argmax(dim=1).item()].backward()  # 对预测概率最高的类别的 logit 执行反向传播，计算目标层梯度

    h.remove()  # 反向传播结束后移除前向 hook，避免内存泄漏
    h2.remove()  # 移除反向 hook

    feat = features[0]  # 取出 hook 捕获的目标层特征图，形状 [1, C, H, W]
    grad = grads[0]  # 取出 hook 捕获的目标层梯度，形状 [1, C, H, W]
    weights = grad.mean(dim=[2, 3], keepdim=True)  # 对每个通道的梯度在空间维度 (H,W) 上取全局平均，得到通道重要性权重，形状 [1, C, 1, 1]
    cam = (weights * feat).sum(dim=1, keepdim=True)  # 将权重与特征图逐通道相乘后求和，得到 Grad-CAM 热力图，形状 [1, 1, H, W]
    cam = F.relu(cam)  # 取 ReLU，只保留对目标类别有正向贡献的区域
    cam = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)  # 双线性插值到输入图像尺寸，便于叠加显示
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化到 [0, 1] 范围，便于可视化
    return cam.squeeze().detach().cpu().numpy()  # 移除 batch/channel 维度，脱离计算图，转为 numpy 数组返回


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--pretrained_weights", default="dino_output/resnet50_gc512_lc256/checkpoint.pth")
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--image_path", default="img/test5.jpg")
    parser.add_argument("--output_dir", default="gradcam_maps")
    parser.add_argument("--image_size", default=1024, type=int)
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
        img_tensor = tfm(img).unsqueeze(0).requires_grad_(True)  # 每个 layer 需要独立的输入张量
        heatmap = gradcam(model, img_tensor, target_layer)

        # 将热力图叠加到原图上
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)

        save_path = f"{args.output_dir}/{name}_gradcam_{layer_name}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved {layer_name} -> {save_path}")


if __name__ == "__main__":
    main()
