import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models as torchvision_models
from torchvision import transforms as T
from PIL import Image
import numpy as np

import utils


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, file_list, transform_img=None, target_size=(512, 512)):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.transform_img = transform_img
        self.target_size = target_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, name + '.jpg') if os.path.exists(os.path.join(self.img_dir, name + '.jpg')) else os.path.join(self.img_dir, name + '.png')
        img = Image.open(img_path).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, name + '.png'))

        img = img.resize(self.target_size, Image.BILINEAR)
        label = label.resize(self.target_size, Image.NEAREST)

        if self.transform_img:
            img = self.transform_img(img)

        label = torch.from_numpy(np.array(label)).long()
        return img, label


class SegmentationHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)


class ResNet50Features(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


def build_model(pretrained_weights, checkpoint_key, num_classes=3, freeze_backbone=True):
    backbone = torchvision_models.resnet50()
    backbone.fc = nn.Identity()
    if pretrained_weights and os.path.exists(pretrained_weights):
        utils.load_pretrained_weights(backbone, pretrained_weights, checkpoint_key, 'resnet50', 16)
        print(f"Loaded weights from {pretrained_weights}")
    else:
        print("Using random initialization")

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    feature_extractor = ResNet50Features(backbone)
    seg_head = SegmentationHead(in_channels=2048, num_classes=num_classes)
    return feature_extractor, seg_head


def compute_miou(preds, labels, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def train_one_epoch(feature_extractor, seg_head, loader, optimizer, device):
    seg_head.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            features = feature_extractor(imgs)
        out = seg_head(features)
        out = F.interpolate(out, size=labels.shape[1:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(feature_extractor, seg_head, loader, device, num_classes=3):
    seg_head.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        features = feature_extractor(imgs)
        out = seg_head(features)
        out = F.interpolate(out, size=labels.shape[1:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    miou = compute_miou(all_preds, all_labels, num_classes)
    # per-class IoU
    class_ious = []
    for cls in range(num_classes):
        pred_cls = (all_preds == cls)
        label_cls = (all_labels == cls)
        inter = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()
        class_ious.append((inter / union).item() if union > 0 else 0.0)
    return total_loss / len(loader.dataset), miou, class_ious


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # prepare data
    img_dir = os.path.join(args.data_dir, 'img')
    label_dir = os.path.join(args.data_dir, 'label')
    all_names = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    all_names = [n for n in all_names if os.path.exists(os.path.join(label_dir, n + '.png'))]

    # detect num_classes from labels
    all_unique = set()
    for n in all_names:
        arr = np.array(Image.open(os.path.join(label_dir, n + '.png')))
        all_unique.update(np.unique(arr).tolist())
    num_classes = max(all_unique) + 1
    print(f"Detected classes: {sorted(all_unique)}, num_classes={num_classes}")

    random.seed(42)
    random.shuffle(all_names)
    split = int(len(all_names) * 0.8)
    train_names, val_names = all_names[:split], all_names[split:]
    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Total: {len(all_names)}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = SegmentationDataset(img_dir, label_dir, train_names, transform, target_size=(args.input_size, args.input_size))
    val_ds = SegmentationDataset(img_dir, label_dir, val_names, transform, target_size=(args.input_size, args.input_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # build model
    feature_extractor, seg_head = build_model(args.pretrained_weights, args.checkpoint_key, num_classes=num_classes, freeze_backbone=True)
    feature_extractor = feature_extractor.to(device)
    seg_head = seg_head.to(device)

    optimizer = torch.optim.SGD(seg_head.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_miou = 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(feature_extractor, seg_head, train_loader, optimizer, device)
        scheduler.step()
        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            val_loss, miou, class_ious = evaluate(feature_extractor, seg_head, val_loader, device)
            best_miou = max(best_miou, miou)
            print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"mIoU={miou:.4f} (best={best_miou:.4f}) class_ious={[f'{c:.3f}' for c in class_ious]}")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={train_loss:.4f}")

    print(f"\nFinal best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Segmentation evaluation with frozen DINO features')
    parser.add_argument('--data_dir', default='JLD', type=str)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--val_freq', default=5, type=int)
    args = parser.parse_args()
    main(args)
