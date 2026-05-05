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

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

# 获取 torchvision 中所有可用的模型名称（小写开头、可调用），用于 --arch 参数校验
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    """构建 DINO 训练的全部命令行参数。"""
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # ==================== 模型参数 ====================
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base',
                 'xcit_small_12_p16', 'xcit_small_12_p8',
                 'xcit_medium_24_p16', 'xcit_medium_24_p8',
                 'xcit_large_24_p16', 'xcit_large_24_p8',
                 'deit_tiny', 'deit_small'] \
                + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    # DINO 头的输出维度，对于复杂大数据集建议用较大值（如 65536）
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    # 是否对 DINO 头最后一层做权重归一化，vit_base 建议开启，vit_small 可关闭以获得更好性能
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    # 教师网络 EMA 更新的基础动量，训练过程中余弦递增至 1
    # batch_size 较小时建议用更高值（如 0.9995）
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # ==================== 教师温度参数 ====================
    # 教师温度控制输出分布的锐度：温度越低分布越尖锐
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # ==================== 训练/优化参数 ====================
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    # ViT 训练时建议初始 weight_decay 较小，后期增大
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    # 梯度裁剪阈值，较大 ViT 建议用 0.3~1.0 防止梯度爆炸
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    # 训练初期冻结输出层的 epoch 数，有助于稳定训练
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    # 学习率按 batch_size 线性缩放，此处指定的值对应 batch_size=256
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    # 随机深度（Stochastic Depth）的丢弃概率，用于正则化
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # ==================== 多裁剪（Multi-Crop）参数 ====================
    # 全局裁剪的缩放范围，用于生成两个大视角裁剪
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    # 局部小裁剪的数量，设为 0 则禁用 multi-crop
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    # 局部裁剪的缩放范围
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    # 裁剪输出尺寸
    parser.add_argument('--global_crop_size', type=int, default=224,
        help="Global crop output size (default: 224). Use 512 for high-res training.")
    parser.add_argument('--local_crop_size', type=int, default=96,
        help="Local crop output size (default: 96).")
    # ==================== 杂项 ====================
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    """DINO 训练主函数：初始化数据、模型、优化器，然后执行训练循环。"""
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ 准备数据 ============
    # 构造多裁剪数据增强：2 个全局裁剪 + local_crops_number 个局部裁剪
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.global_crop_size,
        args.local_crop_size,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    # 分布式采样器，确保每个 GPU 看到不同的数据子集
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # 丢弃最后不完整的 batch，保证分布式训练一致性
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ 构建学生和教师网络 ============
    # DeiT 和 ViT 是同一架构，统一名称
    args.arch = args.arch.replace("deit", "vit")

    # 根据架构类型选择不同的模型构建方式
    # 1) ViT 系列（vit_tiny/vit_small/vit_base）—— 来自本地 vision_transformer.py
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # 随机深度，仅学生使用
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # 2) XCiT 系列 —— 通过 torch.hub 从外部仓库加载（需要网络）
    elif args.arch.startswith("xcit"):
        # 仅 rank 0 下载，避免多进程并发 makedirs 冲突
        if args.rank == 0:
            torch.hub.load('facebookresearch/xcit:main', args.arch,
                           pretrained=False, drop_path_rate=args.drop_path_rate)
        if utils.is_dist_avail_and_initialized():
            dist.barrier()
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # 3) torchvision CNN（如 resnet50）—— 从 torchvision.models 加载
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]  # CNN 从全连接层获取特征维度
    else:
        print(f"Unknow architecture: {args.arch}")

    # MultiCropWrapper 统一封装 backbone + DINOHead，这是兼容不同架构的关键：
    #
    #   ViT:   backbone.head = Identity()   embed_dim = 384 (vit_small)
    #   CNN:   backbone.fc  = Identity()    embed_dim = 2048 (resnet50)
    #   XCiT:  backbone.head = Identity()   embed_dim = 384 (xcit_small)
    #
    #   MultiCropWrapper.__init__ 中同时执行了:
    #     backbone.fc, backbone.head = nn.Identity(), nn.Identity()
    #   ViT 有 .head 无 .fc，CNN 有 .fc 无 .head，XCiT 有 .head 无 .fc。
    #   因为两个属性都替换了，哪个存在就替换哪个（不存在的属性赋值后也没影响），
    #   所以无论哪种架构，backbone 都变成了纯粹的"特征提取器"。
    #
    #   之后 MultiCropWrapper.forward 把不同分辨率的裁剪按形状分组 batch 送入 backbone，
    #   拼接所有特征后统一送入 DINOHead。这样 ViT/CNN/XCiT 的差异在 Wrapper 之内
    #   就被消除了，外部（损失函数、训练循环）完全不感知底层架构。
    #
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # 将网络移至 GPU
    student, teacher = student.cuda(), teacher.cuda()

    # 如果网络中存在 BatchNorm 层，需要转换为 SyncBatchNorm 以支持分布式训练
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # SyncBN 需要 DDP 包装才能跨进程同步
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # 无 BN 层时，teacher_without_ddp 和 teacher 是同一个对象
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # 教师网络从学生网络的权重初始化（不是随机初始化）
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # 教师网络不需要梯度，仅通过 EMA 从学生更新
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ 准备损失函数 ============
    # ncrops = 2 个全局裁剪 + local_crops_number 个局部裁剪
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ 准备优化器 ============
    # get_params_groups 将参数分为需正则化和不需正则化两组（bias 和 Norm 不做权重衰减）
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # 适用于 CNN 和大 batch
    # 混合精度训练的 GradScaler
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ 初始化学习率/权重衰减/动量的调度器 ============
    # 学习率按 batch_size 线性缩放（linear scaling rule）
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    # 权重衰减从 weight_decay 余弦变化到 weight_decay_end
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # 教师 EMA 动量从 momentum_teacher 余弦递增至 1（训练后期教师趋于稳定）
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ 可选：从 checkpoint 恢复训练 ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    # ============ 训练循环 ============
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        # 每个 epoch 设置不同的随机种子，保证每轮数据顺序不同
        data_loader.sampler.set_epoch(epoch)

        # 训练一个 epoch
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # 保存 checkpoint（包含学生、教师、优化器、损失函数等完整状态）
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        # 始终保存最新 checkpoint（覆盖式），用于恢复训练
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        # 每隔 saveckp_freq 个 epoch 保存一份带编号的 checkpoint
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        # 写入训练日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    """执行 DINO 的单个训练 epoch。

    核心流程：
    1. 学生处理所有裁剪（全局+局部），教师只处理 2 个全局裁剪
    2. 计算学生-教师之间的交叉熵损失
    3. 反向传播更新学生参数
    4. 用 EMA 更新教师参数（不经过梯度）
    """
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # 按迭代步数更新学习率和权重衰减（per-iteration schedule）
        it = len(data_loader) * epoch + it  # 全局迭代步数
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # 仅第一组参数（非 bias/Norm 的权重）做权重衰减
                param_group["weight_decay"] = wd_schedule[it]

        # 将图像列表移至 GPU（每个图像是不同裁剪，形状可能不同）
        images = [im.cuda(non_blocking=True) for im in images]

        # 前向传播 + 计算 DINO 损失
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # 教师只处理前 2 个全局裁剪
            teacher_output = teacher(images[:2])
            # 学生处理所有裁剪（2 全局 + N 局部）
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        # 损失值异常时立即停止训练
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # 反向传播 + 梯度裁剪 + 参数更新
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            # 全精度训练路径
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            # 训练初期冻结 DINO 头的最后一层
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            # 混合精度训练路径：先 scale 梯度，反向传播后再 unscale 裁剪
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA 更新教师网络（不经过梯度，直接用学生参数的滑动平均）
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # 记录训练指标
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # 跨进程汇总指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    """DINO 损失函数：学生输出与教师输出之间的交叉熵。

    教师输出经过 centering（减去均值中心）和 sharpening（除以温度），
    然后用 softmax 产生软标签。学生的每个视角与教师的每个视角配对计算损失，
    跳过学生和教师使用同一视角的情况。

    center 通过 EMA 在各进程间同步更新，防止教师输出坍缩。
    """

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        # center 向量，用于教师输出的中心化，防止模式坍缩
        self.register_buffer("center", torch.zeros(1, out_dim))
        # 教师温度调度：前 warmup_teacher_temp_epochs 线性增加，之后保持恒定
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        计算学生-教师之间的交叉熵损失。

        Args:
            student_output: 学生对所有裁剪的输出 (ncrops * batch_size, out_dim)
            teacher_output: 教师对 2 个全局裁剪的输出 (2 * batch_size, out_dim)
            epoch: 当前 epoch，用于查找教师温度
        """
        # 学生输出除以学生温度后分块（每个裁剪一块）
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # 教师输出：先 centering 再 sharpening，然后 softmax
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # 分为 2 个全局视角，detach 防止梯度回传

        # 对每个教师视角和每个学生视角（跳过同一视角）计算交叉熵
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # 跳过学生和教师使用同一视角的情况
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        # 更新 center 向量
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """用 EMA 更新教师输出的中心向量。

        先跨进程汇总教师输出的均值，再用 EMA 更新 center。
        center 的作用是防止教师输出坍缩到单一模式。
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # 跨所有进程求和
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # EMA 更新 center
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    """DINO 的多裁剪数据增强策略。

    对每张图像生成：
    - 2 个全局裁剪（224x224）：分别使用不同的增强流水线
    - local_crops_number 个局部小裁剪（96x96）：使用相同的增强流水线

    这些裁剪作为学生的输入（全部）和教师的输入（仅全局裁剪）。
    """

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,
                 global_crop_size=224, local_crop_size=96):
        # 显微图增强：保留几何裁剪和翻转，弱化颜色扰动以保护低对比细节。
        flip_and_weak_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)],
                p=0.3
            ),
        ])
        # 标准化（ImageNet 均值和标准差）
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 全局裁剪仅保留低概率轻微模糊，避免抹掉划痕、孔洞和细晶界。
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_weak_color_jitter,
            utils.GaussianBlur(0.1),
            normalize,
        ])
        # 第二个全局裁剪去掉 Solarization，保留同样的弱增强策略。
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_weak_color_jitter,
            utils.GaussianBlur(0.1),
            normalize,
        ])
        # 局部小裁剪用于学习微小缺陷和局部组织，不做模糊。
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_weak_color_jitter,
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
