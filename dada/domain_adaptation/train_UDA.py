# --------------------------------------------------------
# Domain adaptation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from advent.domain_adaptation.train_UDA import print_losses, log_losses_tensorboard
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss, prob_2_entropy

from dada.utils.func import loss_calc_depth
from dada.utils.viz_segmask import colorize_mask


def train_dada(model, trainloader, targetloader, cfg):
    """ UDA training with dada
    """
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(
        model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
        lr=cfg.TRAIN.LEARNING_RATE,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(
        d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D, betas=(0.9, 0.99)
    )

    # interpolate output segmaps
    interp = nn.Upsample(
        size=(input_size_source[1], input_size_source[0]),
        mode="bilinear",
        align_corners=True,
    )
    interp_target = nn.Upsample(
        size=(input_size_target[1], input_size_target[0]),
        mode="bilinear",
        align_corners=True,
    )

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP+1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, depth, _, _ = batch
        _, pred_src_main, pred_depth_src_main = model(images_source.cuda(device))
        pred_src_main = interp(pred_src_main)
        pred_depth_src_main = interp(pred_depth_src_main)
        loss_depth_src_main = loss_calc_depth(pred_depth_src_main, depth, device)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = ( cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_DEPTH_MAIN * loss_depth_src_main)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        _, pred_trg_main, pred_depth_trg_main = model(images.cuda(device))
        pred_trg_main = interp_target(pred_trg_main)
        pred_depth_trg_main = interp_target(pred_depth_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_depth_trg_main)
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        pred_src_main = pred_src_main.detach()
        pred_depth_src_main = pred_depth_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)) * pred_depth_src_main)
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main
        loss_d_main.backward()

        # train with target
        pred_trg_main = pred_trg_main.detach()
        pred_depth_trg_main = pred_depth_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)) * pred_depth_trg_main)
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_main.step()

        current_losses = {
            "loss_seg_src_main": loss_seg_src_main,
            "loss_depth_src_main": loss_depth_src_main,
            "loss_adv_trg_main": loss_adv_trg_main,
            "loss_d_main": loss_d_main,
        }
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print("taking snapshot ...")
            print("exp =", cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f"model_{i_iter}.pth")
            torch.save(d_main.state_dict(), snapshot_dir / f"model_{i_iter}_D_main.pth")
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, "T")
                draw_in_tensorboard(
                    writer, images_source, i_iter, pred_src_main, num_classes, "S"
                )


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f"Image - {type_}", grid_image, i_iter)

    softmax = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    mask = colorize_mask(num_classes, np.asarray(np.argmax(softmax, axis=2), dtype=np.uint8)).convert("RGB")
    grid_image = make_grid(torch.from_numpy(np.array(mask).transpose(2, 0, 1)),
                           3,
                           normalize=False,
                           range=(0, 255))
    writer.add_image(f"Prediction - {type_}", grid_image, i_iter)

def train_domain_adaptation_with_depth(model, trainloader, targetloader, cfg):
    assert cfg.TRAIN.DA_METHOD in {"DADA"}, "Not yet supported DA method {}".format(cfg.TRAIN.DA_METHOD)
    train_dada(model, trainloader, targetloader, cfg)