import datetime

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from .dataset import CS_Dataset
from .losses import WeightedCrossEntropyLoss, DiceLoss
from .model import CSNet3D
from .utils import numeric_score
from .utils import save_model, get_device


def __setup_trainig(dir_train_img, dir_train_gt, dir_val_img, dir_val_gt, config):
    device = get_device()

    ds = CS_Dataset(dir_train_img, dir_train_gt)
    dl = DataLoader(ds, batch_size=config.batch_size,
                    num_workers=config.batch_size, shuffle=True)

    ds_val = CS_Dataset(dir_val_img, dir_val_gt)
    dl_val = DataLoader(ds_val, batch_size=config.batch_size,
                        num_workers=config.batch_size, shuffle=False)

    net = CSNet3D(classes=2, channels=1).to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=config.factor,
                                                              patience=config.patience)

    wce_loss = WeightedCrossEntropyLoss().to(device)
    dice_loss = DiceLoss().to(device)
    return dl, dl_val, net, optimizer, lr_scheduler, wce_loss, dice_loss, device


def __get_model_name(log_progress):
    if log_progress:
        model_name = wandb.run.name
    else:
        model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return model_name


def train(dir_train_img, dir_train_gt, dir_val_img, dir_val_gt,
          config, log_progress=False):
    dl, dl_val, net, optimizer, lr_scheduler, \
    wce_loss, dice_loss, device = __setup_trainig(dir_train_img, dir_train_gt,
                                                  dir_val_img, dir_val_gt, config)
    model_name = __get_model_name(log_progress)

    for epoch in range(config.epochs):
        net.train()
        tr_loss = []
        for idx, batch in enumerate(dl):
            image = batch[0].to(device)
            label = batch[1].to(device)
            optimizer.zero_grad()
            pred = net(image)
            loss = (config.wce_loss_weight * wce_loss(pred, label.squeeze(1))
                    + config.dice_loss_weight * dice_loss(pred, label))
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())
        tr_loss = np.mean(tr_loss)
        val_loss = get_val_loss(net, dl_val, config, wce_loss, dice_loss, device)
        lr_scheduler.step(val_loss)

        print(rf"Epoch {epoch + 1}, training loss: {tr_loss}")
        wandb.log({'training loss': tr_loss,
                   'epoch': epoch + 1,
                   'lr': optimizer.param_groups[0]['lr']})

        if (epoch + 1) % config.val_step == 0:
            recall, precision, iou = get_val_accuracy(net, dl_val, device)
            print(rf"Epoch {epoch + 1}, val loss: {val_loss};"
                  rf" recall: {recall}; precision: {precision}; IOU: {iou}")
            wandb.log({'validation loss': val_loss,
                       'Recall': recall,
                       'Precision': precision,
                       'IOU': iou})

        if (epoch + 1) % config.snapshot == 0:
            save_model(net, epoch + 1, config.model_path, model_name)


def get_val_loss(net, dl, config, wce_loss, dice_loss, device=None):
    if device is None:
        device = get_device()
    net.eval()
    val_loss = []
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            image = batch[0].to(device)
            label = batch[1].to(device)
            pred = net(image)
            loss = (config.wce_loss_weight * wce_loss(pred, label.squeeze(1))
                    + config.dice_loss_weight * dice_loss(pred, label))
            val_loss.append(loss.item())
    return np.mean(val_loss)


def get_val_accuracy(net, dl, device=None):
    if device is None:
        device = get_device()
    net.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            image = batch[0].to(device)
            label = batch[1].to(device)
            pred = net(image)

            for i in range(len(pred)):
                FP, FN, TP, TN = numeric_score(torch.argmax(pred, dim=1).cpu().numpy()[i],
                                               label.cpu().numpy()[i])
                tp += TP
                fp += FP
                fn += FN
                tn += TN

    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    iou = tp / (tp + fn + fp + 1e-10)
    return recall, precision, iou
