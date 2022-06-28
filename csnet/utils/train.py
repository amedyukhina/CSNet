import os

import numpy as np
import torch
import wandb
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

from ..transforms.postroc import get_PostprocTransforms
from ..utils import get_device, save_model


def __get_batch_data(batch_data, image_key='image', label_key='label', device=None):
    if device is None:
        device = get_device()
    return batch_data[image_key].to(device), batch_data[label_key].to(device)


def __detatch_norm_project(img):
    img = img.detach().cpu().numpy()
    img = img - np.min(img)
    img = img * 255. / img.max()
    if len(img.shape) > 2:
        img = img.max(0)
    return img.astype(np.uint8)


def __log_images(writer, input_img, output_img, target_img, iteration):
    input_img, output_img, target_img = [__detatch_norm_project(img)
                                         for img in [input_img, output_img, target_img]]
    writer.add_image('input', input_img, iteration, dataformats='HW')
    writer.add_image('output', output_img, iteration, dataformats='HW')
    writer.add_image('target', target_img, iteration, dataformats='HW')


def train(train_dl, val_dl, model, loss_function, metric, config, log_tensorboard=False):
    device = get_device()
    model.to(device)
    best_metric = -1
    post_pred, post_label = get_PostprocTransforms()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=config.factor,
                                                              patience=config.patience)
    tbwriter = None
    if log_tensorboard:
        tbwriter = SummaryWriter(log_dir=os.path.join(config.model_path, 'logs'))

    for epoch in range(config.epochs):

        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_dl:
            step += 1
            inputs, labels = __get_batch_data(batch_data, device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            wandb.log({'training loss': loss.item()})
            if log_tensorboard:
                tbwriter.add_scalar('training loss', loss.item(), step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} training loss: {epoch_loss:.4f}")
        wandb.log({'average training loss': epoch_loss,
                   'epoch': epoch + 1,
                   'lr': optimizer.param_groups[0]['lr']})

        if log_tensorboard:
            tbwriter.add_scalar('average training loss', epoch_loss, epoch + 1)
            tbwriter.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch + 1)

        model.eval()
        epoch_loss = 0
        step = 0
        with torch.no_grad():
            for val_data in val_dl:
                step += 1
                val_inputs, val_labels = __get_batch_data(val_data, device=device)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, config.roi_size, sw_batch_size, model)
                val_loss = loss_function(val_outputs, val_labels)
                epoch_loss += val_loss.item()

                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                # compute metric for current iteration
                metric(y_pred=val_outputs, y=val_labels)

            epoch_loss /= step
            # aggregate the final mean dice result
            val_metric = metric.aggregate().item()
            # reset the status for next validation round
            metric.reset()
        print(f"epoch {epoch + 1} validation loss: {epoch_loss:.4f}; {config.metric_name}: {val_metric:.4f}")
        wandb.log({'validation loss': epoch_loss,
                   config.metric_name: val_metric})

        if log_tensorboard:
            tbwriter.add_scalar('validation loss', epoch_loss, epoch + 1)
            tbwriter.add_scalar(config.metric_name, val_metric, epoch + 1)
            __log_images(tbwriter, val_inputs[0][0], val_outputs[0][1], val_labels[0][1], epoch + 1)

        lr_scheduler.step(epoch_loss)

        if val_metric > best_metric:
            best_metric = val_metric
            save_model(model, config.model_path)
