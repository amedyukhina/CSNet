import numpy as np
import torch

from .utils import numeric_score


def model_eval(net, dl, wce_loss_weight, dice_loss_weight, wce_loss, dice_loss):
    net.eval()
    tp = tn = fp = fn = 0
    val_loss = []
    with torch.no_grad():
        for idx, batch in enumerate(dl):
            image = batch[0].cuda()
            label = batch[1].cuda()
            pred = net(image)
            loss = (wce_loss_weight * wce_loss(pred, label.squeeze(1))
                    + dice_loss_weight * dice_loss(pred, label))
            val_loss.append(loss.item())

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
    return np.mean(val_loss), recall, precision, iou
