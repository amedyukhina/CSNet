import os

import numpy as np
import pylab as plt
import torch
from skimage import io


def get_paired_file_list(dir1, dir2):
    files = os.listdir(dir1)
    files1 = [os.path.join(dir1, fn) for fn in files
              if fn in os.listdir(dir2)]
    files2 = [os.path.join(dir2, fn) for fn in files
              if fn in os.listdir(dir2)]
    return files1, files2


def show_image_grid(images, panel_size=3):
    """
    Display pytorch tensors as a grid.

    Parameters
    ----------
    images : tuple or list
        List of pytorch tensors to display as columns.
        Each tensor must have a shape BxCxZxYxX, where B is the batch size (number of rows);
            C is the number of channels; Z,Y,X are the three dimensions.
    panel_size : scalar
        Size of each panel

    """
    cols = len(images)
    rows = len(images[0])
    figure, ax = plt.subplots(nrows=rows, ncols=cols,
                              figsize=(panel_size * rows, panel_size * cols))
    for i in range(len(images)):
        for j in range(len(images[0])):
            plt.sca(ax[j, i])
            io.imshow(images[i][j].cpu().numpy().transpose(1, 2, 3, 0).max(0))
    plt.tight_layout()


def save_model(net, epoch, model_path, model_name):
    fn_out = os.path.join(model_path, model_name, rf"{model_name}_{epoch}.pkl")
    torch.save(net, fn_out)
    print(rf"Saved model to: {fn_out}")


def numeric_score(pred, gt):
    fp = float(np.sum((pred > 0) & (gt == 0)))
    fn = float(np.sum((pred == 0) & (gt > 0)))
    tp = float(np.sum((pred > 0) & (gt > 0)))
    tn = float(np.sum((pred == 0) & (gt == 0)))
    return fp, fn, tp, tn
