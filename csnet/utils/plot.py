import numpy as np
import pylab as plt
from skimage import io


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
            img = np.moveaxis(images[i][j].cpu().numpy(), 0, -1)
            if len(img.shape) > 3:
                img = img.max(0)
            io.imshow(img)
    plt.tight_layout()


def plot_projections(imgs, panel_size=3):
    imgs = [img.numpy() for img in imgs]

    fig, ax = plt.subplots(3, len(imgs), figsize=(2 * panel_size, 3 * panel_size))
    for i in range(3):
        for j in range(len(imgs)):
            plt.sca(ax[i, j])
            io.imshow(imgs[j].max(i))
    plt.tight_layout()
