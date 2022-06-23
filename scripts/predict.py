import argparse
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import DataLoader
from tqdm import tqdm

from csnet.dataset import CS_Dataset
from csnet.utils import get_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str,
                        help='Input directory with images to predict', required=True)
    parser.add_argument('-r', '--ref-dir', type=str,
                        help='Reference directory with the ground truth, optional',
                        default=None)
    parser.add_argument('-o', '--output-dir', type=str,
                        help='Directory to save predicted images', required=True)
    parser.add_argument('-m', '--model-path', type=str,
                        help='Model name', required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='Batch size')

    args = parser.parse_args()
    if args.ref_dir is None:
        args.ref_dir = args.input_dir

    print('\nThe following are the parameters that will be used:')
    print(vars(args))
    print('\n')

    ds = CS_Dataset(args.input_dir, args.ref_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=False)
    device = get_device()
    net = torch.load(args.model_path).to(device)

    imgs = []
    lbls = []
    predictions = []

    with torch.no_grad():
        net.eval()
        for idx, batch in enumerate(dl):
            image = batch[0].to(device)
            label = batch[1].to(device)
            pred = net(image)

            imgs.append(image)
            lbls.append(label)
            predictions.append(torch.argmax(pred, dim=1))
    imgs = torch.concat(imgs)
    lbls = torch.concat(lbls)
    predictions = torch.concat(predictions)

    os.makedirs(args.output_dir, exist_ok=True)
    for pred, fn in zip(tqdm(predictions), ds.image_fns):
        io.imsave(fn.replace(args.input_dir.rstrip('/'), args.output_dir.rstrip('/')),
                  (pred.cpu().numpy() * 255).astype(np.uint8))
