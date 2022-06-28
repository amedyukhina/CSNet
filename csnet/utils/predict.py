import os
import warnings

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType
)
from skimage import io
from tqdm import tqdm

from .utils import get_device
from ..transforms.default import get_test_transform


def predict(input_dir, output_dir, model, roi_size, batch_size=2, return_last=False):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # setup test data
    test_images = sorted([os.path.join(input_dir, fn) for fn in os.listdir(input_dir)])
    test_data = [{"image": image} for image in test_images]
    transform = get_test_transform()
    ds = Dataset(data=test_data, transform=transform)
    dl = DataLoader(ds, batch_size=1, num_workers=4)

    model.to(device)
    model.eval()
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    with torch.no_grad():
        for test_data in tqdm(dl):
            test_inputs = test_data["image"].to(device)
            predicted = sliding_window_inference(test_inputs, roi_size, batch_size, model)
            predicted = post_pred(decollate_batch(predicted)[0])[1].cpu()
            fn = test_data['image_meta_dict']['filename_or_obj'][0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(fn.replace(input_dir.rstrip('/'), output_dir.rstrip('/')),
                          (predicted.numpy() * 255).astype(np.uint8))

    if return_last:
        return test_inputs[0][0].cpu(), predicted
