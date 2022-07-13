import numpy as np
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    EnsureTyped,
    RandAffined
)

from .load_and_convert import get_LoadAndConvertTransform


def get_default_train_transforms(roi_size, return_val=True, image_key='image', label_key='label', instance=False):
    keys = [image_key, label_key]

    default_train_transforms = Compose(
        [
            get_LoadAndConvertTransform(image_key, label_key, instance=instance),
            RandCropByPosNegLabeld(
                keys=keys, label_key=label_key, image_key=image_key, spatial_size=roi_size,
                pos=1, neg=1, num_samples=4, image_threshold=0,
            ),
            RandAffined(
                keys=keys, mode=('bilinear', 'nearest'), prob=1.0,
                rotate_range=(0, 0, np.pi / 15), scale_range=(0.1, 0.1, 0.1)
            ),
            EnsureTyped(keys=keys),
        ]
    )

    default_val_transforms = Compose(
        [
            get_LoadAndConvertTransform(image_key, label_key, instance=instance),
            RandCropByPosNegLabeld(
                keys=keys, label_key=label_key, image_key=image_key, spatial_size=roi_size,
                pos=1, neg=1, num_samples=4, image_threshold=0,
            ),
            EnsureTyped(keys=keys),
        ]
    )
    if return_val:
        return default_train_transforms, default_val_transforms
    else:
        return default_train_transforms


def get_test_transform(image_key='image'):
    transform = Compose(
        [
            get_LoadAndConvertTransform(image_key, label_key=None),
            EnsureTyped(keys=[image_key]),

        ]
    )
    return transform
