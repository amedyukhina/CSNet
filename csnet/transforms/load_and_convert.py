from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged
)


def get_LoadAndConvertTransform(image_key='image', label_key='label'):
    if label_key is not None:
        keys = [image_key, label_key]
        scale = Compose([
            ScaleIntensityRanged(keys=[image_key], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=[label_key], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
        ])
    else:
        keys = [image_key]
        scale = ScaleIntensityRanged(keys=[image_key], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True)

    transform = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="SPL"),
            scale
        ]
    )
    return transform
