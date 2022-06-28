from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType
)


def get_PostprocTransforms():
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    return post_pred, post_label
