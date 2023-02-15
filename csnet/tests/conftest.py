import os
import shutil
import tempfile

import pytest
from cs_sim.batch.batch_corrupt import batch_corrupt_image
from cs_sim.batch.batch_synth import batch_generate_img_with_filaments
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from ..transforms.default import get_default_train_transforms
from ..utils import get_data_dict
from ..utils import get_device


@pytest.fixture(scope='module')
def data_paths():
    path = tempfile.mkdtemp()
    dir_gt = os.path.join(path, 'gt')
    dir_img = os.path.join(path, 'img')
    os.makedirs(dir_gt, exist_ok=True)
    os.makedirs(dir_img, exist_ok=True)

    batch_generate_img_with_filaments(n_img=10, n_jobs=1, dir_out=dir_gt,
                                      imgshape=(16, 64, 64), n_filaments=10, maxval=255)
    batch_corrupt_image(dir_gt, dir_img, n_jobs=1,
                        corruption_steps=[
                            ('poisson_noise', {'snr': 2}),
                            ('convolve', {'sigma': 2}),
                            ('gaussian_noise', {'snr': 50})
                        ])

    yield dir_img, dir_gt
    shutil.rmtree(path)


@pytest.fixture(scope='module')
def model_path():
    path = tempfile.mkdtemp()
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope='module')
def output_path():
    path = tempfile.mkdtemp()
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope='module')
def loaders(data_paths):
    train_files = get_data_dict(*data_paths)
    val_files = train_files
    train_tr, val_tr = get_default_train_transforms(roi_size=(16, 64, 64))
    tr_ds = CacheDataset(data=train_files, transform=train_tr, cache_rate=1, num_workers=4)
    tr_dl = DataLoader(tr_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_files, transform=val_tr, cache_rate=1.0, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=4)
    return tr_dl, val_dl


@pytest.fixture(scope='module')
def config(model_path):
    return dict(
        epochs=2,
        batch_size=2,
        lr=0.0001,
        weight_decay=0.0005,
        factor=0.1,
        patience=2,
        roi_size=(16, 64, 64),
        model_path=model_path,
        metric_name='Dice Metric'
    )


@pytest.fixture(scope='module')
def unet_setup():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(get_device())

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    return model, loss_function, dice_metric
