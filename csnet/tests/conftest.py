import os
import shutil
import tempfile

import pytest
from cs_sim.batch.batch_corrupt import batch_corrupt_image
from cs_sim.batch.batch_synth import batch_generate_img_with_lines


@pytest.fixture(scope='module')
def data_paths():
    path = tempfile.mkdtemp()
    dir_gt = os.path.join(path, 'gt')
    dir_img = os.path.join(path, 'img')
    os.makedirs(dir_gt, exist_ok=True)
    os.makedirs(dir_img, exist_ok=True)

    batch_generate_img_with_lines(n_img=10, n_jobs=1, dir_out=dir_gt,
                                  imgshape=(16, 64, 64), n_lines=10, maxval=255)
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
