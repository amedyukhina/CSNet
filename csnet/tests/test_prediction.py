import os

from ..utils.predict import predict


def test_prediction(data_paths, unet_setup, output_path):
    dir_img, _ = data_paths
    model, _, _ = unet_setup
    predict(dir_img, output_path, model, roi_size=(16, 64, 64))
    assert len(os.listdir(dir_img)) == len(os.listdir(output_path))
