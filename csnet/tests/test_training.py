import argparse
import os

import wandb

from ..utils.train import train


def test_training(loaders, config, unet_setup):
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project='', config=config)

    config = argparse.Namespace(**config)
    train_dl, val_dl = loaders
    model, loss_function, metric_function = unet_setup

    train(train_dl, val_dl, model, loss_function, metric_function, config, log_tensorboard=True)
    wandb.finish()
    assert os.path.exists(os.path.join(config.model_path, 'best_model.pth'))
