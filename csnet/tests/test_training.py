import argparse
import os

import wandb

from ..train import train


def test_training(data_paths, model_path):
    dir_img, dir_gt = data_paths
    config = dict(
        epochs=2,
        batch_size=2,
        lr=0.0001,
        weight_decay=0.0005,
        wce_loss_weight=0.6,
        dice_loss_weight=0.4,
        factor=0.1,
        patience=2,
        snapshot=1,
        val_step=2,
        model_path=model_path
    )
    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project='', config=config)
    config = argparse.Namespace(**config)
    train(dir_img, dir_gt, dir_img, dir_gt, config, log_progress=False)
    wandb.finish()
    models = os.listdir(config.model_path)
    assert len(models) > 0
    assert os.path.exists(os.path.join(config.model_path, models[0],
                                       rf"{models[0]}_{config.epochs}.pkl"))
