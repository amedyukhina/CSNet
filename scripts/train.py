import argparse
import json
import os

import numpy as np
import wandb
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from csnet.models.csnet import CSNet
from csnet.models.csnet_orig import CSNetOrig
from csnet.transforms.default import get_default_train_transforms
from csnet.utils import get_data_dict, get_model_name
from csnet.utils.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
                        help='Directory with the data (training, validation, test)', required=True)
    parser.add_argument('-t', '--train-dirname', type=str, default='train',
                        help='Subdirectory within "data-dir" to use for training')
    parser.add_argument('-v', '--val-dirname', type=str, default='val',
                        help='Subdirectory within "data-dir" to use for validation')
    parser.add_argument('-im', '--img-dirname', type=str, default='img',
                        help='Subdirectory within "data-dir/train-dir" with input images')
    parser.add_argument('-gt', '--gt-dirname', type=str, default='gt',
                        help='Subdirectory within "data-dir/train-dir" with ground truth predictions')
    parser.add_argument('-m', '--model', type=str, default='unet',
                        help='Model type to use for prediction, one of ["unet", "csnet", "csnet_orig"]')
    parser.add_argument('-c', '--n-channels', type=str, default="16,32,64,128",
                        help='Number of channels in each UNet layers, separated by ","')
    parser.add_argument('-r', '--num-res-units', type=int, default=1,
                        help='Number of residual units in each block of Unet and CSNet')
    parser.add_argument('-mp', '--model-path', type=str,
                        help='Directory for model checkpoints', default='model')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('-lr', '--lr', type=float, default=0.0001, help='Starting learning rate')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0005,
                        help='Weight decay for Adam optimizer')
    parser.add_argument('-f', '--factor', type=float, default=0.1,
                        help='Factor parameter for ReduceOnPlateau learning rate scheduler')
    parser.add_argument('-p', '--patience', type=int, default=10,
                        help='Patience parameter for ReduceOnPlateau learning rate scheduler')
    parser.add_argument('-pr', '--wandb-project', type=str, default='',
                        help='wandb project name')
    parser.add_argument('-log', '--log-progress', action='store_true')

    config = parser.parse_args()
    config.n_channels = np.int_(config.n_channels.split(','))

    print('\nThe following are the parameters that will be used:')
    print(vars(config))
    print('\n')

    # Initialize wandb project
    if config.log_progress:
        with open('/home/amedyukh/.wandb_api_key') as f:
            key = f.read()
        os.environ['WANDB_API_KEY'] = key
    else:
        os.environ['WANDB_MODE'] = 'offline'

    wandb.init(project=config.wandb_project, config=vars(config))

    # Update model path
    config.model_path = os.path.join(config.model_path, get_model_name(config.log_progress))

    # Save training parameters
    os.makedirs(config['model_path'], exist_ok=True)
    with open(os.path.join(config['model_path'], 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=4)

    # Setup model, loss, and metric
    if config.model.lower() == 'unet':
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=config.model_channels,
            strides=(2,) * (len(config.model_channels) - 1),
            num_res_units=config.num_residual_units,
            norm=Norm.BATCH,
        )
    elif config.model.lower() == 'csnet':
        net = CSNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=config.model_channels,
            strides=(2,) * (len(config.model_channels) - 1),
            num_res_units=config.num_residual_units,
            norm=Norm.BATCH,
        )
    elif config.model.lower() == 'csnet_orig':
        net = CSNetOrig(2, 1)
    else:
        raise NotImplementedError(
            rf'{config.model} is an invalid model; must be one of ["unet", "csnet", "csnet_orig"]')

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Setup data loaders
    
    # Train and validation transforms
    train_tr, val_tr = get_default_train_transforms(roi_size=config.roi_size)

    # Training and validation file lists
    train_files = get_data_dict(os.path.join(config.data_dir, config.train_dirname, config.img_dirname),
                                os.path.join(config.data_dir, config.train_dirname, config.gt_dirname))
    val_files = get_data_dict(os.path.join(config.data_dir, config.val_dirname, config.img_dirname),
                              os.path.join(config.data_dir, config.val_dirname, config.gt_dirname))

    # Dataset and dataloader for training
    tr_ds = CacheDataset(data=train_files, transform=train_tr, cache_rate=1, num_workers=2 * config.batch_size)
    train_dl = DataLoader(tr_ds, batch_size=config.batch_size, shuffle=True, num_workers=2 * config.batch_size)

    # Dataset and dataloader for validation
    val_ds = CacheDataset(data=val_files, transform=val_tr, cache_rate=1.0, num_workers=2 * config.batch_size)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, num_workers=2 * config.batch_size)

    # train
    train(train_dl, val_dl, net, loss_function, dice_metric, config, log_tensorboard=True)
    wandb.finish()
