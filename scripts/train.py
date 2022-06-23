import argparse
import os

import wandb

from csnet.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tg', '--dir-train-gt', type=str,
                        help='Directory with ground truth labels for training', required=True)
    parser.add_argument('-ti', '--dir-train-img', type=str,
                        help='Directory with input images for training', required=True)
    parser.add_argument('-vg', '--dir-val-gt', type=str,
                        help='Directory with ground truth labels for validation', required=True)
    parser.add_argument('-vi', '--dir-val-img', type=str,
                        help='Directory with input images for validation', required=True)
    parser.add_argument('-m', '--model-path', type=str,
                        help='Directory for model checkpoints', default='model')
    parser.add_argument('-s', '--snapshot', type=int, default=5,
                        help='Save model snapshot every s epochs')
    parser.add_argument('-v', '--val-step', type=int, default=5,
                        help='Calculate validation accuracy every v epochs')
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
    parser.add_argument('-ww', '--wce-loss-weight', type=float, default=0.6,
                        help='Weight for the weighted cross entropy component of the loss')
    parser.add_argument('-dw', '--dice-loss-weight', type=float, default=0.4,
                        help='Weight for the dice component of the loss')
    parser.add_argument('-pr', '--wandb-project', type=str, default='',
                        help='wandb project name')
    parser.add_argument('-log', '--log-progress', action='store_true')

    config = parser.parse_args()

    print('\nThe following are the parameters that will be used:')
    print(vars(config))
    print('\n')

    if config.log_progress:
        with open('/home/amedyukh/.wandb_api_key') as f:
            key = f.read()
        os.environ['WANDB_API_KEY'] = key
    else:
        os.environ['WANDB_MODE'] = 'offline'

    wandb.init(project=config.wandb_project, config=vars(config))
    train(config.dir_train_img, config.dir_train_gt,
          config.dir_val_img, config.dir_val_gt, config, config.log_progress)
    wandb.finish()
