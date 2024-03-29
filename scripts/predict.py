import argparse
import json
import os

import torch

from csnet.utils.model import get_model
from csnet.utils.predict import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str,
                        help='Input directory with images to predict', required=True)
    parser.add_argument('-o', '--output-dir', type=str,
                        help='Directory to save predicted images', required=True)
    parser.add_argument('-m', '--model-path', type=str,
                        help='Model name', required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help='Batch size')

    args = parser.parse_args()

    print('\nThe following are the parameters that will be used:')
    print(vars(args))
    print('\n')

    # Setup and load model
    with open(os.path.join(os.path.dirname(args.model_path), 'config.json')) as f:
        config = json.load(f)
    config = argparse.Namespace(**config)

    net = get_model(config)
    net.load_state_dict(torch.load(args.model_path))

    image, predicted = predict(args.input_dir, args.output_dir, net,
                               roi_size=config.roi_size, return_last=True, batch_size=args.batch_size)
