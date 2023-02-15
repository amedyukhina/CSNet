import argparse
import os

import numpy as np
from cs_sim.batch.batch_corrupt import batch_corrupt_image
from cs_sim.batch.batch_synth import batch_generate_img_with_filaments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str,
                        help='Directory to save generated data', required=True)
    parser.add_argument('-s', '--img-shape',
                        type=str, default="16,64,64", help='Image shape, separated by ","')
    parser.add_argument('-gt', '--name-gt', type=str, default='gt',
                        help='Subfolder name for ground gruth labels')
    parser.add_argument('-img', '--name-img', type=str, default='img',
                        help='Subfolder name for input images')
    parser.add_argument('-sf', '--subfolders', type=str, default='train,val,test',
                        help='List of subfolders for train-validation-test split, separated by ","')
    parser.add_argument('-n', '--n-img', type=str, default="20,10,5",
                        help='Number of images for each train-validation-test split, separated by ","')
    parser.add_argument('-j', '--n-jobs', type=int, default=8)

    args = parser.parse_args()
    args.img_shape = tuple(np.int_(args.img_shape.split(',')))
    args.n_img = np.int_(args.n_img.split(','))
    args.subfolders = args.subfolders.split(',')

    print('\nThe following are the parameters that will be used:')
    print(vars(args))
    print('\n')

    for sf, n in zip(args.subfolders, args.n_img):
        print(sf, n)
        dir_gt = os.path.join(args.data_dir, sf, args.name_gt)
        dir_img = os.path.join(args.data_dir, sf, args.name_img)

        os.makedirs(dir_gt, exist_ok=True)
        os.makedirs(dir_img, exist_ok=True)

        batch_generate_img_with_filaments(n_img=n, n_jobs=10, dir_out=dir_gt,
                                          imgshape=args.img_shape, n_filaments=10,
                                          maxval=255)
        batch_corrupt_image(dir_gt, dir_img, n_jobs=20,
                            corruption_steps=[
                                ('poisson_noise', {'snr': 2}),
                                ('convolve', {'sigma': 2}),
                                ('gaussian_noise', {'snr': 50})
                            ])
