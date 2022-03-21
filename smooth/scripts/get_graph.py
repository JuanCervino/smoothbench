import argparse
import torch
import os
import json
import pandas as pd
import time
from datetime import datetime
import pickle

# Juan Added
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from smooth import laplacian


from humanfriendly import format_timespan

from sklearn.neighbors import kneighbors_graph


from smooth import datasets
from smooth import algorithms
from smooth import attacks
from smooth import hparams_registry
from smooth.lib import misc, meters
from smooth import laplacian


def main(args, hparams, test_hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['regularizer'] = args.regularizer
    hparams['unlabeled_batch_size'] = args.unlabeled_batch_size
    hparams['heat_kernel_t'] = args.heat_kernel_t


    dataset = vars(datasets)[args.dataset](args.data_dir, args.per_labeled)
    # 'train_labeled', 'train_unlabeled', 'train_all', 'test'
    train_lab_ldr, train_unl_ldr, train_all_ldr, test_ldr = datasets.to_loaders(dataset, hparams)
    print(len(train_lab_ldr),len(train_all_ldr))
    DataLoader(
        dataset=dataset,
        batch_size=dataset.splits['train_all'].__len__(),
        num_workers=all_datasets.N_WORKERS,
        shuffle=False)



    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    # parser.add_argument('--test_attacks', type=str, nargs='+', default=['PGD_Linf'])  # Juan Changed this
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')

    # Juan added this
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize the Laplacian')
    parser.add_argument('--regularizer', type=float, default=.1, help='Regularizer for the SSL')
    parser.add_argument('--per_labeled', type=float, default=1., help='Percentage of training set that will be labeled (between (0,1])')
    parser.add_argument('--unlabeled_batch_size', type=int, default=128, help='Batchsize used to compute Laplacian')
    parser.add_argument('--heat_kernel_t', type=float, default=1., help='Value of t in the Heat Kernel computation ')



    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.algorithm) + '_' + str(args.per_labeled) + '_' + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        seed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, seed)

    print ('Hparams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2)

    test_hparams = hparams_registry.test_hparams(args.algorithm, args.dataset)

    print('Test hparams:')
    for k, v in sorted(test_hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'test_hparams.json'), 'w') as f:
        json.dump(test_hparams, f, indent=2)

    main(args, hparams, test_hparams)