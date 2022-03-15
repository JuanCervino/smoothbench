import argparse
import torch
import os
import json
import pandas as pd
import time
from datetime import datetime

# Juan Added
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from smooth import laplacian


from humanfriendly import format_timespan



from advbench import datasets
from advbench import algorithms
from advbench import attacks
from advbench import hparams_registry
from advbench.lib import misc, meters
from advbench import laplacian


def main(args, hparams, test_hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['regularizer'] = args.regularizer
    hparams['unlabeled_batch_size'] = args.unlabeled_batch_size

    dataset = vars(datasets)[args.dataset](args.data_dir, args.per_labeled)
    # 'train_labeled', 'train_unlabeled', 'train_all', 'test'
    train_lab_ldr, train_unl_ldr, train_all_ldr, test_ldr = datasets.to_loaders(dataset, hparams)
    print(len(train_lab_ldr),len(train_all_ldr))

    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE,
        dataset.NUM_CLASSES,
        hparams,
        device).to(device)

    adjust_lr = None if dataset.HAS_LR_SCHEDULE is False else dataset.adjust_lr

    # test_attacks = {
    #     a: vars(attacks)[a](algorithm.classifier, test_hparams, device) for a in args.test_attacks}
    #
    columns = ['Epoch', 'Accuracy', 'Eval-Method', 'Split', 'Train-Alg', 'Dataset', 'Trial-Seed', 'Output-Dir']
    results_df = pd.DataFrame(columns=columns)
    def add_results_row(data):
        defaults = [args.algorithm, args.dataset, args.trial_seed, args.output_dir]
        results_df.loc[len(results_df)] = data + defaults

    total_time = 0
    # Juan Added this
    # print(dataset.splits.items(), dataset.splits['train_all'])
    lambdas = np.ones(len(dataset.splits['train_all']))/len(dataset.splits['train_all'])
    print(len(dataset.splits['train_all']),len(lambdas))


    train_all_ldr_full = DataLoader(dataset.splits['train_all'], batch_size=dataset.splits['train_all'].__len__(),
                                    shuffle=False)
    train_all_ldr_full_iter = iter(train_all_ldr_full)
    dataset_unlab, _ = next(train_all_ldr_full_iter)
    # dataset_unlab = dataset_unlab.to(device)
    print('Here')
    A = laplacian.get_pairwise_euclidean_distance_matrix(dataset_unlab)
    print('Have A')
    print('Average distance is ', np.linalg.norm(A,'fro')/ (2*A.size()[0]) )
    eA = np.exp(A)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--data_dir', type=str, default='./advbench/data')
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