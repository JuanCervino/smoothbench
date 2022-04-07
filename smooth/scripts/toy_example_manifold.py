import argparse
import torch
import os
import json
import pandas as pd
import time
from datetime import datetime
import pickle
import scipy

# Juan Added
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision.utils import save_image
from smooth import laplacian
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl

from humanfriendly import format_timespan

from sklearn.neighbors import kneighbors_graph
from torchvision.datasets import CIFAR10 as CIFAR10_


from smooth import datasets
from smooth import algorithms
from smooth import attacks
from smooth import hparams_registry
from smooth.lib import misc, meters
from smooth import laplacian

import torchvision.models as models
import torchvision.transforms as transforms
import sklearn
import lpips




def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--output_dir', type=str, default='knn')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('--number_knn', type=int, default=10, help='Number of KNNs')
    parser.add_argument('--metric', type=str, choices=['euclidean', 'cosine_similarity', 'lpips_alex', 'lpips_resnet'],
                        default='euclidean', help='Distance to use')
    parser.add_argument('--model', type=str, choices=['alexnet', 'resnet18', 'None'],
                        default='None', help='Model To Use, if none you work with data')
    parser.add_argument('--pretrained', type=str, choices=['None','cifar10', 'imagenet'],
                        default='None', help='Where was the model pretrained')
    parser.add_argument('--transforms', type=str, choices=['None','normalized'],
                        default='None', help='What transform to apply to your data')
    parser.add_argument('--layer_n', type=int, default = 11 , help='At what layer are we truncating the neural network')

    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.model) + str(args.layer_n) + '_' + str(args.transforms)+'_' + str(args.metric) + '_' + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    main(args)