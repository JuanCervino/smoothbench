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


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Choose transform
    if args.transforms == 'None':
        train_transforms = transforms.ToTensor()
    elif args.transforms == 'Normalized':
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
        ])

    # Create Dataset
    train_data = CIFAR10_(args.data_dir, train=True, transform=train_transforms, download=True)


    # Create Model
    if args.model == 'None':

        train_all_ldr = DataLoader(dataset=train_data, batch_size=int(train_data.__len__()), shuffle=False)
        train_all_ldr_iter = iter(train_all_ldr)
        dataset_unlab, _ = next(train_all_ldr_iter)
        flat = dataset_unlab.flatten(start_dim=1)

    elif args.model == 'resnet18':
        DataLoader(dataset=train_data, batch_size=int(train_data.__len__()/10), shuffle=False)
        print('Create it')




    # Commpute Adjacency Matrix

    if args.distance == 'Euclidean':
        Adj = torch.cdist(flat,flat)

    elif args.distance == 'cosine_similarity':
        Adj = sklearn.metrics.pairwise.cosine_distances(flat)

    k = args.number_knn

    # Get KNN
    knn = kneighbors_graph(Adj, k)

    # Save it
    pickle.dump(knn, open(args.output_dir+"/knn.p", "wb"))

    # Analyze it
    counter = [0 for i in range(10)]
    occurences = [0 for i in range(10)]
    train_all_ldr_one = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

    for batch_idx, (imgs, labels) in enumerate(train_all_ldr_one):
        subset_samples_knn = torch.utils.data.Subset(train_data,
                                                         knn[batch_idx].indices.tolist())
        ldr_samples_ldr_iterator = iter(
            DataLoader(subset_samples_knn, batch_size=subset_samples_knn.__len__(),
                       shuffle=False))  # set shuffle to False
        batch_samples_ldr_iterator, batch_samples_ldr_iterator_labels = next(ldr_samples_ldr_iterator)

        for nn in range(10):

            if labels in batch_samples_ldr_iterator_labels[0:nn + 1]:
                counter[nn] += 1
                occurences[nn] += 100 * batch_samples_ldr_iterator_labels.tolist()[0:nn + 1].count(
                    labels.tolist()[0]) / (50000 * (nn + 1))


    pickle.dump([counter, occurences], open(args.output_dir+ '/occurences.p' , "wb"))
    with open(args.output_dir + '/results.txt', 'w') as f:
        f.write(str([counter, occurences]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='K Nearest Neighbours Evaluation')
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--output_dir', type=str, default='knn')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('--number_knn', type=int, default=10, help='Number of KNNs')
    parser.add_argument('--metric', type=str, choices=['euclidean', 'cosine_similarity'],
                        default='euclidean', help='Distance to use')
    parser.add_argument('--model', type=str, choices=['resnet18', 'None'],
                        default='None', help='Model To Use')
    parser.add_argument('--pretrained', type=str, choices=['None','cifar10', 'imagenet'],
                        default='None', help='Where was the model pretrained')
    parser.add_argument('--transforms', type=str, choices=['None','Normalized'],
                        default='None', help='Where was the model pretrained')


    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.model) + '_' + str(args.metric) + '_' + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    main(args)