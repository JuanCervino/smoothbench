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


class new_alexnet(torch.nn.Module):
    def __init__(self, output_layer=None, layer_n = 11):
        # layer_n corresponds to the last layer to consider:
        # 11 is conv 5
        # 9 is conv 4
        # 7 is conv 3
        # 4 is conv 2
        # 1 is conv 1

        super().__init__()
        self.pretrained = models.alexnet(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count  ):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = torch.nn.Sequential(self.pretrained._modules['features'][0:layer_n])
        print(self.net._modules)

        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x

class new_resnet18(torch.nn.Module):
    def __init__(self, output_layer=None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = torch.nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Choose transform
    if args.model == 'None':
        if args.metric in ['euclidean','cosine_similarity']:
            if args.transforms == 'None':
                train_transforms = transforms.ToTensor()
            elif args.transforms == 'normalized':
                train_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),
                ])
        elif args.metric in ['lpips_alex', 'lpips_vgg']:
            if args.transforms == 'None':
                train_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(64),])
            elif args.transforms == 'normalized':
                train_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(64),
                                    transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                                         std=[0.24703233, 0.24348505, 0.26158768]), ])

    elif args.model in ['alexnet','resnet18']:
        if args.transforms == 'None':
            train_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
        elif args.transforms == 'normalized':
            train_transforms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    # Create Dataset
    train_data = CIFAR10_(args.data_dir, train=True, transform=train_transforms, download=True)


    # Create Data
    if args.model == 'None':
        if args.metric in ['euclidean','cosine_similarity']:
            train_all_ldr = DataLoader(dataset=train_data, batch_size=int(train_data.__len__()), shuffle=False)
            train_all_ldr_iter = iter(train_all_ldr)
            dataset_unlab, _ = next(train_all_ldr_iter)
            flat = dataset_unlab.flatten(start_dim=1)
        elif args.metric in ['lpips_alex', 'lpips_vgg']:
            train_all_ldr = DataLoader(dataset=train_data, batch_size=int(train_data.__len__()), shuffle=False)
            train_all_ldr_iter = iter(train_all_ldr)
            dataset_unlab, _ = next(train_all_ldr_iter)


    elif args.model == 'alexnet':
        if args.pretrained == 'imagenet':
            train_all_ldr = DataLoader(dataset=train_data, batch_size=int(train_data.__len__()/10), shuffle=False)
            net = new_alexnet(output_layer='features', layer_n = args.layer_n)
            embedding = torch.Tensor()
            for batch_idx, (imgs_unlab, _) in enumerate(train_all_ldr):
                with torch.no_grad():

                    intermediate_layer = net(imgs_unlab)
                    intermediate_layer = intermediate_layer.to('cpu')
                    embedding = torch.cat((embedding, intermediate_layer))
                    print(embedding.shape)

            flat = embedding.flatten(start_dim=1)

    elif args.model == 'resnet18':
        if args.pretrained == 'imagenet':
            train_all_ldr = DataLoader(dataset=train_data, batch_size=int(train_data.__len__()/10), shuffle=False)
            net = new_resnet18(output_layer='layer'+str(args.layer_n))
            embedding = torch.Tensor()
            for batch_idx, (imgs_unlab, _) in enumerate(train_all_ldr):
                with torch.no_grad():

                    intermediate_layer = net(imgs_unlab)
                    intermediate_layer = intermediate_layer.to('cpu')
                    embedding = torch.cat((embedding, intermediate_layer))
                    print(embedding.shape)

            flat = embedding.flatten(start_dim=1)


    # Commpute Adjacency Matrix

    if args.metric == 'euclidean':
        Adj = torch.cdist(flat,flat)

    elif args.metric == 'cosine_similarity':
        Adj = sklearn.metrics.pairwise.cosine_distances(flat)

    elif args.metric == 'lpips_alex':
        print('Here')
        loss_fn_alex = lpips.LPIPS(net='alex')
        loss_fn_alex = loss_fn_alex.to(device)
        Adj = torch.Tensor(50000,50000)
        with torch.no_grad():
            for i in range(50000):
                for j in range(4):
                    print(i,j,dataset_unlab.shape)
                    d = loss_fn_alex(dataset_unlab[0+12500 * j : 12500 + 12500 * j ,:,:,:].to(device), dataset_unlab[i,:,:,:].to(device))
                    d = d.to('cpu')
                    Adj[0+12500 * j : 12500 + 12500 * j, i] = d [:,0,0,0]
                    print(Adj)

    elif args.metric == 'lpips_vgg':
        print('Here')
        loss_fn_resnet = lpips.LPIPS(net='vgg')
        loss_fn_resnet = loss_fn_resnet.to(device)
        Adj = torch.Tensor(50000,50000)
        with torch.no_grad():
            for i in range(50000):
                for j in range(20):
                    print(i,j,dataset_unlab.shape)
                    d = loss_fn_resnet(dataset_unlab[0+2500 * j : 2500 + 2500 * j ,:,:,:].to(device), dataset_unlab[i,:,:,:].to(device))
                    d = d.to('cpu')
                    Adj[0+2500 * j : 2500 + 2500 * j, i] = d [:,0,0,0]
                    print(Adj)

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
    parser.add_argument('--metric', type=str, choices=['euclidean', 'cosine_similarity', 'lpips_alex', 'lpips_vgg'],
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