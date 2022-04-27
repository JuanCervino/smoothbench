import argparse

import matplotlib.pyplot as plt
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
from torch.utils.data import Dataset, Subset, DataLoader, random_split, TensorDataset
from torchvision.utils import save_image
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl
from smooth.lib import utils

from humanfriendly import format_timespan

from sklearn.neighbors import kneighbors_graph
from torchvision.datasets import CIFAR10 as CIFAR10_
import torch.nn.functional as F
from torch import nn

from smooth import datasets
from smooth import algorithms
from smooth import attacks
from smooth import hparams_registry
from smooth.lib import misc, meters
from smooth import laplacian
from smooth.lib import toyexample

import torchvision.models as models
import torchvision.transforms as transforms
import sklearn
import sklearn.manifold as sk_manifold
import lpips
import torch.optim as optim

@torch.no_grad()
def accuracy(net, loader, device):
    correct, total = 0, 0
    net = net.to(device)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = net(imgs).to(device)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)

    return 100. * correct / total

class FCNN(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 64, num_classes = 2):
        super(FCNN, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim,out_features=num_classes,bias=False)

    def forward(self, x):
        # out = F.relu(self.layer1(x))
        out = torch.tanh(self.layer1(x))
        out = self.layer2(out)
        return out


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Create Dataset
    [X_lab,y_lab,X_unlab,y_unlab] = toyexample.create_dataset (args.dataset, args.n_dim, args.n_train, args.n_unlab, args.n_test)
    toyexample.save_dataset(X_lab,y_lab,X_unlab,y_unlab, args.output_dir)

    # Create NN
    net = FCNN().to(device)


    # Create the optimer
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # Train
    if args.algorithm == 'ERM':

        columns = ['Epoch', 'Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        X_lab, y_lab = torch.Tensor(X_lab), torch.Tensor(y_lab).type(torch.LongTensor)
        X_lab, y_lab = X_lab.to(device), y_lab.to(device)
        X_unlab,y_unlab = torch.Tensor(X_unlab).to(device),torch.Tensor(y_unlab).to(device)

        unlab_dataset = TensorDataset(X_unlab, y_unlab)  # create your datset
        # unlab_dataloader = DataLoader(unlab_dataset, batch_size = int(unlab_dataset.__len__()/10),num_workers=10)
        unlab_dataloader = DataLoader(unlab_dataset)

        for epoch in range(args.epochs):
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.999
            optimizer.zero_grad()
            loss = F.cross_entropy(net(X_lab), y_lab)
            loss.backward()
            optimizer.step()
            acc = accuracy(net,unlab_dataloader,'cuda')
            utils.save_state(args.output_dir, epoch, loss.item(),acc , filename = 'losses.csv')

        out_lab = net(X_lab).argmax(dim=1, keepdim=True)
        out_unlab = net(X_unlab).argmax(dim=1, keepdim=True)

        out_lab_np = out_lab.cpu().detach().numpy()
        out_lab_np = out_lab_np.squeeze()
        out_unlab_np = out_unlab.cpu().detach().numpy()
        out_unlab_np = out_unlab_np.squeeze()
        toyexample.save_output(X_lab.cpu().detach().numpy(), out_lab_np,
                                X_unlab.cpu().detach().numpy(), out_unlab_np, args.output_dir)
        print('Final Accuracy', accuracy(net, unlab_dataloader, 'cuda') )
    if args.algorithm == 'LAPLACIAN_REGULARIZATION':

        columns = ['Epoch', 'Loss CE','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)

        X_lab, y_lab = torch.Tensor(X_lab).to(device), torch.Tensor(y_lab).type(torch.LongTensor).to(device)
        X_unlab, y_unlab = torch.Tensor(X_unlab).to(device), torch.Tensor(y_unlab).to(device)

        unlab_dataset = TensorDataset(X_unlab, y_unlab)  # create your datset
        # unlab_dataloader = DataLoader(unlab_dataset, batch_size = int(unlab_dataset.__len__()/10),num_workers=10)
        unlab_dataloader = DataLoader(unlab_dataset)

        L = laplacian.get_laplacian(X_unlab, args.normalize, heat_kernel_t=args.heat_kernel_t).to(device)
        # zero = torch.zeros_like(L)
        # L_smooth =  torch.where(L > 0, L, zero)
        print(L)
        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        for epoch in range(args.epochs):
            optimizer.zero_grad()
            # print(y_lab)
            loss = F.cross_entropy(net(X_lab), y_lab)
            loss_cel = loss
            # print(loss)
            f = F.softmax(net(X_unlab))
            loss += args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))

            # print(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
            loss.backward()
            optimizer.step()
            acc = accuracy(net,unlab_dataloader,'cuda')
            utils.save_state(args.output_dir, epoch, loss_cel.item(),args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))).item() ,torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))).item(),acc , filename = 'losses.csv')
            print(epoch,loss_cel.item(), torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))).item(),(args.regularizer*torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))).item(),acc)


        out_lab = net(X_lab).argmax(dim=1, keepdim=True)
        out_unlab = net(X_unlab).argmax(dim=1, keepdim=True)

        out_lab_np = out_lab.cpu().detach().numpy()
        out_lab_np = out_lab_np.squeeze()
        out_unlab_np = out_unlab.cpu().detach().numpy()
        out_unlab_np = out_unlab_np.squeeze()
        toyexample.save_output(X_lab.cpu().detach().numpy(), out_lab_np,
                                X_unlab.cpu().detach().numpy(), out_unlab_np, args.output_dir)

    if args.algorithm == 'MANIFOLD_GRADIENT':

        columns = ['Epoch', 'Loss CE','Regularized Laplacian Loss', 'Laplacian Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)

        X_lab, y_lab = torch.Tensor(X_lab).to(device), torch.Tensor(y_lab).type(torch.LongTensor).to(device)
        X_unlab, y_unlab = torch.Tensor(X_unlab).to(device), torch.Tensor(y_unlab).to(device)

        unlab_dataset = TensorDataset(X_unlab, y_unlab)  # create your datset
        # unlab_dataloader = DataLoader(unlab_dataset, batch_size = int(unlab_dataset.__len__()/10),num_workers=10)
        unlab_dataloader = DataLoader(unlab_dataset)



        adj_matrix = torch.cdist(X_unlab, X_unlab)
        L = laplacian.get_laplacian(X_unlab,  True, heat_kernel_t=args.heat_kernel_t, clamp_value = 0.001).to(device)
        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        lambda_dual = torch.ones(len(y_unlab)) / len(y_unlab)
        lambda_dual = lambda_dual.to(device).detach().requires_grad_(False)
        mu_dual = torch.Tensor(1).to(device).detach().requires_grad_(False)
        rho_primal = torch.Tensor(1).to(device).detach().requires_grad_(False)

        for epoch in range(args.epochs):
            ############################################
            # Primal Update
            ############################################
            optimizer.zero_grad()
            loss = mu_dual * F.cross_entropy(net(X_lab), y_lab)
            loss_cel = loss
            f = F.softmax(net(X_unlab))
            loss += args.regularizer * torch.trace(torch.matmul((torch.diag(lambda_dual)@f).transpose(0,1),torch.matmul(L, f)))

            # print(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
            print("here loss",loss)
            loss.backward()

            optimizer.step()
            acc = accuracy(net,unlab_dataloader,'cuda')
            utils.save_state(args.output_dir, epoch, loss_cel.item(),args.regularizer * torch.trace(torch.matmul((torch.diag(lambda_dual)@f).transpose(0,1),torch.matmul(L, f))).item() ,torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))).item(),acc , filename = 'losses.csv')
            print(epoch,loss_cel.item(), torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))).item(),(args.regularizer*torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))).item(),acc)
            ############################################
            # Now update rho
            ############################################
            with torch.no_grad():
                rho_primal = torch.nn.functional.relu(rho_primal - args.rho_step * (1 - torch.sum(lambda_dual)))

            ############################################
            # Dual Update
            ############################################
            with torch.no_grad():
                mu_dual = torch.nn.functional.relu(mu_dual + args.dual_step_mu * (F.cross_entropy(net(X_lab), y_lab) - args.epsilon))
                f_prime = F.softmax(net(X_unlab))
                f_matrix= []
                f_matrix.append([])
                f_matrix [0] = torch.cat([f_prime[:,0]] * f_prime.shape[0]).reshape((f_prime.shape[0], f_prime.shape[0]))
                f_matrix.append([])
                f_matrix[1] = torch.cat([f_prime[:,1]] * f_prime.shape[0]).reshape((f_prime.shape[0], f_prime.shape[0]))

                numerator = torch.abs (f_matrix [0] - f_matrix[0].transpose(0,1)) + torch.abs(f_matrix [1] - f_matrix[1].transpose(0,1)).to(device)
                division = torch.div(numerator, (adj_matrix + torch.eye(f_prime.shape[0]).to(device)))
                [grads,indices] = torch.max(division, 1)
                grads = grads.pow(2)
                lambda_dual = F.relu(lambda_dual + args.dual_step_mu*(grads-rho_primal))


        out_lab = net(X_lab).argmax(dim=1, keepdim=True)
        out_unlab = net(X_unlab).argmax(dim=1, keepdim=True)

        out_lab_np = out_lab.cpu().detach().numpy()
        out_lab_np = out_lab_np.squeeze()
        out_unlab_np = out_unlab.cpu().detach().numpy()
        out_unlab_np = out_unlab_np.squeeze()
        toyexample.save_output(X_lab.cpu().detach().numpy(), out_lab_np,
                                X_unlab.cpu().detach().numpy(), out_unlab_np, args.output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='toy')
    parser.add_argument('--dataset', type=str, default='two_moons')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=100, help='Number of samples per class')
    parser.add_argument('--n_test', type=int, default=10)

    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--regularizer', type=float, default=1)
    parser.add_argument('--heat_kernel_t', type=float, default=0.05)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.9)

    parser.add_argument('--dual_step_mu', type=float, default=0.1)
    parser.add_argument('--dual_step_lambda', type=float, default=0.1)
    parser.add_argument('--rho_step', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.01)

    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.dataset) +  '_' + args.algorithm+  '_'  + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # if args.dataset not in vars(datasets):
    #     raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    main(args)