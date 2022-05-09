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
from smooth.lib import navigation

import torchvision.models as models
import torchvision.transforms as transforms
import sklearn
import sklearn.manifold as sk_manifold
import lpips
import torch.optim as optim
from matplotlib.patches import Rectangle


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
    def __init__(self, input_dim = 4, hidden_dim = 64, num_classes = 2):
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
    [X_lab,y_lab,X_unlab,y_unlab] = navigation.create_dataset (args.dataset, args.n_dim, args.n_train, args.n_unlab, args.width, args.data_dir)
    # toyexample.save_dataset(X_lab,y_lab,X_unlab,y_unlab, args.output_dir)
    goal = [19, 1]
    start = [1, 1]
    intermediate_points = [[10, 5]]
    # Trajectories
    fig, ax = plt.subplots()
    plt.plot(X_lab[:,0],X_lab[:,1],'.')

    ax.plot(goal[0],goal[1],'r*')
    ax.plot(start[0],start[1],'g*')
    ax.plot(np.array(intermediate_points)[:,0],np.array(intermediate_points)[:,1],'bo')
    ax.quiver(X_lab[:,0],X_lab[:,1],X_lab[:,2],X_lab[:,3],color="#0000ff") # Blue Velocity
    ax.quiver(X_lab[:,0],X_lab[:,1],y_lab[:,0],y_lab[:,1],color="#ff0000") # Red Accelaration
    ax.add_patch(Rectangle((9, 0), 2, 4,
                           edgecolor='black',
                           facecolor='black',
                           fill=True,
                           lw=5))

    ax.add_patch(Rectangle((9, 6), 2, 4,
                           edgecolor='black',
                           facecolor='black',
                           fill=True,
                           lw=5))
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(0, 10)
    plt.savefig(args.output_dir+'/labeled_traj.pdf')

    # Unlabeled Trajectory
    fig, ax = plt.subplots()
    plt.plot(X_lab[:,0],X_lab[:,1],'.')

    ax.plot(goal[0],goal[1],'r*')
    ax.plot(start[0],start[1],'g*')
    ax.plot(np.array(intermediate_points)[:,0],np.array(intermediate_points)[:,1],'bo')
    ax.quiver(X_lab[:,0],X_lab[:,1],X_lab[:,2],X_lab[:,3],color="#0000ff") # Blue Velocity
    ax.quiver(X_unlab[:,0],X_unlab[:,1],X_unlab[:,2],X_unlab[:,3],color="#0000ff") # Blue Velocity
    ax.quiver(X_lab[:,0],X_lab[:,1],y_lab[:,0],y_lab[:,1],color="#ff0000") # Red Accelaration
    ax.add_patch(Rectangle((10-args.width, 0), 2*args.width, 4,
                           edgecolor='black',
                           facecolor='black',
                           fill=True,
                           lw=5))

    ax.add_patch(Rectangle((9-args.width, 6), 2*args.width, 4,
                           edgecolor='black',
                           facecolor='black',
                           fill=True,
                           lw=5))
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(0, 10)
    plt.savefig(args.output_dir+'/unlabeled_traj.pdf')


    # Load the data in Device
    X_lab, y_lab = torch.Tensor(X_lab).to(device), torch.Tensor(y_lab).type(torch.Tensor).to(device)

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
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            loss = F.mse_loss(net(X_lab), y_lab)
            loss.backward()
            optimizer.step()
            if epoch%500 == 0:
                print(loss)
            # acc = accuracy(net, unlab_dataloader, 'cuda')
            # utils.save_state(args.output_dir, epoch, loss.item(), acc, filename='losses.csv')

        print(F.mse_loss(net(X_lab), y_lab))
        # Evaluate in a couple of trajectories
        # x, y, x_dot, y_dot
        initials = [[10,5,2,0],[10,5,2,0],[1,1,2,2],[1,1,4,4],[1,1,1,1],[1,1,0.5,0.5],[1,1,0.1,0.1],[1,1,0.01,0.01]]
        time_step = 4 / args.n_train
        total_time = 6
        trajs = [np.array([]) for i in range(len(initials))]
        accelerations = [np.array([]) for i in range(len(initials))]

        for i,init in enumerate(initials):
            state = np.array(init)
            trajs[i] = state
            for t in range(int(total_time/time_step)):
                with torch.no_grad():
                    acc = net(torch.Tensor(state).to(device)).cpu().detach().numpy()
                state = navigation.step(state,acc,time_step)
                if len(accelerations[i])==0:
                    accelerations[i] = acc

                trajs[i] = np.vstack((trajs[i],state))
                accelerations[i] = np.vstack((accelerations[i],acc))

            fig, ax = plt.subplots()
            plt.plot(trajs[i][:,0], trajs[i][:,1], '.-')

            ax.plot(goal[0], goal[1], 'r*')
            ax.plot(initials[i][0], initials[i][1], 'g*')
            ax.quiver(trajs[i][:,0], trajs[i][:, 1], trajs[i][:, 2], trajs[i][:, 3], color="#0000ff")  # Blue Velocity
            ax.quiver(trajs[i][:, 0], trajs[i][:, 1], accelerations[i][:, 0], accelerations[i][:, 1], color="#ff0000")  # Red Accelaration
            ax.add_patch(Rectangle((9, 0), 2, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((9, 6), 2, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))
            plt.grid(True)
            plt.xlim(0, 20)
            plt.ylim(0, 10)
            plt.savefig(args.output_dir + '/traj_generated'+str(i)+'.pdf')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='trajectories')
    parser.add_argument('--dataset', type=str, default='window')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=100, help='Number of samples per class')
    parser.add_argument('--width', type=float, default=1.)
    parser.add_argument('--data_dir', type=str, default='./smooth/data')


    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--regularizer', type=float, default=1)
    parser.add_argument('--heat_kernel_t', type=float, default=0.05)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.9)


    parser.add_argument('--dual_step_mu', type=float, default=0.5)
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