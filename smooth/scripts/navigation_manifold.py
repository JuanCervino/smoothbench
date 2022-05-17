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
    def __init__(self, input_dim = 4, hidden_dim = 256, num_classes = 2):
        super(FCNN, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim,out_features=hidden_dim,bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim,out_features=num_classes,bias=False)

    def forward(self, x):
        # out = F.relu(self.layer1(x))
        out = torch.relu(self.layer1(x))
        out = self.layer2(out)
        return out

class FCNN2(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = [256,256], num_classes = 2):
        super(FCNN2, self).__init__()

        self.layer1 = nn.Linear(in_features=input_dim,out_features=hidden_dim[0],bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim[0],out_features=hidden_dim[1],bias=True)
        self.layer3 = nn.Linear(in_features=hidden_dim[1],out_features=num_classes,bias=False)

    def forward(self, x):
        # out = F.relu(self.layer1(x))
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = self.layer3(out)
        return out

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Create Dataset
    [X_lab,y_lab,X_unlab,y_unlab], adj_matrix = navigation.create_dataset (args.dataset, args.n_dim, args.n_train, args.n_unlab, args.data_dir, args.width, args.resolution)


    plot = True
    goal = [19,1]
    if plot:
        if args.dataset in ['center','window']:
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
            ax.add_patch(Rectangle((10-args.width, 0), 2*args.width, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((10-args.width, 6), 2*args.width, 4,
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

            ax.add_patch(Rectangle((10-args.width, 6), 2*args.width, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))
            plt.grid(True)
            plt.xlim(0, 20)
            plt.ylim(0, 10)
            plt.savefig(args.output_dir+'/unlabeled_traj.pdf')
        elif args.dataset in ['Dijkstra_grid_window','Dijkstra_random_window']:
            plt.savefig(args.output_dir+'/grid.pdf')
            fig, ax = plt.subplots()
            ax.add_patch(Rectangle((10-args.width, 0), 2*args.width, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((10-args.width, 6), 2*args.width, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            plt.plot(X_unlab[:,0], X_unlab[:,1], '*')
            plt.plot(X_lab[:,0], X_lab[:,1], 'r-*')
            plt.savefig(args.output_dir+'/full_grid.pdf')

            fig, ax = plt.subplots()
            ax.add_patch(Rectangle((10-args.width, 0), 2*args.width, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((10-args.width, 6), 2*args.width, 4,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            plt.plot(X_unlab[:,0], X_unlab[:,1], '*')
            plt.plot(X_lab[:,0], X_lab[:,1], 'r*')

            plt.quiver(X_lab[:,0], X_lab[:,1], y_lab[:,0], y_lab[:,1], color="#0000ff")
            plt.savefig(args.output_dir+'/dataset.pdf')
        elif args.dataset in ['Dijkstra_grid_maze','Dijkstra_grid_maze_two_points']:
            plt.savefig(args.output_dir+'/grid.pdf')
            fig, ax = plt.subplots()
            ax.add_patch(Rectangle((5-args.width, 3), 2*args.width, 7,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((15-args.width, 0), 2*args.width, 7,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            plt.plot(X_unlab[:,0], X_unlab[:,1], '*')
            plt.plot(X_lab[:,0], X_lab[:,1], 'r-*')
            plt.savefig(args.output_dir+'/full_grid.pdf')

            fig, ax = plt.subplots()
            ax.add_patch(Rectangle((5-args.width, 3), 2*args.width, 7,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))

            ax.add_patch(Rectangle((15-args.width, 0), 2*args.width, 7,
                                   edgecolor='black',
                                   facecolor='black',
                                   fill=True,
                                   lw=5))
            plt.plot(X_unlab[:,0], X_unlab[:,1], '*')
            plt.plot(X_lab[:,0], X_lab[:,1], 'r*')

            plt.quiver(X_lab[:,0], X_lab[:,1], y_lab[:,0], y_lab[:,1], color="#0000ff")
            plt.savefig(args.output_dir+'/dataset.pdf')


    # Load the data in Device
    X_lab, y_lab = torch.Tensor(X_lab).to(device), torch.Tensor(y_lab).type(torch.Tensor).to(device)
    X_unlab = torch.Tensor(X_unlab).to(device)#, torch.Tensor(y_unlab).type(torch.Tensor).to(device)

    # Create NN
    if args.dataset in ['window','center']:
        # net = FCNN().to(device) # 1 layer
        net = FCNN2().to(device) # 2 layers
    elif args.dataset in ['Dijkstra_grid_window','Dijkstra_random_window','Dijkstra_grid_maze','Dijkstra_grid_maze_two_points']:
        net = FCNN2(input_dim=2, hidden_dim=[2048,64]).to(device) # 1 layer


    # Create the optimer
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.dataset in ['Dijkstra_grid_window','Dijkstra_random_window','Dijkstra_grid_maze','Dijkstra_grid_maze_two_points']:
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.8)
    # Train
    if args.algorithm == 'ERM':

        columns = ['Epoch', 'Loss', 'Accuracy']
        utils.create_csv(args.output_dir, 'losses.csv', columns)
        for epoch in range(args.epochs):
            # for g in optimizer.param_groups:
            #     g['lr'] = g['lr']*args.weight_decay
            optimizer.zero_grad()
            loss = F.mse_loss(net(X_lab), y_lab)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch%1000 == 0:
                # print(epoch,loss)
            # acc = accuracy(net, unlab_dataloader, 'cuda')
                utils.save_state(args.output_dir, epoch, loss.item(), filename='losses.csv')

    elif args.algorithm == 'LAPLACIAN_REGULARIZATION':

        adj_matrix = torch.Tensor(adj_matrix).to(device)

        L = laplacian.get_laplacian_from_adj(adj_matrix, args.normalize, heat_kernel_t=args.heat_kernel_t, clamp_value=0.01).to(device)
        # zero = torch.zeros_like(L)
        # L_smooth =  torch.where(L > 0, L, zero)
        e, V = np.linalg.eig(L.cpu().detach().numpy())
        print('Connected Components', np.sum(e < 0.0001))

        for epoch in range(args.epochs):
            optimizer.zero_grad()
            loss = F.mse_loss(net(X_lab), y_lab)
            loss_cel = loss
            f = F.softmax(net(X_unlab))
            loss += args.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))

            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch % 1000 == 0:
                print(epoch, loss)
                fig, ax = plt.subplots()
                ax.quiver(X_unlab[:,0].cpu(), X_unlab[:,1].cpu(), net(X_unlab).cpu().detach().numpy()[:,0], net(X_unlab).cpu().detach().numpy()[:,1],
                          color="#ff0000")  # Blue Unlab
                ax.quiver(X_lab[:,0].cpu(), X_lab[:,1].cpu(), net(X_lab).cpu().detach().numpy()[:,0], net(X_lab).cpu().detach().numpy()[:,1],
                          color="#0000ff")
                ax.plot(goal[0], goal[1], 'r*')
                if args.dataset in ['Dijkstra_grid_window','Dijkstra_random_window']:
                    ax.add_patch(Rectangle((10 - args.width, 0), 2 * args.width, 4,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))

                    ax.add_patch(Rectangle((10 - args.width, 6), 2 * args.width, 4,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))
                if args.dataset in ['Dijkstra_grid_maze']:
                    ax.add_patch(Rectangle((5 - args.width, 3), 2 * args.width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))

                    ax.add_patch(Rectangle((15 - args.width, 0), 2 * args.width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))
                ax.plot(goal[0], goal[1], 'r*')
                plt.savefig(args.output_dir + '/traj_generated'+str(epoch)+'.pdf')

    elif args.algorithm == 'LIPSCHITZ_NO_RHO':
        columns = ['Epoch', 'Loss', 'Accuracy','MSE','mu_dual','laplacian']
        utils.create_csv(args.output_dir, 'losses.csv', columns)

        adj_matrix = torch.cdist(X_unlab,X_unlab).to(device)
        L = laplacian.get_euclidean_laplacian_from_adj(adj_matrix, args.normalize, clamp_value=args.clamp).to(device)
        # Plot graph
        sparseL = scipy.sparse.coo_matrix(L.cpu())
        fig, ax = plt.subplots()
        for i, j, v in zip(sparseL.row, sparseL.col, sparseL.data):
                arr = np.vstack((X_unlab[i, :].cpu(), X_unlab[j, :].cpu()))
                plt.plot(arr[:, 0], arr[:, 1], 'b-')
        ax.plot(X_unlab[:, 0].cpu(), X_unlab[:, 1].cpu(), '*')
        plt.savefig(args.output_dir + '/laplacian'+str(args.heat_kernel_t)+'.pdf')

        lambda_dual = torch.ones(X_unlab.shape[0]) / X_unlab.shape[0]
        lambda_dual = lambda_dual.to(device).detach().requires_grad_(False)
        mu_dual = torch.Tensor(1).to(device).detach().requires_grad_(False)

        for epoch in range(args.epochs):

            optimizer.zero_grad()
            loss = mu_dual *  F.mse_loss(net(X_lab), y_lab)
            loss_MSE = loss.item()
            f = net(X_unlab)
            loss += torch.trace(torch.matmul((torch.diag(lambda_dual)@f).transpose(0,1),torch.matmul(L, f)))
            # loss += args.regularizer * torch.trace(torch.matmul((f).transpose(0,1),torch.matmul(L, f)))

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if epoch % 1000 == 0:
                print('------------------------------')
                print(epoch,loss_MSE, loss.item(), torch.trace(torch.matmul((torch.diag(lambda_dual)@f).transpose(0,1),torch.matmul(L, f))) )
                print('mu',mu_dual.item())
                print('norm lambda', torch.sum(lambda_dual).item())
                print('------------------------------')
                utils.save_state(args.output_dir, epoch, loss.item(), loss_MSE, mu_dual,torch.trace(torch.matmul((torch.diag(lambda_dual)@f).transpose(0,1),torch.matmul(L, f))) , filename='losses.csv')
            ############################################
            # Dual Update
            ############################################
            with torch.no_grad():
                mu_dual = mu_dual + args.dual_step_mu * (F.mse_loss(net(X_lab), y_lab) - args.epsilon)
                mu_dual = torch.clamp(mu_dual,0,2)
                f_prime = net(X_unlab)
                f_matrix= []
                f_matrix.append([])
                f_matrix [0] = torch.cat([f_prime[:,0]] * f_prime.shape[0]).reshape((f_prime.shape[0], f_prime.shape[0]))
                f_matrix.append([])
                f_matrix[1] = torch.cat([f_prime[:,1]] * f_prime.shape[0]).reshape((f_prime.shape[0], f_prime.shape[0]))

                numerator = torch.abs (f_matrix [0] - f_matrix[0].transpose(0,1)) + torch.abs(f_matrix [1] - f_matrix[1].transpose(0,1)).to(device)
                division = torch.div(numerator, (adj_matrix + torch.eye(f_prime.shape[0]).to(device)))
                [grads,indices] = torch.max(division, 1)
                # grads = grads.pow(2)
                lambda_dual = F.relu(lambda_dual + args.dual_step_mu*(grads))

                # Juan Needs to Correct This
                lambda_dual = lambda_dual/torch.sum(lambda_dual).item()

            if epoch % 1000 == 0:
                print(epoch, loss)
                fig, ax = plt.subplots()
                ax.quiver(X_unlab[:,0].cpu(), X_unlab[:,1].cpu(), net(X_unlab).cpu().detach().numpy()[:,0], net(X_unlab).cpu().detach().numpy()[:,1],
                          color="#ff0000")  # Blue Unlab
                ax.quiver(X_lab[:,0].cpu(), X_lab[:,1].cpu(), net(X_lab).cpu().detach().numpy()[:,0], net(X_lab).cpu().detach().numpy()[:,1],
                          color="#0000ff")
                ax.plot(goal[0], goal[1], 'r*')
                if args.dataset in ['Dijkstra_grid_window','Dijkstra_random_window']:
                    ax.add_patch(Rectangle((10 - args.width, 0), 2 * args.width, 4,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))

                    ax.add_patch(Rectangle((10 - args.width, 6), 2 * args.width, 4,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))
                if args.dataset in ['Dijkstra_grid_maze']:
                    ax.add_patch(Rectangle((5 - args.width, 3), 2 * args.width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))

                    ax.add_patch(Rectangle((15 - args.width, 0), 2 * args.width, 7,
                                           edgecolor='black',
                                           facecolor='black',
                                           fill=True,
                                           lw=5))
                ax.plot(goal[0], goal[1], 'r*')
                plt.savefig(args.output_dir + '/traj_generated'+str(epoch)+'.pdf')
                fig, ax = plt.subplots()
                x = np.linspace(0, 20, 2 * args.n_train - 1)
                y = np.linspace(0, 10, args.n_train)
                # full coorindate arrays
                xx, yy = np.meshgrid(x, y)
                grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to('cuda')
                max_lambda = np.max(lambda_dual.detach().cpu().numpy())
                lambdas = 60 * lambda_dual.detach().cpu().numpy() / max_lambda
                colors = np.array(
                    ["#377eb8", "#ff7f00", "#4daf4a"]
                )
                plt.scatter(X_unlab[:, 0].detach().cpu().numpy(), X_unlab[:, 1].detach().cpu().numpy(), s=lambdas)
                plt.savefig(args.output_dir + '/lambdas'+str(epoch)+'.pdf')


    # print(F.mse_loss(net(X_lab), y_lab))
    # Evaluate in a couple of trajectories
    # x, y, x_dot, y_dot
    # initials = [[10,5,2,0],[10,5,2,0],[10,5,3,0],[10,5,4,0],[1,1,2,2],[1,1,4,4],[1,1,1,1],[1,1,0.5,0.5],[1,1,0.1,0.1],[1,1,0.01,0.01]]
    initials = [[1,1],[1, 7],[2, 9],[1, 9],X_lab[0,:].cpu(),X_lab[1,:].cpu(),X_lab[5,:].cpu(),X_lab[-1,:].cpu(),X_lab[-2,:].cpu()]

    if args.dataset in ['window','center']:
        time_step = 4 / args.n_train
    elif args.dataset in ['Dijkstra_random_window','Dijkstra_grid_window','Dijkstra_grid_maze','Dijkstra_grid_maze_two_points']:
        time_step = 0.1


    total_time = 20
    trajs = [np.array([]) for i in range(len(initials))]
    accelerations = [np.array([]) for i in range(len(initials))]

    fig, ax = plt.subplots()

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

        # fig, ax = plt.subplots()
        plt.plot(trajs[i][:,0], trajs[i][:,1], '.-')

        ax.plot(goal[0], goal[1], 'r*')
        ax.plot(initials[i][0], initials[i][1], 'g*')
        if args.dataset in ['window','center']:
            ax.quiver(trajs[i][:,0], trajs[i][:, 1], trajs[i][:, 2], trajs[i][:, 3], color="#0000ff")  # Blue Velocity
            ax.quiver(trajs[i][:, 0], trajs[i][:, 1], accelerations[i][:, 0], accelerations[i][:, 1], color="#ff0000")  # Red Accelaration
        elif args.dataset in ['Dijkstra_random_window', 'Dijkstra_grid_window','Dijkstra_grid_maze','Dijkstra_grid_maze_two_points']:
            ax.quiver(X_unlab[:,0].cpu(), X_unlab[:,1].cpu(), net(X_unlab).cpu().detach().numpy()[:,0], net(X_unlab).cpu().detach().numpy()[:,1],
                      color="#ff0000")  # Blue Unlab
            ax.quiver(X_lab[:,0].cpu(), X_lab[:,1].cpu(), net(X_lab).cpu().detach().numpy()[:,0], net(X_lab).cpu().detach().numpy()[:,1],
                      color="#0000ff")  # Blue Lab


    ax.plot(goal[0], goal[1], 'r*')
    ax.plot(initials[i][0], initials[i][1], 'g*')
    if args.dataset in ['Dijkstra_grid_window', 'Dijkstra_random_window']:
        ax.add_patch(Rectangle((10 - args.width, 0), 2 * args.width, 4,
                               edgecolor='black',
                               facecolor='black',
                               fill=True,
                               lw=5))

        ax.add_patch(Rectangle((10 - args.width, 6), 2 * args.width, 4,
                               edgecolor='black',
                               facecolor='black',
                               fill=True,
                               lw=5))
    if args.dataset in ['Dijkstra_grid_maze']:
        ax.add_patch(Rectangle((5 - args.width, 3), 2 * args.width, 7,
                               edgecolor='black',
                               facecolor='black',
                               fill=True,
                               lw=5))

        ax.add_patch(Rectangle((15 - args.width, 0), 2 * args.width, 7,
                               edgecolor='black',
                               facecolor='black',
                               fill=True,
                               lw=5))
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(0, 10)
    plt.savefig(args.output_dir + '/traj_generated_all.pdf')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Manifold Regularization with Synthetic Data')

    parser.add_argument('--output_dir', type=str, default='trajectories')
    parser.add_argument('--dataset', type=str, default='window')
    parser.add_argument('--n_dim', type=int, default=2, help='Dimension')
    parser.add_argument('--n_train', type=int, default=1)
    parser.add_argument('--n_unlab', type=int, default=100, help='Number of samples per class')
    parser.add_argument('--data_dir', type=str, default='./smooth/data')


    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--regularizer', type=float, default=0.)
    parser.add_argument('--heat_kernel_t', type=float, default=0.05)
    parser.add_argument('--normalize', type=bool, default=True)

    parser.add_argument('--hidden_neurons', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.9)

    parser.add_argument('--resolution', type=float, default=0.4)
    parser.add_argument('--width', type=float, default=1.)



    parser.add_argument('--dual_step_mu', type=float, default=0.01)
    parser.add_argument('--dual_step_lambda', type=float, default=0.1)
    parser.add_argument('--rho_step', type=float, default=0.)
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--clamp', type=float, default=0.4)


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