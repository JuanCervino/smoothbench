import numpy as np
import torch
from sklearn.metrics import pairwise_distances

#  From https://github.com/tegusi/RGCNN

def get_pairwise_euclidean_distance_matrix(tensor):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    flat = tensor.flatten(start_dim = 1)
    adj_matrix = torch.cdist(flat,flat)

    return adj_matrix

def get_pairwise_distance_matrix(tensor, t):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
        t: scalar
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    # t = 10.55 # Average distance of CIFAR10
    # t = 10.55**2 # Average distance square of CIFAR10

    flat = tensor.flatten(start_dim = 1)
    adj_matrix = torch.cdist(flat,flat)
    adj_matrix = torch.square(adj_matrix)

    adj_matrix = torch.div( adj_matrix, -4*t)
    adj_matrix = torch.exp(adj_matrix)
    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements


    return adj_matrix

def get_laplacian(imgs, normalize = False, heat_kernel_t = 10):
    """Compute pairwise distance of a point cloud.

    Args:
        pairwise distance: tensor (batch_size, num_points, num_points)

    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    adj_matrix = get_pairwise_distance_matrix(imgs, heat_kernel_t)
    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to('cuda') # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    return L

