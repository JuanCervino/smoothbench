import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle
import torch



def create_dataset(dataset, n_dim, n_train, n_unlab, n_test, noise):

    assert dataset in['ellipsoid','two_moons']

    # start with two_moons
    # ignore n_dim
    [X_lab,y_lab] = datasets.make_moons(n_samples=2 * n_train, shuffle=False, noise=noise, random_state=None)
    [X_unlab, y_unlab] = datasets.make_moons(n_samples=2 * n_unlab, shuffle=False, noise=noise, random_state=None)

    return [X_lab,y_lab,X_unlab,y_unlab]

def save_dataset(X_lab,y_lab,X_unlab,y_unlab,dir):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=20, color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=10, color="#4daf4a")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(dir+'/dataset.pdf')
    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab], open(dir + "/dataset.p", "wb"))
    pass

def save_output(X_lab,y_lab,X_unlab,y_unlab,dir,name=None):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=30, color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=10, color=colors[y_unlab])
    plt.grid(True)
    plt.axis('equal')
    if name!=None:
        plt.savefig(dir+'/output'+str(name)+'.pdf')
    else:
        plt.savefig(dir+'/output.pdf')

    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab], open(dir + "/output.p", "wb"))
    pass

def save_output_allspace(net,X_lab,y_lab,X_unlab,y_unlab,dir,name=None):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    x_points = 75
    y_points = 75
    x_span = np.linspace(-1.5,2.5, x_points)
    y_span = np.linspace(-1,1.5, y_points)
    xx, yy = np.meshgrid(x_span, y_span)
    grid_np = np.c_[xx.ravel(), yy.ravel()]
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to('cuda')
    out = net(grid)
    out = torch.nn.functional.softmax(out).cpu().detach().numpy()
    label = out[:,0]
    # label = np.argmax(out, axis=1).astype(int)

    z = label.reshape((x_points,y_points))
    # plt.scatter(grid_np[:, 0], grid_np[:, 1], s=30, color=colors[label])

    # plt.figure()
    CS = plt.contourf(xx, yy,z,cmap ='RdGy', vmin=0., vmax=1., levels = np.linspace(0,1,11))
    plt.colorbar(CS)
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=30, marker='^', color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=10, color=colors[y_unlab])

    plt.grid(True)
    plt.axis('equal')
    if name!=None:
        plt.savefig(dir+'/output'+str(name)+'.pdf')
    else:
        plt.savefig(dir+'/output.pdf')

    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab], open(dir + "/output.p", "wb"))
    pass

def save_lambdas(X_lab,y_lab,X_unlab,y_unlab,lambdas,dir,name=None):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    max_lambda = np.max(lambdas)
    lambdas = 60 * lambdas/max_lambda
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=30, color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=lambdas, color=colors[y_unlab])
    plt.grid(True)
    plt.axis('equal')
    if name!=None:
        plt.savefig(dir+'/lambdas'+str(name)+'.pdf')
    else:
        plt.savefig(dir+'/lambdas.pdf')

    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab,lambdas], open(dir + "/lambdas.p", "wb"))
    pass

def save_lambdas_all_space(net, X_lab,y_lab,X_unlab,y_unlab,lambdas,dir,name=None):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    x_points = 60
    y_points = 60
    x_span = np.linspace(-1.5,2.5, x_points)
    y_span = np.linspace(-1,1.5, y_points)
    xx, yy = np.meshgrid(x_span, y_span)
    grid_np = np.c_[xx.ravel(), yy.ravel()]
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).to('cuda')
    out = net(grid)
    out = torch.nn.functional.softmax(out).cpu().detach().numpy()
    label = out[:,0]
    # label = np.argmax(out, axis=1).astype(int)

    z = label.reshape((x_points,y_points))
    # plt.scatter(grid_np[:, 0], grid_np[:, 1], s=30, color=colors[label])

    # plt.figure()
    max_lambda = np.max(lambdas)
    lambdas = 60 * lambdas/max_lambda
    CS = plt.contourf(xx, yy,z,cmap ='RdGy', vmin=0., vmax=1., levels = np.linspace(0,1,11))
    plt.colorbar(CS)
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=30, marker='^', color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=lambdas, color=colors[y_unlab])
    plt.xlim(-1.5,2.5)
    plt.ylim(-1,1.5)
    plt.grid(True)
    plt.axis('equal')
    if name!=None:
        plt.savefig(dir+'/lambdas'+str(name)+'.pdf')
    else:
        plt.savefig(dir+'/lambdas.pdf')

    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab,lambdas], open(dir + "/lambdas.p", "wb"))
    pass