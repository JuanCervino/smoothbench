import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle



def create_dataset(dataset, n_dim, n_train, n_unlab, n_test):

    assert dataset in['ellipsoid','two_moons']

    # start with two_moons
    # ignore n_dim
    [X_lab,y_lab] = datasets.make_moons(n_samples=2 * n_train, shuffle=False, noise=0.05, random_state=None)
    [X_unlab, y_unlab] = datasets.make_moons(n_samples=2 * n_unlab, shuffle=False, noise=0.05, random_state=None)

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

def save_output(X_lab,y_lab,X_unlab,y_unlab,dir):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=30, color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=10, color=colors[y_unlab])
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(dir+'/output.pdf')
    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab], open(dir + "/output.p", "wb"))
    pass

def save_lambdas(X_lab,y_lab,X_unlab,y_unlab,dir,lambdas):
    colors = np.array(
        ["#377eb8","#ff7f00","#4daf4a"]
    )
    # plt.figure()
    plt.scatter(X_lab[:, 0], X_lab[:, 1], s=30, color=colors[y_lab])
    plt.scatter(X_unlab[:, 0], X_unlab[:, 1], s=lambdas, color=colors[y_unlab])
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(dir+'/lambdas.pdf')
    plt.close()
    pickle.dump([X_lab,y_lab,X_unlab,y_unlab,lambdas], open(dir + "/lambdas.p", "wb"))
    pass