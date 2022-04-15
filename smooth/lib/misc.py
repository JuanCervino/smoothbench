import torch
import hashlib
import sys
from functools import wraps
from time import time
import pandas as pd

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts:.3f} sec')
        return result
    return wrap

def seed_hash(*args):
    """Derive an integer hash from all args, for use as a random seed."""

    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

@torch.no_grad()
def accuracy(algorithm, loader, device):
    correct, total = 0, 0

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    algorithm.train()

    return 100. * correct / total


@torch.no_grad()
def class_wise_accuracy(algorithm, loader, device, dataset):
    assert dataset in ['MNIST','CIFAR10','CIFAR100','STL10','CIFAR100coarse']
    if dataset in ['MNIST','CIFAR10','STL10']:
        n_classes = 10
    elif dataset == 'CIFAR100':
        n_classes = 100
    elif dataset == 'CIFAR100coarse':
        n_classes = 20

    correct_hist = torch.zeros(n_classes).to(device)
    label_hist = torch.zeros(n_classes).to(device)

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = algorithm.predict(imgs)

        label_hist += torch.histc(labels, n_classes, max=n_classes)
        output = algorithm.predict(imgs)
        pred = output.argmax(dim=1, keepdim=True)
        correct_index = pred.eq(labels.view_as(pred)).squeeze()
        label_correct = labels[correct_index]
        correct_hist += torch.histc(label_correct, n_classes, max=n_classes)
    correct_rate_hist = correct_hist / label_hist
    algorithm.train()

    return [correct_rate_hist.cpu().numpy()]


def adv_accuracy(algorithm, loader, device, attack):
    correct, total = 0, 0

    algorithm.eval()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack(imgs, labels)

        with torch.no_grad():
            output = algorithm.predict(adv_imgs)
            pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.size(0)
    algorithm.train()

    return 100. * correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()