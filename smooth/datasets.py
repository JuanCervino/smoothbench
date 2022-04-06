from torch.utils.data import Dataset, Subset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as CIFAR10_
from torchvision.datasets import MNIST as MNIST_

# SPLITS = ['train', 'val', 'test']

SPLITS = ['train_labeled','train_unlabeled', 'train_all', 'test']
DATASETS = ['CIFAR10', 'MNIST']

# def to_loaders(all_datasets, hparams):
#
#     def _to_loader(split, dataset):
#         batch_size = hparams['batch_size'] if split == 'train' else 100
#         return DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             num_workers=all_datasets.N_WORKERS,
#             shuffle=(split == 'train'))
#
#     return [_to_loader(s, d) for (s, d) in all_datasets.splits.items()]


def to_loaders(all_datasets, hparams):
    def _to_loader(split, dataset, batch_size ):
        # batch_size = hparams['batch_size'] if split == 'train' else 100
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=all_datasets.N_WORKERS,
            shuffle=(split != 'test'))
    loaders = []
    for (s, d) in all_datasets.splits.items():
        if s in ['train_labeled','train_unlabeled']:
            if d == None:
                loaders += [None]
            else:
                loaders += [_to_loader(s, d, hparams['batch_size']) ]
        elif s == 'train_all':
            loaders += [_to_loader(s, d, hparams['unlab_batch_size'])]
        else:
            loaders += [_to_loader(s, d, 100)]
    return loaders



class AdvRobDataset(Dataset):

    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    NUM_CLASSES = None       # Subclasses should override
    N_EPOCHS = None          # Subclasses should override
    CHECKPOINT_FREQ = None   # Subclasses should override
    LOG_INTERVAL = None      # Subclasses should override
    HAS_LR_SCHEDULE = False  # Default, subclass may override

    def __init__(self):
        self.splits = dict.fromkeys(SPLITS)


class CIFAR10(AdvRobDataset):
 
    INPUT_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10
    N_EPOCHS = 200
    CHECKPOINT_FREQ = 10
    EPSILON = 8/ 255.
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = True

    # test adversary parameters
    ADV_STEP_SIZE = 2/255.
    N_ADV_STEPS = 20

    def __init__(self, root, per_labeled = 1, transform = True):
        super(CIFAR10, self).__init__()
        if transform:
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            train_transforms = transforms.ToTensor()

        test_transforms = transforms.ToTensor()

        train_data = CIFAR10_(root, train=True, transform=train_transforms, download = True) # Juan Here
        self.splits['train_all'] = train_data
        labeled_len = int(per_labeled * len(train_data))
        unlabeled_len = len(train_data) - labeled_len
        if per_labeled != 1:
            self.splits['train_labeled'], self.splits['train_unlabeled'] = random_split(train_data,[labeled_len, unlabeled_len])
        else:
            self.splits['train_labeled'] = train_data

        # train_data = CIFAR10_(root, train=True, transform=train_transforms)
        # self.splits['val'] = Subset(train_data, range(45000, 50000))

        self.splits['test'] = CIFAR10_(root, train=False, transform=test_transforms)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):

        lr = hparams['learning_rate']
        if epoch >= 75:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 90:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 100:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class MNIST(AdvRobDataset):

    INPUT_SHAPE = (1, 28, 28)
    NUM_CLASSES = 10
    N_EPOCHS = 50
    CHECKPOINT_FREQ = 10
    EPSILON = 0.3
    LOG_INTERVAL = 100
    HAS_LR_SCHEDULE = True

    # test adversary parameters
    ADV_STEP_SIZE = 0.1
    N_ADV_STEPS = 10

    def __init__(self, root, per_labeled = 1):
        super(MNIST, self).__init__()
        
        xforms = transforms.ToTensor()

        train_data = MNIST_(root, train=True, transform=xforms, download = True)
        # self.splits['train'] = Subset(train_data, range(54000))
        #
        # train_data = MNIST_(root, train=True, transform=xforms)
        # self.splits['val'] = Subset(train_data, range(54000, 60000))
        self.splits['train_all'] = train_data
        labeled_len = int(per_labeled * len(train_data))
        unlabeled_len = len(train_data) - labeled_len
        self.splits['train_labeled'], self.splits['train_unlabeled'] = random_split(train_data,[labeled_len, unlabeled_len])


        self.splits['test'] = MNIST_(root, train=False, transform=xforms)

    @staticmethod
    def adjust_lr(optimizer, epoch, hparams):

        lr = hparams['learning_rate']
        if epoch >= 55:
            lr = hparams['learning_rate'] * 0.1
        if epoch >= 75:
            lr = hparams['learning_rate'] * 0.01
        if epoch >= 90:
            lr = hparams['learning_rate'] * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr