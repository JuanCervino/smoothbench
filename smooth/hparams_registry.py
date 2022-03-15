import numpy as np

from smooth.lib import misc
from smooth import datasets

def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}

def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}

def _hparams(algorithm: str, dataset: str, random_seed: int):
    """Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""

        assert(name not in hparams)
        random_state = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    # Juan added this
    _hparam('regularizer', 0.1, lambda r: int(2 ** r.uniform(0.0001,1)))
    _hparam('normalize', False, lambda r: True if r.uniform()>1/2 else False  )

    _hparam('batch_size', 128, lambda r: int(2 ** r.uniform(3, 8)))

    # optimization
    _hparam('learning_rate', 0.01, lambda r: 10 ** r.uniform(-4.5, -2.5))
    _hparam('sgd_momentum', 0.9, lambda r: r.uniform(0.8, 0.95))
    _hparam('weight_decay', 3.5e-3, lambda r: 10 ** r.uniform(-6, -3))

    if dataset == 'MNIST':
        _hparam('epsilon', 0.3, lambda r: 0.3)
    else:
        _hparam('epsilon', 0.031, lambda r: 0.031)

    # Algorithm specific

    ##### PGD #####
    if dataset == 'MNIST':
        _hparam('pgd_n_steps', 7, lambda r: 7)
        _hparam('pgd_step_size', 0.1, lambda r: 0.1)
    elif dataset == 'CIFAR10':
        _hparam('pgd_n_steps', 10, lambda r: 10)
        _hparam('pgd_step_size', 0.007, lambda r: 0.007)

    ##### TRADES #####
    if dataset == 'MNIST':
        _hparam('trades_n_steps', 7, lambda r: 7)
        _hparam('trades_step_size', 0.1, lambda r: r.uniform(0.01, 0.1))
        _hparam('trades_beta', 1.0, lambda r: r.uniform(0.1, 10.0))
    elif dataset == 'CIFAR10':
        _hparam('trades_n_steps', 10, lambda r: 7)
        _hparam('trades_step_size', 2/255., lambda r: r.uniform(0.01, 0.1))
        _hparam('trades_beta', 6.0, lambda r: r.uniform(0.1, 10.0))

    ##### MART #####
    if dataset == 'MNIST':
        _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))
    elif dataset == 'CIFAR10':
        _hparam('mart_beta', 5.0, lambda r: r.uniform(0.1, 10.0))

    ##### Gaussian DALE #####
    if dataset == 'MNIST':
        _hparam('g_dale_n_steps', 7, lambda r: 7)
        _hparam('g_dale_step_size', 0.1, lambda r: 0.1)
        _hparam('g_dale_noise_coeff', 0.001, lambda r: 10 ** r.uniform(-6.0, -2.0))
    elif dataset == 'CIFAR10':
        _hparam('g_dale_n_steps', 10, lambda r: 10)
        _hparam('g_dale_step_size', 0.007, lambda r: 0.007)
        _hparam('g_dale_noise_coeff', 1e-2, lambda r: 1e-2)
    _hparam('g_dale_nu', 0.1, lambda r: 0.1)

    # DALE (Laplacian-HMC)
    if dataset == 'MNIST':
        _hparam('l_dale_n_steps', 7, lambda r: 7)
        _hparam('l_dale_step_size', 0.1, lambda r: 0.1)
        _hparam('l_dale_noise_coeff', 0.001, lambda r: 10 ** r.uniform(-6.0, -2.0))
    elif dataset == 'CIFAR10':
        _hparam('l_dale_n_steps', 10, lambda r: 10)
        _hparam('l_dale_step_size', 0.007, lambda r: 0.007)
        _hparam('l_dale_noise_coeff', 1e-2, lambda r: 1e-2)
    _hparam('l_dale_nu', 0.1, lambda r: 0.1)

    # DALE-PD (Gaussian-HMC)
    _hparam('g_dale_pd_step_size', 0.01, lambda r: 0.01)
    _hparam('g_dale_pd_margin', 0.1, lambda r: 0.1)

    return hparams

def test_hparams(algorithm: str, dataset: str):

    hparams = {}

    def _hparam(name, default_val):
        """Define a hyperparameter for test adversaries."""

        assert(name not in hparams)
        hparams[name] = default_val

    if dataset == 'MNIST':
        _hparam('epsilon', 0.3)
    elif dataset == 'CIFAR10':
        _hparam('epsilon', 8/255.)

    ##### PGD #####
    if dataset == 'MNIST':
        _hparam('pgd_n_steps', 20)
        _hparam('pgd_step_size', 0.1)
    elif dataset == 'CIFAR10':
        _hparam('pgd_n_steps', 20)
        _hparam('pgd_step_size', 0.003)

    ##### TRADES #####
    if dataset == 'MNIST':
        _hparam('trades_n_steps', 10)
        _hparam('trades_step_size', 0.1)
    elif dataset == 'CIFAR10':
        _hparam('trades_n_steps', 20)
        _hparam('trades_step_size', 2/255.)

    return hparams