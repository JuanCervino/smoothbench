import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd

from smooth import networks
from smooth import optimizers
from smooth import attacks
from smooth import laplacian
from smooth.lib import meters

ALGORITHMS = [
    'ERM',
    'PGD',
    'FGSM',
    'TRADES',
    'ALP',
    'CLP',
    'Gaussian_DALE',
    'Laplacian_DALE',
    'Gaussian_DALE_PD'
]

class Algorithm(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.classifier = networks.Classifier(
            input_shape, num_classes, hparams)
        self.optimizer = optimizers.Optimizer(
            self.classifier, hparams)
        self.device = device
        
        self.meters = OrderedDict()
        self.meters['loss'] = meters.AverageMeter()
        self.meters_df = None

    def step(self, imgs, labels):
        raise NotImplementedError

    def predict(self, imgs):
        return self.classifier(imgs)

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()

    def meters_to_df(self, epoch):
        if self.meters_df is None:
            columns = ['Epoch'] + list(self.meters.keys())
            self.meters_df = pd.DataFrame(columns=columns)

        values = [epoch] + [m.avg for m in self.meters.values()]
        self.meters_df.loc[len(self.meters_df)] = values
        return self.meters_df



class DISTANCE(Algorithm):

    def __init__(self, input_shape, num_classes, hparams, device):
        super(DISTANCE, self).__init__(input_shape, num_classes, hparams, device)

    def step(self, imgs, labels):
        raise NotImplementedError

    # JUAN ADDED THIS
    def get_middle_layer(self,imgs):
        activation = {}

        def get_activation(name):
            def hook(classifier, imgs, output):
                activation[name] = output.detach()

            return hook

        self.classifier.layer4.register_forward_hook(get_activation('layer4'))
        output = self.classifier(imgs)
        return activation['layer4']



class ERM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM, self).__init__(input_shape, num_classes, hparams, device)

    def step(self, imgs, labels):
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class ERM_AVG_LIP_RND(Algorithm):
    # This code implements the avg lipschitz algorithm constructing the laplacian with random samples
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_AVG_LIP_RND, self).__init__(input_shape, num_classes, hparams, device)
        self.regularizer = hparams['regularizer'] # Regularizer for the Avg Lip
        self.normalize = hparams['normalize']
        self.heat_kernel_t = hparams['heat_kernel_t']

    def step(self, imgs, labels, imgs_unlab):
        # Change this to add the unlabeled data
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)

        L = laplacian.get_laplacian(imgs_unlab, self.normalize, self.heat_kernel_t)
        f = self.predict(imgs_unlab)

        loss += self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
        loss.backward()
        self.optimizer.step()

        print(F.cross_entropy(self.predict(imgs), labels),self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))))

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class ERM_AVG_LIP_KNN(Algorithm):
    # This code implements the avg lipschitz algorithm constructing the laplacian with random samples
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_AVG_LIP_KNN, self).__init__(input_shape, num_classes, hparams, device)
        self.regularizer = hparams['regularizer'] # Regularizer for the Avg Lip
        self.normalize = hparams['normalize']
        self.heat_kernel_t = hparams['heat_kernel_t']

    def step(self, imgs, labels, imgs_unlab_lst):
        # Change this to add the unlabeled data
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)

        for ele in imgs_unlab_lst:
            L = laplacian.get_laplacian(ele, self.normalize, self.heat_kernel_t)
            f = self.predict(ele)

            loss += self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
        loss.backward()
        self.optimizer.step()

        print(F.cross_entropy(self.predict(imgs), labels),self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))))

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class ERM_AVG_LIP_CHEAT(Algorithm):
    # This code implements the avg lipschitz algorithm constructing the laplacian with random samples
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_AVG_LIP_CHEAT, self).__init__(input_shape, num_classes, hparams, device)
        self.regularizer = hparams['regularizer'] # Regularizer for the Avg Lip
        self.normalize = hparams['normalize']
        self.heat_kernel_t = hparams['heat_kernel_t']

    def step(self, imgs, labels, imgs_unlab):
        # Change this to add the unlabeled data
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)

        L = laplacian.get_laplacian(imgs_unlab, self.normalize, self.heat_kernel_t)
        f = self.predict(imgs_unlab)

        loss += self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
        loss.backward()
        self.optimizer.step()

        print(F.cross_entropy(self.predict(imgs), labels),self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))))

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class ERM_AVG_LIP_TRANSFORM(Algorithm):
    # This code implements the avg lipschitz algorithm constructing the laplacian with random samples
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_AVG_LIP_TRANSFORM, self).__init__(input_shape, num_classes, hparams, device)
        self.regularizer = hparams['regularizer'] # Regularizer for the Avg Lip
        self.normalize = hparams['normalize']
        self.heat_kernel_t = hparams['heat_kernel_t']

    def step(self, imgs, labels, imgs_unlab):
        # Change this to add the unlabeled data
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)

        L = laplacian.get_laplacian(imgs_unlab, self.normalize, self.heat_kernel_t)
        f = self.predict(imgs_unlab)

        loss += self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
        loss.backward()
        self.optimizer.step()

        print(F.cross_entropy(self.predict(imgs), labels),self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))))

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class ERM_AVG_LIP_CNN_METRIC(Algorithm):
    # This code implements the avg lipschitz algorithm constructing the laplacian with random samples
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_AVG_LIP_CNN_METRIC, self).__init__(input_shape, num_classes, hparams, device)
        self.regularizer = hparams['regularizer'] # Regularizer for the Avg Lip
        self.normalize = hparams['normalize']
        self.heat_kernel_t = hparams['heat_kernel_t']

    def step(self, imgs, labels, imgs_unlab):
        # Change this to add the unlabeled data
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)

        L = laplacian.get_laplacian(imgs_unlab, self.normalize, self.heat_kernel_t)
        f = self.predict(imgs_unlab)

        loss += self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f)))
        loss.backward()
        self.optimizer.step()

        print(F.cross_entropy(self.predict(imgs), labels),self.regularizer * torch.trace(torch.matmul(f.transpose(0,1),torch.matmul(L, f))))

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class ERM_LAMBDA_LIP(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ERM_LAMBDA_LIP, self).__init__(input_shape, num_classes, hparams, device)
        self.regularizer = hparams['regularizer'] # Regularizer for the Avg Lip
        self.normalize = hparams['normalize']

    def step(self, imgs, labels, imgs_unlab, lambdas):
        # Change this to add the unlabeled data
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(imgs), labels)
        L = laplacian.get_laplacian(imgs_unlab, self.normalize)
        f = self.predict(imgs_unlab)
        f_lambda = f.transpose(0,1) * lambdas
        loss += self.regularizer * torch.trace(torch.matmul(f_lambda,torch.matmul(L, f)))
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class PGD(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(PGD, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class FGSM(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(FGSM, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.FGSM_Linf(self.classifier, self.hparams, device)

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.predict(adv_imgs), labels)
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))

class TRADES(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(TRADES, self).__init__(input_shape, num_classes, hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')  # TODO(AR): let's write a method to do the log-softmax part
        self.attack = attacks.TRADES_Linf(self.classifier, self.hparams, device)
        
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['invarinace loss'] = meters.AverageMeter()

    def step(self, imgs, labels):

        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        robust_loss = self.kl_loss_fn(
            F.log_softmax(self.predict(adv_imgs), dim=1),
            F.softmax(self.predict(imgs), dim=1))
        total_loss = clean_loss + self.hparams['trades_beta'] * robust_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['invariance loss'].update(robust_loss.item(), n=imgs.size(0))

        return {'loss': total_loss.item()}

class LogitPairingBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(LogitPairingBase, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)
        self.meters['logit loss'] = meters.AverageMeter()

    def pairing_loss(self, imgs, adv_imgs):
        logit_diff = self.predict(adv_imgs) - self.predict(imgs)
        return torch.norm(logit_diff, dim=1).mean()

class ALP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(ALP, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = robust_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['logit loss'].update(logit_pairing_loss.item(), n=imgs.size(0))

class CLP(LogitPairingBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(CLP, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

        self.meters['clean loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        logit_pairing_loss = self.pairing_loss(imgs, adv_imgs)
        total_loss = clean_loss + logit_pairing_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['logit loss'].update(logit_pairing_loss.item(), n=imgs.size(0))

class MART(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(MART, self).__init__(input_shape, num_classes, hparams, device)
        self.kl_loss_fn = nn.KLDivLoss(reduction='none')
        self.attack = attacks.PGD_Linf(self.classifier, self.hparams, device)

        self.meters['robust loss'] = meters.AverageMeter()
        self.meters['invariance loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_output = self.classifier(imgs)
        adv_output = self.classifier(adv_imgs)
        adv_probs = F.softmax(adv_output, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_label = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(adv_output, labels) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_label)
        nat_probs = F.softmax(clean_output, dim=1)
        true_probs = torch.gather(nat_probs, 1, (labels.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / imgs.size(0)) * torch.sum(
            torch.sum(self.kl_loss_fn(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + self.hparams['mart_beta'] * loss_robust
        loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(loss_robust.item(), n=imgs.size(0))
        self.meters['invariance loss'].update(loss_adv.item(), n=imgs.size(0))


class MMA(Algorithm):
    pass

class Gaussian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Gaussian_DALE, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['g_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class Laplacian_DALE(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Laplacian_DALE, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Laplacian_Linf(self.classifier, self.hparams, device)
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.hparams['l_dale_nu'] * clean_loss
        total_loss.backward()
        self.optimizer.step()

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))

class PrimalDualBase(Algorithm):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(PrimalDualBase, self).__init__(input_shape, num_classes, hparams, device)
        self.dual_params = {'dual_var': torch.tensor(1.0).to(self.device)}
        self.meters['clean loss'] = meters.AverageMeter()
        self.meters['robust loss'] = meters.AverageMeter()
        self.meters['dual variable'] = meters.AverageMeter()

class Gaussian_DALE_PD(PrimalDualBase):
    def __init__(self, input_shape, num_classes, hparams, device):
        super(Gaussian_DALE_PD, self).__init__(input_shape, num_classes, hparams, device)
        self.attack = attacks.LMC_Gaussian_Linf(self.classifier, self.hparams, device)
        self.pd_optimizer = optimizers.PrimalDualOptimizer(
            parameters=self.dual_params,
            margin=self.hparams['g_dale_pd_margin'],
            eta=self.hparams['g_dale_pd_step_size'])

    def step(self, imgs, labels):
        adv_imgs = self.attack(imgs, labels)
        self.optimizer.zero_grad()
        clean_loss = F.cross_entropy(self.predict(imgs), labels)
        robust_loss = F.cross_entropy(self.predict(adv_imgs), labels)
        total_loss = robust_loss + self.dual_params['dual_var'] * clean_loss
        total_loss.backward()
        self.optimizer.step()
        self.pd_optimizer.step(clean_loss.detach())

        self.meters['loss'].update(total_loss.item(), n=imgs.size(0))
        self.meters['clean loss'].update(clean_loss.item(), n=imgs.size(0))
        self.meters['robust loss'].update(robust_loss.item(), n=imgs.size(0))
        self.meters['dual variable'].update(self.dual_params['dual_var'].item(), n=1)
