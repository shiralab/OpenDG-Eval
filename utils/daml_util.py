import torch
import torch.nn.functional as F
import numpy as np


def create_one_hot(y, classes, device):
    y_onehot = torch.LongTensor(y.size(0), classes).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def get_sample_mixup_random(domains):
    indeces = torch.randperm(domains.size(0))
    return indeces.long()


def get_ratio_mixup_Dirichlet(domains, mixup_dir_list):
    RG = np.random.default_rng()
    return torch.from_numpy(
        RG.dirichlet(mixup_dir_list, size=domains.size(0))
    ).float()  # N * 3


def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions, dim=1), dim=1))
    return loss


def DistillKL(y_s, y_t, T):
    """KL divergence for distillation"""
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = y_t
    loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]
    return loss
