# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network, ARPL_network
from alg.algs.base import Algorithm

import numpy as np


class ARPL(Algorithm):
    def __init__(self, args):
        super(ARPL, self).__init__(args)
        self.featurizer = get_fea(args)
        self.criterion = ARPL_network.ARPLoss(args, self.featurizer.in_features)
        self.args = args

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        all_z = self.featurizer(all_x)

        # loss = F.cross_entropy(self.predict(all_x), all_y)
        _, cls_loss, margin_loss = self.criterion(all_z, all_y)
        loss = cls_loss + self.args.weight_pl * margin_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {
            "class": cls_loss.item(),
            "margin": margin_loss.item(),
            "total": loss.item(),
        }

    def predict(self, x):
        logit = self.criterion(self.featurizer(x), None)
        return logit
