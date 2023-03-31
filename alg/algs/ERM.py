# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm

import numpy as np


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)  # .to(self.device)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )  # .to(self.device)

        self.network = nn.Sequential(self.featurizer, self.classifier)

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        all_domain = torch.cat([data[2].to(self.device).long() for data in minibatches])
        # all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        # all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        loss = F.cross_entropy(self.predict(all_x), all_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": loss.item()}

    def predict(self, x):
        return self.network(x)
