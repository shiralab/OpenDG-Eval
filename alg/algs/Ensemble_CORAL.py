# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class Ensemble_CORAL(Algorithm):
    def __init__(self, args):
        super(Ensemble_CORAL, self).__init__(args)
        self.featurizer_1 = get_fea(args)
        self.featurizer_2 = get_fea(args)
        self.featurizer_3 = get_fea(args)
        self.classifier_1 = common_network.feat_classifier(
            args.num_classes, self.featurizer_1.in_features, args.classifier
        )
        self.classifier_2 = common_network.feat_classifier(
            args.num_classes, self.featurizer_2.in_features, args.classifier
        )
        self.classifier_3 = common_network.feat_classifier(
            args.num_classes, self.featurizer_3.in_features, args.classifier
        )
        self.featurizers = [self.featurizer_1, self.featurizer_2, self.featurizer_3]
        self.classifiers = [self.classifier_1, self.classifier_2, self.classifier_3]
        self.args = args
        self.kernel_type = "mean_cov"
        self.bz = args.batch_size

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        nmb = len(minibatches)  # num of domain
        cls_loss = 0
        penalty = 0
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        all_domain = torch.cat([data[2].to(self.device).long() for data in minibatches])

        for model_domain in range(nmb):
            all_z = self.featurizers[model_domain](all_x)
            all_y_pred = self.classifiers[model_domain](all_z)
            y1_pred = all_y_pred[0 : self.bz]
            y2_pred = all_y_pred[self.bz : 2 * self.bz]
            y3_pred = all_y_pred[2 * self.bz : 3 * self.bz]
            y1 = all_y[0 : self.bz]
            y2 = all_y[self.bz : 2 * self.bz]
            y3 = all_y[2 * self.bz : 3 * self.bz]
            cls_loss += (
                F.cross_entropy(y1_pred, y1)
                + F.cross_entropy(y2_pred, y2)
                + F.cross_entropy(y3_pred, y3)
            )
            if model_domain == 1:
                cls_loss += 2 * F.cross_entropy(y1_pred, y1)
            if model_domain == 2:
                cls_loss += 2 * F.cross_entropy(y2_pred, y2)
            if model_domain == 3:
                cls_loss += 2 * F.cross_entropy(y3_pred, y3)
            # penalty
            z1 = all_z[0 : self.bz]
            z2 = all_z[self.bz : 2 * self.bz]
            z3 = all_z[2 * self.bz : 3 * self.bz]
            penalty += self.coral(z1, z2) + self.coral(z1, z3) + self.coral(z2, z3)

        total_loss = cls_loss + penalty

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "class": cls_loss.item(),
            "coral": penalty,
            "total": total_loss.item(),
        }

    def predict(self, x):
        logits = []
        for i in range(3):
            z = self.featurizers[i](x)
            logits.append(self.classifiers[i](z))

        mean_logits = torch.mean(torch.stack(logits), dim=0)
        # assert logits[0].shape == mean_logits.shape  # (N,known_C)
        return mean_logits
