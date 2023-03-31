# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
from utils.daml_util import (
    get_ratio_mixup_Dirichlet,
    get_sample_mixup_random,
    manual_CE,
)


class CORAL_Dir_mixup(ERM):
    def __init__(self, args):
        super(CORAL_Dir_mixup, self).__init__(args)
        self.args = args
        self.kernel_type = "mean_cov"

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
        bz = len(minibatches[0][2])

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

        ### Dir loss Start ###

        one_hot_targets = [
            F.one_hot(y, num_classes=self.args.num_classes) for y in all_y
        ]
        mixup_dir_list = [1, 1, 1]
        mix_indeces = [
            get_sample_mixup_random(len(minibatch[2])) for minibatch in minibatches
        ]  # 3 * N

        mixup_ratios = get_ratio_mixup_Dirichlet(bz, mixup_dir_list).to(
            self.device
        )  # N * 3
        mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)  # 3 * N * 1

        new_features = [
            feature[mix_index].detach()
            for feature, mix_index in zip(features, mix_indeces)
        ]  # 3 * N * 512
        mixup_features = torch.stack(new_features)  # 3 * N * 512
        '''
        new_targets = [
            target[mix_index] for target, mix_index in zip(targets, mix_indeces)
        ]  # 3 * N
        '''
        new_one_hot_targets = [
            one_hot_target[mix_index]
            for one_hot_target, mix_index in zip(one_hot_targets, mix_indeces)
        ]  # 3 * N * C
        mixup_one_hot_targets = torch.stack(new_one_hot_targets)  # 3 * N * C
        '''
        mixup_features = torch.stack(
            [
                new_features[0],
                new_features[1],
                new_features[2],
            ]
        )  # 3 * N * 512
        '''
        mixup_features = torch.sum((mixup_features * mixup_ratios), dim=0)
        mixup_one_hot_targets = torch.sum((mixup_one_hot_targets * mixup_ratios), dim=0)
        mixup_predictions = self.classifier(mixup_features)
        dirmixup_loss = manual_CE(mixup_predictions, mixup_one_hot_targets)

        ### Dir loss End ###

        # total_loss += dirmixup_loss
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
            # "dirmixup": dirmixup_loss.item(),
            "total": total_loss.item(),
        }
