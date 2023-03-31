# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm

from utils.daml_util import *


class Ensemble_CORAL_with_Dir_mixup(Algorithm):
    def __init__(self, args):
        super(Ensemble_CORAL_with_Dir_mixup, self).__init__(args)
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

        self.bz = args.batch_size
        self.num_classes = args.num_classes

        # hyper-parameters
        self.T = args.T
        self.trade = args.trade
        self.trade2 = args.trade2
        self.trade3 = args.trade3
        self.trade4 = args.trade4
        self.mixup_dir = args.mixup_dir
        self.mixup_dir2 = args.mixup_dir2
        self.stop_gradient = args.stop_gradient
        self.meta_step_size = args.meta_step_size

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
        mixup_loss = 0.0
        meta_train_loss = 0.0

        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        x1 = all_x[0 : self.bz]
        x2 = all_x[self.bz : 2 * self.bz]
        x3 = all_x[2 * self.bz : 3 * self.bz]

        y1 = all_y[0 : self.bz]
        y2 = all_y[self.bz : 2 * self.bz]
        y3 = all_y[2 * self.bz : 3 * self.bz]
        X = [x1, x2, x3]
        Y = [y1, y2, y3]

        all_domain = torch.cat([data[2].to(self.device).long() for data in minibatches])
        ###### Meta Train ##########

        total_all_f_s = [[], [], []]  # model_domains * 3batch_size
        all_one_hot_labels = []  # 3batch_size
        for data_domain, x_s_and_labels_s in enumerate(zip(X, Y)):
            x_s, labels_s = x_s_and_labels_s
            one_hot_labels = create_one_hot(labels_s, self.num_classes, self.device)
            all_one_hot_labels.append(one_hot_labels)

            # compute output
            y_s_distill = []
            for model_domain in range(nmb):
                f_s = self.featurizers[model_domain](x_s)
                y_s = self.classifiers[model_domain](f_s)
                total_all_f_s[model_domain].append(f_s)

                if model_domain != data_domain:
                    y_s_distill.append(y_s)
                    cls_loss += F.cross_entropy(y_s, labels_s)
                else:
                    cls_loss += 3 * F.cross_entropy(y_s, labels_s)
            meta_train_loss += cls_loss

            # Distill
            """
            y_s_distill = torch.stack(y_s_distill)  # 2 * N * C
            y_s_distill = F.softmax(y_s_distill / self.T, dim=2)
            domains = [0] * self.bz
            domains = torch.LongTensor(domains)

            mixup_ratios = get_ratio_mixup_Dirichlet(domains, [1.0, 1.0])
            mixup_ratios = mixup_ratios.to(self.device)  # N * 2
            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)  # 2 * N * 1
            y_s_distill = torch.sum(y_s_distill * mixup_ratios, dim=0)
            kd_loss = DistillKL(y_s_pred, y_s_distill.detach(), self.T)
            meta_train_loss += self.trade2 * kd_loss
            """
        # coral loss
        for model_domain in range(nmb):
            z_model_domain_0 = total_all_f_s[model_domain][0]
            z_model_domain_1 = total_all_f_s[model_domain][1]
            z_model_domain_2 = total_all_f_s[model_domain][2]
            penalty += self.coral(z_model_domain_0, z_model_domain_1)
            penalty += self.coral(z_model_domain_0, z_model_domain_2)
            penalty += self.coral(z_model_domain_1, z_model_domain_2)

        meta_train_loss += penalty

        # Dirichlet Mixup
        all_one_hot_labels = torch.cat(all_one_hot_labels, dim=0)

        for model_domain in range(3):
            # MixUp
            all_f_s = torch.cat(total_all_f_s[model_domain], dim=0)
            domains = [0] * self.bz
            domains = torch.LongTensor(domains)

            all_f_s_1 = all_f_s[(0 * self.bz) : ((0 + 1) * self.bz)]
            all_f_s_2 = all_f_s[(1 * self.bz) : ((1 + 1) * self.bz)]
            all_f_s_3 = all_f_s[(2 * self.bz) : ((2 + 1) * self.bz)]

            all_one_hot_labels_1 = all_one_hot_labels[
                (0 * self.bz) : ((0 + 1) * self.bz)
            ]
            all_one_hot_labels_2 = all_one_hot_labels[
                (1 * self.bz) : ((1 + 1) * self.bz)
            ]
            all_one_hot_labels_3 = all_one_hot_labels[
                (2 * self.bz) : ((2 + 1) * self.bz)
            ]

            mixup_dir_list = [self.mixup_dir2, self.mixup_dir2, self.mixup_dir2]
            mixup_dir_list[model_domain] = self.mixup_dir

            mixup_ratios = get_ratio_mixup_Dirichlet(domains, mixup_dir_list)
            mixup_ratios = mixup_ratios.to(self.device)  # N * 3
            mix_indeces_1 = get_sample_mixup_random(domains)
            mix_indeces_2 = get_sample_mixup_random(domains)
            mix_indeces_3 = get_sample_mixup_random(domains)

            mixup_features = torch.stack(
                [
                    all_f_s_1[mix_indeces_1],
                    all_f_s_2[mix_indeces_2],
                    all_f_s_3[mix_indeces_3],
                ]
            )  # 3 * N * D
            mixup_labels = torch.stack(
                [
                    all_one_hot_labels_1[mix_indeces_1],
                    all_one_hot_labels_2[mix_indeces_2],
                    all_one_hot_labels_3[mix_indeces_3],
                ]
            )  # 3 * N * C

            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)

            mixup_features = torch.sum((mixup_features * mixup_ratios), dim=0)
            mixup_labels = torch.sum((mixup_labels * mixup_ratios), dim=0)

            # mixup_features_predictions = model.heads[model_domain](mixup_features)
            mixup_features_predictions = self.classifiers[model_domain](mixup_features)
            mixup_feature_loss = manual_CE(mixup_features_predictions, mixup_labels)

            mixup_loss += mixup_feature_loss

        meta_train_loss += self.trade * mixup_loss

        total_loss = meta_train_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {
            "class": cls_loss.item(),
            "coral": penalty.item(),
            'mixup': mixup_loss.item(),
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
