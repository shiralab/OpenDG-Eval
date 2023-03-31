# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm

from utils.daml_util import *


class DAML_wo_distill(Algorithm):
    def __init__(self, args):
        super(DAML_wo_distill, self).__init__(args)
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

        self.bz = args.batch_size // 2
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

    def update(self, minibatches, opt, sch):
        nmb = len(minibatches)  # num of domain
        cls_loss = 0
        penalty = 0
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])
        x1_meta_tr = all_x[0 : self.bz]
        x1_meta_vl = all_x[self.bz : 2 * self.bz]
        x2_meta_tr = all_x[2 * self.bz : 3 * self.bz]
        x2_meta_vl = all_x[3 * self.bz : 4 * self.bz]
        x3_meta_tr = all_x[4 * self.bz : 5 * self.bz]
        x3_meta_vl = all_x[5 * self.bz : 6 * self.bz]

        y1_meta_tr = all_y[0 : self.bz]
        y1_meta_vl = all_y[self.bz : 2 * self.bz]
        y2_meta_tr = all_y[2 * self.bz : 3 * self.bz]
        y2_meta_vl = all_y[3 * self.bz : 4 * self.bz]
        y3_meta_tr = all_y[4 * self.bz : 5 * self.bz]
        y3_meta_vl = all_y[5 * self.bz : 6 * self.bz]

        X_meta_tr = [x1_meta_tr, x2_meta_tr, x3_meta_tr]
        Y_meta_tr = [y1_meta_tr, y2_meta_tr, y3_meta_tr]
        X_meta_vl = [x1_meta_vl, x2_meta_vl, x3_meta_vl]
        Y_meta_vl = [y1_meta_vl, y2_meta_vl, y3_meta_vl]

        all_domain = torch.cat([data[2].to(self.device).long() for data in minibatches])
        ###### Meta Train ##########
        meta_train_loss = 0.0
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None

        total_all_f_s = [[], [], []]  # model_domains * 3batch_size
        all_one_hot_labels = []  # 3batch_size
        for data_domain, x_s_and_labels_s in enumerate(zip(X_meta_tr, Y_meta_tr)):
            x_s, labels_s = x_s_and_labels_s
            one_hot_labels = create_one_hot(labels_s, self.num_classes, self.device)
            all_one_hot_labels.append(one_hot_labels)

            # compute output
            y_s_distill = []
            for model_domain in range(nmb):
                f_s = self.featurizers[model_domain](x_s)
                y_s = self.classifiers[model_domain](f_s)
                # y_s, f_s = model(x_s, domain=model_domain)
                if model_domain != data_domain:
                    y_s_distill.append(y_s)
                else:
                    y_s_pred = y_s
                total_all_f_s[model_domain].append(f_s)

            cls_loss = F.cross_entropy(y_s_pred, labels_s)
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
        # Dirichlet Mixup

        all_one_hot_labels = torch.cat(all_one_hot_labels, dim=0)
        mixup_loss = 0.0

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

        #############################################################
        # -----------------  Meta Objective -----------------------------------
        meta_val_loss = 0.0

        grad = torch.autograd.grad(meta_train_loss, fast_parameters, create_graph=True)

        if self.stop_gradient:
            grad = [g.detach() for g in grad]

        fast_parameters = []
        for k, weight in enumerate(self.parameters()):
            if weight.fast is None:
                weight.fast = weight - self.meta_step_size * grad[k]
            else:
                weight.fast = weight.fast - self.meta_step_size * grad[k]
            fast_parameters.append(weight.fast)

        total_all_f_s = [[], [], []]  # model_domains * 3batch_size
        all_one_hot_labels = []  # 3batch_size

        for data_domain, x_s_and_labels_s in enumerate(zip(X_meta_vl, Y_meta_vl)):
            x_s, labels_s = x_s_and_labels_s
            # x_s, labels_s, _ = next(train_source_iter)
            # x_s = x_s.to(self.device)
            # labels_s = labels_s.to(device)
            one_hot_labels = create_one_hot(labels_s, self.num_classes, self.device)
            all_one_hot_labels.append(one_hot_labels)

            # compute output
            y_s_list = []
            for model_domain in range(3):
                f_s = self.featurizers[model_domain](x_s)
                y_s = self.classifiers[model_domain](f_s)
                # y_s, f_s = model(x_s, domain=model_domain)
                y_s_list.append(y_s)
                total_all_f_s[model_domain].append(f_s)

                if model_domain != data_domain:
                    cls_loss = F.cross_entropy(y_s, labels_s)
                    meta_val_loss = meta_val_loss + self.trade3 * cls_loss

        all_one_hot_labels = torch.cat(all_one_hot_labels, dim=0)

        # Dirichelet Mixup

        mixup_loss_dirichlet = 0.0

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

            mixup_dir_list = [self.mixup_dir, self.mixup_dir, self.mixup_dir]
            mixup_dir_list[model_domain] = self.mixup_dir2

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
            mixup_feature_loss_dirichlet = manual_CE(
                mixup_features_predictions, mixup_labels
            )

            mixup_loss_dirichlet += mixup_feature_loss_dirichlet

        meta_val_loss += self.trade4 * mixup_loss_dirichlet

        # meta_val_loss = torch.tensor([0], device=self.device)
        total_loss = meta_train_loss + meta_val_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        if sch:
            sch.step()

        return {
            "meta_train": meta_train_loss.item(),
            "meta_val": meta_val_loss.item(),
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
