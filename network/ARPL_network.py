import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Dist(nn.Module):
    def __init__(self, num_classes, num_centers=1, feat_dim=2, init='random'):
        super(Dist, self).__init__()
        self.feat_dim = feat_dim  # 128 or 512
        self.num_classes = num_classes
        self.num_centers = num_centers
        print(
            f'self.feat_dim:{self.feat_dim}, self.num_classes:{self.num_classes}, self.num_centers:{self.num_centers}'
        )

        if init == 'random':
            # self.centers.shape = torch.Size([#known classes * #centers, #feat_dim]). e.g (6,512)
            self.centers = nn.Parameter(
                0.1 * torch.randn(num_classes * num_centers, self.feat_dim)
            )

        else:
            self.centers = nn.Parameter(
                torch.Tensor(num_classes * num_centers, self.feat_dim)
            )
            self.centers.data.fill_(0)

    def forward(self, features, center=None, metric='l2'):
        if metric == 'l2':  # default
            # f_2.shape = torch.Size([N, 1])
            f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
            if center is None:
                # c_2.shape = torch.Size([#known classes, 1])
                c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)
                # dist.shape = torch.Size([N, #known classes])
                dist = (
                    f_2
                    - 2 * torch.matmul(features, torch.transpose(self.centers, 1, 0))
                    + torch.transpose(c_2, 1, 0)
                )
            else:
                c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
                dist = (
                    f_2
                    - 2 * torch.matmul(features, torch.transpose(center, 1, 0))
                    + torch.transpose(c_2, 1, 0)
                )
            dist = dist / float(features.shape[1])
        else:
            if center is None:
                center = self.centers
            else:
                center = center
            dist = features.matmul(center.t())
        # dist.shape = torch.Size([N, #known classes, 1])
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        # dist.shape = torch.Size([16, 6])
        dist = torch.mean(dist, dim=2)

        return dist


class ARPLoss(nn.CrossEntropyLoss):
    def __init__(self, args, feat_dim):
        super(ARPLoss, self).__init__()
        # self.gpu_id: str = args.gpu_id
        self.device = f'cuda:{args.gpu_id}'
        # self.weight_pl = float(args.weight_pl)
        self.temp = args.temp
        self.Dist = Dist(num_classes=args.num_classes, feat_dim=feat_dim)
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(self, z, labels=None):
        dist_dot_p = self.Dist(z, center=self.points, metric='dot')
        dist_l2_p = self.Dist(z, center=self.points)
        logits = dist_l2_p - dist_dot_p

        if labels is None:
            return logits
        cls_loss = F.cross_entropy(logits / self.temp, labels)

        center_batch = self.points[labels, :]
        _dis_known = (z - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).to(self.device)
        margin_loss = self.margin_loss(self.radius, _dis_known, target)

        # loss = cls_loss + self.weight_pl * loss_r

        return logits, cls_loss, margin_loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss
