# coding=utf-8
import sys

from torch.utils.data import Dataset
import numpy as np

# from DeepDG.datautil.util import Nmax
# from DeepDG.datautil.imgdata.util import rgb_loader, l_loader

from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class ImageDataset(object):
    def __init__(
        self,
        dataset: str,
        task: str,
        root_dir: str,
        domain_name: int,
        domain_label: int,  # domain_label=-1,
        needed_classes: list,  # list(list)
        labels=None,
        transform=None,
        target_transform=None,
        indices=None,
        test_envs=[],
        mode="Default",
        is_needed_only_one_class=False,
        one_class_id_list=None,
    ):

        self.imgs: list(tuple(str, int)) = ImageFolder(root_dir + domain_name).imgs
        self.domain_num = 0
        self.task = task
        self.dataset = dataset

        # imgs: list(str) = [item[0] for item in self.imgs]
        # labels: list(int) = [item[1] for item in self.imgs]
        if is_needed_only_one_class:
            domain_specific_classes = one_class_id_list
        else:
            domain_specific_classes = needed_classes[domain_label]
        imgs: list(str) = [
            item[0] for item in self.imgs if (item[1] in domain_specific_classes)
        ]
        labels: list(int) = [
            item[1] for item in self.imgs if (item[1] in domain_specific_classes)
        ]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == "Default":
            self.loader = default_loader
        elif mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * (
            domain_label - Nmax(test_envs, domain_label)
        )

    def set_labels(self, tlabels=None, label_type="domain_label"):
        assert len(tlabels) == len(self.x)
        if label_type == "domain_label":
            self.dlabels = tlabels
        elif label_type == "class_label":
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def __len__(self):
        return len(self.indices)
