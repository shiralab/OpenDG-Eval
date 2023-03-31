# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL
import torch.nn.functional as F
import re

import pickle


def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


def DistillKL(y_s, y_t, T):
    """KL divergence for distillation"""
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = y_t
    loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]
    return loss


def get_ratio_mixup_Dirichlet(domains, mixup_dir_list):
    RG = np.random.default_rng()
    return torch.from_numpy(
        RG.dirichlet(mixup_dir_list, size=domains.size(0))
    ).float()  # N * 3


def create_one_hot(y, classes, device):
    """quoted by DAML"""
    y_onehot = torch.LongTensor(y.size(0), classes).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def get_k_u_classes_set(args, eval_name_dict, train_loaders, eval_loaders):
    # don't support in the case of multi target domains
    idx: list = eval_name_dict['target']
    if args.loader_name == 'DeepDG_loader':
        unique_src_labels_list = [
            np.unique(loader.dataset.labels) for loader in train_loaders
        ]
        unique_all_target_labels = np.unique(eval_loaders[idx[0]].dataset.labels)
    elif args.loader_name == 'daml_loader':
        unique_src_labels_list = [
            loader.dataset.filter_class for loader in train_loaders
        ]
        unique_all_target_labels = eval_loaders[idx[0]].dataset.filter_class

    known_classes_set = set(np.concatenate(unique_src_labels_list, axis=None))

    unknown_classes_set = (
        set(np.concatenate(unique_all_target_labels, axis=None)) - known_classes_set
    )

    return known_classes_set, unknown_classes_set


def get_major_middle_minor(args, eval_name_dict, train_loaders, eval_loaders):
    # don't support in the case of multi target domains
    idx: list = eval_name_dict['target']
    if args.loader_name == 'DeepDG_loader':
        unique_src_labels_set_list = [
            set(np.unique(loader.dataset.labels)) for loader in train_loaders
        ]
        T = set(np.unique(eval_loaders[idx[0]].dataset.labels))

    elif args.loader_name == 'daml_loader':
        unique_src_labels_set_list = [
            set(loader.dataset.filter_class) for loader in train_loaders
        ]
        T = set(eval_loaders[idx[0]].dataset.filter_class)
    A, B, C = unique_src_labels_set_list
    major = A & B & C
    middle = (A & B | B & C | C & A) - major
    minor = (A | B | C) - (A & B | B & C | C & A)
    major_in_T = major & T
    middle_in_T = middle & T
    minor_in_T = minor & T
    return major_in_T, middle_in_T, minor_in_T


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    # save_dict = {"args": vars(args), "model_dict": alg.cpu().state_dict()} # original code
    save_dict = {"args": vars(args), "model_dict": alg.state_dict()}
    torch.save(save_dict, os.path.join(args.output, filename))


def get_best_acc_model_and_args_epoch_max_pkl(path, algorithm_name):
    # Get paths to all files containing best_acc_model_and_args_epochx.pkl
    files = os.listdir(path)

    # Extract epoch number from file name
    numbers = []
    for file in files:
        # Make sure the file name starts with ...
        if file.startswith('best_acc_model_and_args_epoch'):
            # Extract the numbers following ...
            number = re.search(r'best_acc_model_and_args_epoch(\d+)\.pkl', file).group(
                1
            )
            numbers.append(int(number))

    max_number = max(numbers)

    # Get the file with the largest number
    for file in files:
        if file.startswith('best_acc_model_and_args_epoch') and file.endswith(
            f'{max_number}.pkl'
        ):
            max_file = file
            break
    return max_file


def load_checkpoint(filename, alg, args):
    loaded_dict = torch.load(os.path.join(args.output, filename))
    args, model_dict = loaded_dict['args'], loaded_dict['model_dict']
    return args, model_dict


def train_valid_target_eval_names(args):
    eval_name_dict = {"train": [], "valid": [], "target": [], "source_unknown": []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict["train"].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict["valid"].append(t)
        else:
            eval_name_dict["target"].append(t)
        t += 1

    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {
        "ANDMask": ["total"],
        "CORAL": ["class", "coral", "total"],
        "DANN": ["class", "dis", "total"],
        "ERM": ["class"],
        "Mixup": ["class"],
        "MLDG": ["total"],
        "MMD": ["class", "mmd", "total"],
        "GroupDRO": ["group"],
        "RSC": ["class"],
        "VREx": ["loss", "nll", "penalty"],
        "DIFEX": ["class", "dist", "exp", "align", "total"],
        "ARPL": ["class", "margin", "total"],
        "OpenMax": ["class"],
        "DAEL": ["class", "distill", "total"],
        "CORAL_Dir_mixup": ["class", "coral", "dirmixup", "total"],
        "Ensemble_CORAL": [
            "class",
            "coral",
            "total",
        ],
        "Ensemble_CORAL_with_Dir_mixup": ["class", "coral", "mixup", "total"],
        "Ensemble_MMD_with_Dir_mixup": ["class", "mmd", "mixup", "total"],
        "Ensemble_CORAL_with_Distill": ["class", "coral", "distill", "total"],
        "Ensemble_MMD_with_Distill": ["class", "mmd", "distill", "total"],
        "Ensemble_MMD": [
            "class",
            "mmd",
            "total",
        ],
        "DAML": [
            "meta_train",
            "meta_val",
            "total",
        ],
        "DAML_wo_Dir_mixup": [
            "meta_train",
            "meta_val",
            "total",
        ],
        "DAML_wo_distill": [
            "meta_train",
            "meta_val",
            "total",
        ],
        "DAML_wo_Dmix_and_dst": [
            "meta_train",
            "meta_val",
            "total",
        ],
        "DAML_wo_metatest": [
            "meta_train",
            "meta_val",
            "total",
        ],
        "Single_CORAL_with_Dir_mixup": ["class", "coral", "mixup", "total"],
        "Double_Single_CORAL_with_Dir_mixup": ["class", "coral", "mixup", "total"],
        "Single_DAML": [
            "meta_train",
            "meta_val",
            "total",
        ],
    }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == "office":
        domains = ["amazon", "dslr", "webcam"]
    elif dataset == "office-caltech":
        domains = ["amazon", "dslr", "webcam", "caltech"]
    elif dataset == "office-home" or dataset == "mini-office-home":
        domains = ["Art", "Clipart", "Product", "Real_World"]
    elif dataset == "dg5":
        domains = ["mnist", "mnist_m", "svhn", "syn", "usps"]
    elif dataset == "PACS":
        domains = ["art_painting", "cartoon", "photo", "sketch"]
    elif dataset == "VLCS":
        domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
    elif dataset == "MultiDataSet":  # or dataset == "mini-MultiDataSet":
        domains = ["A", "hoge", "S", "V"]  # C is target domain
    else:
        print("No such dataset exists!")
    args.domains = domains
    args.img_dataset = {
        "office": ["amazon", "dslr", "webcam"],
        "office-caltech": ["amazon", "dslr", "webcam", "caltech"],
        "office-home": ["Art", "Clipart", "Product", "Real_World"],
        "mini-office-home": ["Art", "Clipart", "Product", "Real_World"],
        "PACS": ["art_painting", "cartoon", "photo", "sketch"],
        "dg5": ["mnist", "mnist_m", "svhn", "syn", "usps"],
        "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
        "MultiDataSet": ["A", "hoge", "S", "V"],
        # "mini-MultiDataSet": ["A", "hoge", "S", "V"],
    }
    if dataset == "dg5":
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == "office-home":
            args.num_classes = 65
        elif args.dataset == "office":
            args.num_classes = 31
        elif (
            args.dataset == "PACS"
            or args.dataset == "mini-office-home"
            # or args.dataset == 'mini-MultiDataSet'
        ):
            args.num_classes = 7
        elif args.dataset == "VLCS":
            args.num_classes = 5
    return args
