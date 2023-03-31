# coding=utf-8
import torch
from network import img_network
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


def get_fea(args):
    if args.dataset == "dg5":
        net = img_network.DTNBase()
    elif args.net.startswith("res"):
        net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net


def accuracy(
    network,
    loader,
    item,
    known_classes_set,
    unknown_classes_set,
    gpu_id,
):
    # DAML
    for weight in network.parameters():
        weight.fast = None
    correct = 0
    total = 0
    device = f'cuda:{gpu_id}'
    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()  # torch.Size([64, 3, 224, 224])
            y = data[1].to(device).long()  # torch.Size([64])
            p = network.predict(x)  # p means logits. torch.Size([64, known_classes])

            if p.size(1) == 1:  # binary classification
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()

            if item == 'target' or item == 'source_unknown':
                # Number of known classes in y
                known_samples = sum(
                    [True if (c.item() in known_classes_set) else False for c in y]
                )
                total += known_samples

            else:
                total += len(x)
    network.train()
    return correct / total


def accuracy_per_class(
    dataset_name,
    network,
    loader,
    item,
    known_classes_set,
    unknown_classes_set,
    gpu_id,
    major_class_set_in_t,
    middle_class_set_in_t,
    minor_class_set_in_t,
):
    # DAML
    for weight in network.parameters():
        weight.fast = None
    device = f'cuda:{gpu_id}'
    network.eval()
    major_correct = 0
    middle_correct = 0
    minor_correct = 0
    major_total = 0
    middle_total = 0
    minor_total = 0
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()  # torch.Size([3*bs, 3, 224, 224])
            y = data[1].to(device).long()  # torch.Size([3*s])
            p = network.predict(x)  # p means logits. torch.Size([3*bs, known_classes])
            correct_flag = p.argmax(1).eq(y)
            if item == 'target':
                kind_of_number = []
                for c in y:
                    if c.item() in major_class_set_in_t:
                        kind_of_number.append('major')
                    elif c.item() in middle_class_set_in_t:
                        kind_of_number.append('middle')
                    elif c.item() in minor_class_set_in_t:
                        kind_of_number.append('minor')
                    else:  # c is unknown class
                        pass

                for i in range(len(kind_of_number)):
                    # correct_flag[i] == True: Predicted value of the i-th data is correct
                    if correct_flag[i] == True:
                        if kind_of_number[i] == 'major':
                            major_correct += 1
                        elif kind_of_number[i] == 'middle':
                            middle_correct += 1
                        elif kind_of_number[i] == 'minor':
                            minor_correct += 1

                major_total += sum(
                    [True if (c.item() in major_class_set_in_t) else False for c in y]
                )
                middle_total += sum(
                    [True if (c.item() in middle_class_set_in_t) else False for c in y]
                )
                minor_total += sum(
                    [True if (c.item() in minor_class_set_in_t) else False for c in y]
                )
            else:
                exit
    network.train()

    # AD-HOC because PACS datset major class is None
    if (
        dataset_name == 'PACS'
        or dataset_name == 'mini-office-home'
        or dataset_name == 'mini-MultiDataSet'
        or dataset_name == 'MultiDataSet'
    ):
        major_correct_rate = -1
    else:
        major_correct_rate = major_correct / major_total
    middle_correct_rate = middle_correct / middle_total
    minor_correct_rate = minor_correct / minor_total
    return major_correct_rate, middle_correct_rate, minor_correct_rate


def target_auroc_per_class(
    dataset_name,
    network,
    loader,
    item,
    known_classes_set,
    unknown_classes_set,
    gpu_id,
    major_class_set_in_t,
    middle_class_set_in_t,
    minor_class_set_in_t,
):
    # DAML
    for weight in network.parameters():
        weight.fast = None
    device = f'cuda:{gpu_id}'
    network.eval()
    max_logits = []
    max_logits_major = []
    max_logits_middle = []
    max_logits_minor = []

    y_binaries = []
    y_binaries_major = []
    y_binaries_middle = []
    y_binaries_minor = []
    with torch.no_grad():
        for data in loader:
            sample_idx = 0
            x = data[0].to(device).float()  # torch.Size([64, 3, 224, 224])
            y = data[1].to(device).long()  # torch.Size([64])
            major_and_unknown_idx = []
            middle_and_unknown_idx = []
            minor_and_unknown_idx = []
            for c in y:
                if c.item() in major_class_set_in_t:
                    y_binaries_major.append(True)
                    major_and_unknown_idx.append(sample_idx)
                elif c.item() in middle_class_set_in_t:
                    y_binaries_middle.append(True)
                    middle_and_unknown_idx.append(sample_idx)
                elif c.item() in minor_class_set_in_t:
                    y_binaries_minor.append(True)
                    minor_and_unknown_idx.append(sample_idx)
                elif c.item() in unknown_classes_set:
                    y_binaries_major.append(False)
                    y_binaries_middle.append(False)
                    y_binaries_minor.append(False)
                    major_and_unknown_idx.append(sample_idx)
                    middle_and_unknown_idx.append(sample_idx)
                    minor_and_unknown_idx.append(sample_idx)

            # y_binary = [True if (c.item() in known_classes_set) else False for c in y]
            # y_binaries.extend(y_binary)
            p = network.predict(x)  # p means logits. torch.Size([64, 65])

            major_class_list = list(major_class_set_in_t)
            middle_class_list = list(middle_class_set_in_t)
            minor_class_list = list(minor_class_set_in_t)
            if (
                dataset_name == "PACS"
                or dataset_name == "mini-office-home"
                or dataset_name == "mini-MultiDataSet"
            ):
                max_logit_major = torch.tensor(
                    [-1]
                )  # because of none major class in PACS
            else:

                max_logit_major: torch.tensor = (
                    p[major_and_unknown_idx, :][:, major_class_list].max(dim=1).values
                )
            max_logits_major.extend(max_logit_major.tolist())

            # breakpoint()
            max_logit_middle: torch.tensor = (
                p[middle_and_unknown_idx, :][:, middle_class_list].max(dim=1).values
            )
            max_logits_middle.extend(max_logit_middle.tolist())
            max_logit_minor: torch.tensor = (
                p[minor_and_unknown_idx, :][:, minor_class_list].max(dim=1).values
            )
            max_logits_minor.extend(max_logit_minor.tolist())

            # max_logit: torch.tensor = p.max(dim=1).values
            # max_logits.extend(max_logit.tolist())
    if (
        dataset_name == "PACS"
        or dataset_name == "mini-office-home"
        or dataset_name == 'mini-MultiDataSet'
    ):
        auroc_major = 0
    else:
        auroc_major = roc_auc_score(y_binaries_major, max_logits_major)
    auroc_middle = roc_auc_score(y_binaries_middle, max_logits_middle)
    auroc_minor = roc_auc_score(y_binaries_minor, max_logits_minor)
    # auroc = roc_auc_score(y_binaries, max_logits)
    network.train()
    return auroc_major, auroc_middle, auroc_minor


def auroc(
    network,
    loader,
    item,
    known_classes_set,
    unknown_classes_set,
    gpu_id,
):

    # DAML
    for weight in network.parameters():
        weight.fast = None
    device = f'cuda:{gpu_id}'
    network.eval()
    max_logits = []
    y_binaries = []
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()  # torch.Size([64, 3, 224, 224])
            y = data[1].to(device).long()  # torch.Size([64])
            # x = data[0].cuda().float()  # torch.Size([64, 3, 224, 224])
            # y = data[1].cuda().long()  # torch.Size([64])
            y_binary = [True if (c.item() in known_classes_set) else False for c in y]
            y_binaries.extend(y_binary)
            p = network.predict(x)  # p means logits. torch.Size([64, 65])
            max_logit: torch.tensor = p.max(dim=1).values
            max_logits.extend(max_logit.tolist())

    auroc = roc_auc_score(y_binaries, max_logits)
    network.train()
    return auroc


def source_unknown_auroc(
    network,
    loaders,
    known_classes_set,
    unknown_classes_set,
    gpu_id,
):
    # DAML
    for weight in network.parameters():
        weight.fast = None
    device = f'cuda:{gpu_id}'
    network.eval()
    max_logits = []
    y_binaries = []
    auroc_list = []
    with torch.no_grad():
        for loader in loaders:
            for data in loader:
                x = data[0].to(device).float()  # torch.Size([64, 3, 224, 224])
                y = data[1].to(device).long()  # torch.Size([64])
                # x = data[0].cuda().float()  # torch.Size([64, 3, 224, 224])
                # y = data[1].cuda().long()  # torch.Size([64])
                y_binary = [
                    True if (c.item() in known_classes_set) else False for c in y
                ]
                y_binaries.extend(y_binary)
                p = network.predict(x)  # p means logits. torch.Size([64, 65])
                max_logit: torch.tensor = p.max(dim=1).values
                max_logits.extend(max_logit.tolist())

        auroc = roc_auc_score(y_binaries, max_logits)
        auroc_list.append(auroc)
    network.train()
    return sum(auroc_list) / len(auroc_list)


def h_score(
    network,
    loader,
    item,
    known_classes_set,
    unknown_classes_set,
    gpu_id,
    threshold,
):
    # DAML
    for weight in network.parameters():
        weight.fast = None
    correct = 0
    unknown_correct = 0
    known_correct = 0
    total = 0
    k_total = 0
    u_total = 0
    num_classes = len(known_classes_set)
    device = f'cuda:{gpu_id}'
    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()  # torch.Size([64, 3, 224, 224])
            y = data[1].to(device).long()  # torch.Size([64])
            # change all unknown class idx to num_classes idx
            unknown_flag = (y > (num_classes - 1)).long()
            y = y * (1 - unknown_flag) + num_classes * unknown_flag

            p = network.predict(x)  # p means logits. torch.Size([64, known_classes])
            sft_score = F.softmax(p, dim=1)
            # if unknown_flag.sum() > 0 and threshold > 0.7:
            #    breakpoint()

            # change p.shape (N,C) -> (N, C+1)
            is_unknown = (sft_score.max(1).values < threshold.to(device)).float()
            sft_score = torch.cat((sft_score, is_unknown.unsqueeze(1)), dim=1)

            predicted_label = sft_score.argmax(1)
            matching_unknown = (predicted_label == num_classes) & (y == num_classes)
            matching_known = ((predicted_label != num_classes) & (y != num_classes)) & (
                predicted_label == y
            )
            unknown_correct += matching_unknown.sum().item()
            known_correct += matching_known.sum().item()
            # correct += (p.argmax(1).eq(y).float()).sum().item()

            if item == 'target' or item == 'source_unknown':
                # Number of known classes in y
                known_samples = sum(
                    [True if (c.item() in known_classes_set) else False for c in y]
                )
                unknown_samples = len(y) - known_samples

                k_total += known_samples
                u_total += unknown_samples
            else:
                total += len(x)

    k_acc = known_correct / k_total
    u_acc = unknown_correct / u_total
    h_score = 2.0 * k_acc * u_acc / (k_acc + u_acc)

    network.train()

    return h_score


def get_thresholds(network, loader, gpu_id):
    device = f'cuda:{gpu_id}'
    network.eval()
    sft_scores = []
    with torch.no_grad():
        for data in loader:
            x = data[0].to(device).float()  # torch.Size([64, 3, 224, 224])
            p = network.predict(x)  # p means logits. torch.Size([64, known_classes])
            p = F.softmax(p, dim=1)
            sft_scores.append(p.flatten())

    thresholds = torch.cat(sft_scores)
    thd_min = torch.min(thresholds)
    thd_max = torch.max(thresholds)
    threshold_range_list = [thd_min + (thd_max - thd_min) * i / 9 for i in range(10)]
    network.train()

    return threshold_range_list
