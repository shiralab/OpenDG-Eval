import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import os


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)  # = target.shape[0]

        # _, pred = output.topk(maxk, 1, True, True)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        # pred.shape == torch.Size([batch_size, maxk])
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:

            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            # correct_k.shape == torch.Size([1]) regardless of k
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_raw_output_using_domain_classifier(
    val_loader, model, domain_classifier, num_classes
):
    model.eval()
    domain_classifier.eval()

    output_sum = []  # List[List]
    target_sum = []

    with torch.no_grad():
        for the_loader in val_loader:
            for i, (images, target, _) in enumerate(the_loader):
                images = images.cuda()
                target = target.cuda()
                # output_sum phase
                output, _ = model(images)
                assert len(output) == 3 and type(output) is list
                assert output[0].shape == torch.Size([images.shape[0], num_classes])
                output_sum.append(output)
                # target_sum phase
                is_outlier_flag = (target > (num_classes - 1)).float()
                target = target * (1 - is_outlier_flag) + num_classes * is_outlier_flag
                target = target.long()
                target_sum.append(target)
    output_sum = [
        torch.cat([output_sum[j][i] for j in range(len(output_sum))], dim=0)
        for i in range(3)
    ]
    target_sum = torch.cat(target_sum)

    assert len(output_sum) == 3
    assert output_sum[0].shape == torch.Size([len(the_loader.dataset), num_classes])
    assert len(target_sum) == len(the_loader.dataset)
    return output_sum, target_sum


# Calculate for each class included in all_target_classes
def coral_get_raw_output(val_loader, model, num_classes):
    model.eval()
    output_sum = []  # List[List]
    target_sum = []

    with torch.no_grad():
        for the_loader in val_loader:
            for i, (images, target, _) in enumerate(the_loader):
                images = images.cuda()
                target = target.cuda()
                # output_sum phase
                output, _ = model(images)
                output = [output]
                assert len(output) == 1 and type(output) is list
                assert output[0].shape == torch.Size([images.shape[0], num_classes])
                output_sum.append(output)
                # target_sum phase
                is_outlier_flag = (target > (num_classes - 1)).float()
                target = target * (1 - is_outlier_flag) + num_classes * is_outlier_flag
                target = target.long()
                target_sum.append(target)
    output_sum = [
        torch.cat([output_sum[j][i] for j in range(len(output_sum))], dim=0)
        for i in range(1)
    ]
    target_sum = torch.cat(target_sum)

    assert len(output_sum) == 1
    assert output_sum[0].shape == torch.Size([len(the_loader.dataset), num_classes])
    assert len(target_sum) == len(the_loader.dataset)
    return output_sum, target_sum


# Calculate for each class included in all_target_classes
def get_raw_output(val_loader, model, num_classes):
    model.eval()
    output_sum = []  # List[List]
    target_sum = []

    with torch.no_grad():
        for the_loader in val_loader:
            for i, (images, target, _) in enumerate(the_loader):
                images = images.cuda()
                target = target.cuda()
                # output_sum phase
                output, _ = model(images)
                assert len(output) == 3 and type(output) is list
                assert output[0].shape == torch.Size([images.shape[0], num_classes])
                output_sum.append(output)
                # target_sum phase
                is_outlier_flag = (target > (num_classes - 1)).float()
                target = target * (1 - is_outlier_flag) + num_classes * is_outlier_flag
                target = target.long()
                target_sum.append(target)
    output_sum = [
        torch.cat([output_sum[j][i] for j in range(len(output_sum))], dim=0)
        for i in range(3)
    ]
    target_sum = torch.cat(target_sum)

    assert len(output_sum) == 3
    assert output_sum[0].shape == torch.Size([len(the_loader.dataset), num_classes])
    assert len(target_sum) == len(the_loader.dataset)
    return output_sum, target_sum


def get_auroc(test_loader, model, known_classes, unknown_classes):
    model.eval()
    y_scores = []
    targets = []

    with torch.no_grad():
        for images, target, _ in test_loader:
            images = images.cuda()
            # HACK: Dealing with cases where known_classes is randomly determined
            # change known class to 1, unknown class to 0
            target_binary = (target < len(known_classes)).long()
            targets.extend(target_binary.tolist())
            """
            if target in known_classes:
                targets.append(1)
            elif target in unknown_classes:
                targets.append(0)
            """
            [output_0, output_1, output_2], _ = model(images, -1)
            output = (
                F.softmax(output_0, 1) + F.softmax(output_1, 1) + F.softmax(output_2, 1)
            )
            # output.shape == (batch_size, num_known_classes)
            output = output / 3
            y_score = output.max(dim=1).values

            y_scores.extend(y_score.tolist())

    assert len(y_scores) == len(test_loader.dataset)
    auroc = roc_auc_score(targets, y_scores)

    return auroc


def coral_get_auroc(test_loader, model, known_classes, unknown_classes):
    model.eval()
    y_scores = []
    targets = []

    with torch.no_grad():
        for images, target, _ in test_loader:
            images = images.cuda()
            # HACK: Dealing with cases where known_classes is randomly determined
            # change known class to 1, unknown class to 0
            target_binary = (target < len(known_classes)).long()
            targets.extend(target_binary.tolist())
            """
            if target in known_classes:
                targets.append(1)
            elif target in unknown_classes:
                targets.append(0)
            """
            output, _ = model(images)
            output = F.softmax(output, 1)
            # output.shape == (batch_size, num_known_classes)
            y_score = output.max(dim=1).values

            y_scores.extend(y_score.tolist())

    assert len(y_scores) == len(test_loader.dataset)
    auroc = roc_auc_score(targets, y_scores)

    return auroc


# Calculate for each class included in all_target_classes
def get_new_output(raw_output, T):
    assert len(raw_output) == 3 and type(raw_output) == list
    N = raw_output[0].shape[0]  # Num of data
    num_classes = raw_output[0].shape[1]
    output = [F.softmax(headout / T, dim=1) for headout in raw_output]
    assert len(output) == 3 and type(output) == list
    assert output[0].shape == raw_output[0].shape
    # assert output[0].sum(dim=1) == torch.ones(N)
    output_mean = torch.mean(torch.stack(output), 0)
    assert torch.stack(output).shape == torch.Size([3, N, num_classes])
    assert output_mean.shape == torch.Size([N, num_classes])
    # assert output_mean.sum(dim=1) == torch.ones(N)
    max_prob, max_index = torch.max(output_mean, 1)
    assert max_prob.shape == torch.Size([N])
    return output_mean, max_prob


# Calculate for each class included in all_target_classes
def coral_get_new_output(raw_output, T):
    assert len(raw_output) == 1 and type(raw_output) == list
    N = raw_output[0].shape[0]  # Num of data
    num_classes = raw_output[0].shape[1]
    output = [F.softmax(headout / T, dim=1) for headout in raw_output]
    assert len(output) == 1 and type(output) == list
    assert output[0].shape == raw_output[0].shape
    # assert output[0].sum(dim=1) == torch.ones(N)
    output_mean = torch.mean(torch.stack(output), 0)
    assert torch.stack(output).shape == torch.Size([1, N, num_classes])
    assert output_mean.shape == torch.Size([N, num_classes])
    # assert output_mean.sum(dim=1) == torch.ones(N)
    max_prob, max_index = torch.max(output_mean, 1)
    assert max_prob.shape == torch.Size([N])
    return output_mean, max_prob


def get_acc(tsm_output, outlier_indi, outlier_thred, target):
    # top1 = AverageMeter('Acc@1', ':6.2f')

    outlier_pred = (outlier_indi < outlier_thred).float()  # known(0), unkown(1)
    outlier_pred = outlier_pred.view(-1, 1)
    # outlier_pred.shape == torch.Size([N,1])
    # tsm_output.shape == torch.Size([N,num_classes])
    output = torch.cat((tsm_output, outlier_pred.cuda()), dim=1)
    # output.shape == torch.Size([N,num_classes+1])

    # measure accuracy and record loss
    acc1 = accuracy(output, target, topk=(1,))
    # top1.update(val=acc1[0], n=output.shape[0])
    return acc1


def get_pretrained_dir(source_classes, T1, cfg):
    daml_path = os.getcwd()  # Path of daml.py
    project_path = os.path.abspath(
        os.path.join(daml_path, os.pardir)
    )  # Parent directory
    """
    model_path = os.path.join(
        cfg.data,
        cfg.source,
        source_classes,
        'pretrain_' + cfg.is_pretrained,
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_name='best_val.tar'
    """
    if cfg.data == 'PACS':
        # file = f'{cfg.data}_{cfg.source}_{source_classes}_{T1}_pretrain_{cfg.is_pretrained}'
        file = f'{cfg.data}_{cfg.source}_{cfg.target}_pretrain_{cfg.is_pretrained}'
    elif cfg.data == 'OfficeHome':
        max_k = cfg.known_C - 1
        k_range = f'0-{max_k}'
        u_range = f'{cfg.known_C}-63'
        # file = (
        #    f'{cfg.data}_{cfg.source}_{k_range}_{u_range}_pretrain_{cfg.is_pretrained}'
        # )
        file = f'{cfg.data}_{cfg.source}_{cfg.target}_pretrain_{cfg.is_pretrained}'
    # common_name = os.path.join(project_path, 'runs', file)
    # val_save_dir = common_name + '_best_val.tar'
    common_name = os.path.join(project_path, 'runs')
    val_save_dir = os.path.join(common_name, f'val_{cfg.trial}')

    return val_save_dir, file


def get_save_dir(source_classes, T1, cfg):
    daml_path = os.getcwd()  # Path of daml.py
    project_path = os.path.abspath(
        os.path.join(daml_path, os.pardir)
    )  # Parent directory
    if cfg.data == 'PACS':
        # file = f'{cfg.data}_{cfg.source}_{source_classes}_{T1}_pretrain_{cfg.is_pretrained}'
        file = f'{cfg.data}_{cfg.source}_{cfg.target}_pretrain_{cfg.is_pretrained}'
    elif cfg.data == 'OfficeHome':
        max_k = cfg.known_C - 1
        k_range = f'0-{max_k}'
        u_range = f'{cfg.known_C}-63'
        # file = (
        #    f'{cfg.data}_{cfg.source}_{k_range}_{u_range}_pretrain_{cfg.is_pretrained}'
        # )
        file = f'{cfg.data}_{cfg.source}_{cfg.target}_pretrain_{cfg.is_pretrained}'

    # common_name = os.path.join(project_path, 'runs', file)
    common_name = os.path.join(project_path, 'runs')
    val_save_dir = os.path.join(common_name, f'val_{cfg.trial}')
    test_save_dir = os.path.join(common_name, f'test_{cfg.trial}')
    final_save_dir = os.path.join(common_name, f'final_{cfg.trial}')
    # val_save_dir = common_name + '_best_val.tar'
    # test_save_dir = common_name + '_best_test.tar'
    # final_save_dir = common_name + '_final.tar'
    return val_save_dir, test_save_dir, final_save_dir, file
