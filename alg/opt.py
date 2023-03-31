# coding=utf-8
import torch


def get_params(alg, args, inner=False, alias=True, isteacher=False):
    if args.schuse:
        if args.schusech == "cos":
            initlr = args.lr
        else:
            initlr = 1.0
    else:
        if inner:
            initlr = args.inner_lr
        else:
            initlr = args.lr
    if isteacher:
        params = [
            {"params": alg[0].parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg[1].parameters(), "lr": args.lr_decay2 * initlr},
            {"params": alg[2].parameters(), "lr": args.lr_decay2 * initlr},
        ]
        return params
    if inner:
        params = [
            {"params": alg[0].parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg[1].parameters(), "lr": args.lr_decay2 * initlr},
        ]
    elif "ARPL" in args.algorithm:
        params = [
            {"params": alg.featurizer.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.criterion.parameters(), "lr": args.lr_decay2 * initlr},
        ]
    elif (
        "DAEL" == args.algorithm
        or 'Single_CORAL_with_Dir_mixup' == args.algorithm
        or 'Single_DAML' == args.algorithm
    ):
        params = [
            {"params": alg.featurizer.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.classifier_1.parameters(), "lr": args.lr_decay2 * initlr},
            {"params": alg.classifier_2.parameters(), "lr": args.lr_decay2 * initlr},
            {"params": alg.classifier_3.parameters(), "lr": args.lr_decay2 * initlr},
        ]
    elif (
        "Ensemble_CORAL" == args.algorithm
        or "Ensemble_CORAL_with_Dir_mixup" == args.algorithm
        or "Ensemble_CORAL_with_Distill" == args.algorithm
        or "Ensemble_MMD" == args.algorithm
        or "Ensemble_MMD_with_Dir_mixup" == args.algorithm
        or "Ensemble_MMD_with_Distill" == args.algorithm
        or "DAML" == args.algorithm
        or "DAML_wo_Dir_mixup" == args.algorithm
        or "DAML_wo_distill" == args.algorithm
        or "DAML_wo_Dmix_and_dst" == args.algorithm
        or "DAML_wo_metatest" == args.algorithm
    ):
        params = [
            {"params": alg.featurizer_1.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.featurizer_2.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.featurizer_3.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.classifier_1.parameters(), "lr": args.lr_decay2 * initlr},
            {"params": alg.classifier_2.parameters(), "lr": args.lr_decay2 * initlr},
            {"params": alg.classifier_3.parameters(), "lr": args.lr_decay2 * initlr},
        ]

    elif alias:  # alias=True(defalut)
        params = [
            {"params": alg.featurizer.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.classifier.parameters(), "lr": args.lr_decay2 * initlr},
        ]
    else:
        params = [
            {"params": alg[0].parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg[1].parameters(), "lr": args.lr_decay2 * initlr},
        ]

    if ("DANN" in args.algorithm) or ("CDANN" in args.algorithm):
        params.append(
            {"params": alg.discriminator.parameters(), "lr": args.lr_decay2 * initlr}
        )
    if "CDANN" in args.algorithm:
        params.append(
            {"params": alg.class_embeddings.parameters(), "lr": args.lr_decay2 * initlr}
        )

    return params


def get_optimizer(alg, args, inner=False, alias=True, isteacher=False):
    params = get_params(alg, args, inner, alias, isteacher)
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    return optimizer


def get_scheduler(optimizer, args):
    if not args.schuse:
        return None
    if args.schusech == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay),
        )
    return scheduler
