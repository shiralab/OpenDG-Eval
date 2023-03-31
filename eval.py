# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.util import (
    set_random_seed,
    save_checkpoint,
    get_best_acc_model_and_args_epoch_max_pkl,
    load_checkpoint,
    print_args,
    train_valid_target_eval_names,
    alg_loss_dict,
    Tee,
    img_param_init,
    print_environ,
    get_k_u_classes_set,
    get_major_middle_minor,
)
from datautil.getdataloader import (
    get_img_dataloader,
    get_img_daml_dataloader,
    get_img_source_unknown_dataloader,
    get_img_daml_source_unknown_dataloader,
    get_img_daml_multi_dataloader,
    # get_img_daml_multi_source_unknown_dataloader,
)


def get_args():
    parser = argparse.ArgumentParser(description="DG")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--alpha", type=float, default=1, help="DANN dis alpha")
    parser.add_argument(
        "--anneal_iters",
        type=int,
        default=500,
        help="Penalty anneal iters used in VREx",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--beta", type=float, default=1, help="DIFEX beta")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam hyper-param")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument(
        "--checkpoint_freq", type=int, default=1, help="Checkpoint every N epoch"
    )
    parser.add_argument(
        "--classifier", type=str, default="linear", choices=["linear", "wn"]
    )
    parser.add_argument("--data_file", type=str, default="", help="root_dir")
    parser.add_argument("--dataset", type=str, default="office")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument(
        "--dis_hidden", type=int, default=256, help="dis hidden dimension"
    )
    parser.add_argument(
        "--disttype",
        type=str,
        default="2-norm",
        choices=["1-norm", "2-norm", "cos", "norm-2-norm", "norm-1-norm"],
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--groupdro_eta", type=float, default=1, help="groupdro eta")
    parser.add_argument(
        "--inner_lr", type=float, default=1e-2, help="learning rate used in MLDG"
    )
    parser.add_argument(
        "--lam", type=float, default=1, help="tradeoff hyperparameter used in VREx"
    )
    parser.add_argument("--layer", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="for sgd")
    parser.add_argument(
        "--lr_decay1", type=float, default=1.0, help="for pretrained featurizer"
    )
    parser.add_argument(
        "--lr_decay2",
        type=float,
        default=1.0,
        help="inital learning rate decay of network",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.0003, help="for optimizer")
    parser.add_argument("--max_epoch", type=int, default=120, help="max iterations")
    parser.add_argument(
        "--mixupalpha", type=float, default=0.2, help="mixup hyper-param"
    )
    parser.add_argument("--mldg_beta", type=float, default=1, help="mldg hyper-param")
    parser.add_argument(
        "--mmd_gamma", type=float, default=1, help="MMD, CORAL hyper-param"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="for optimizer")
    parser.add_argument(
        "--net",
        type=str,
        default="resnet50",
        help="featurizer: vgg16, resnet50, resnet101,DTNBase",
    )
    parser.add_argument("--N_WORKERS", type=int, default=4)
    parser.add_argument(
        "--rsc_f_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument(
        "--rsc_b_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument("--schuse", action="store_true")
    parser.add_argument("--schusech", type=str, default="cos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split_style",
        type=str,
        default="strat",
        help="the style to split the train and eval datasets",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="img_dg",
        choices=["img_dg"],
        help="now only support image tasks",
    )
    parser.add_argument("--tau", type=float, default=1, help="andmask tau")
    # ARPL hyper-parameters
    parser.add_argument("--weight_pl", type=float, default=0.1, help="arpl weight_pl")
    parser.add_argument("--temp", type=float, default=1, help="arpl temp")

    # OPenMax hyper-parameters
    # Parameters for weibull distribution fitting.
    parser.add_argument(
        '--weibull_tail', default=20, type=int, help='Classes used in testing'
    )
    parser.add_argument(
        '--weibull_alpha', default=3, type=int, help='Classes used in testing'
    )
    parser.add_argument(
        '--weibull_threshold', default=0.9, type=float, help='Classes used in testing'
    )

    # DAML hyper parameter
    parser.add_argument('--T', default=2.0, type=float, help='softmax temperature')
    parser.add_argument("--trade", type=float, default=3.0)
    parser.add_argument("--trade2", type=float, default=1.0)
    parser.add_argument("--trade3", type=float, default=1.0)
    parser.add_argument("--trade4", type=float, default=3.0)
    parser.add_argument("--mixup_dir", type=float, default=0.6)
    parser.add_argument("--mixup_dir2", type=float, default=0.2)
    parser.add_argument(
        "--stop_gradient",
        type=int,
        default=1,
        help='whether stop gradient of the first order gradient',
    )
    parser.add_argument("--meta_step_size", type=float, default=0.01)

    parser.add_argument(
        "--test_envs", type=int, nargs="+", default=[0], help="target domains"
    )
    parser.add_argument(
        "--output", type=str, default="train_output", help="result output path"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--is_different_class_space", type=int, default=1)
    parser.add_argument("--is_data_aug", type=str, default='no')
    parser.add_argument("--loader_name", type=str, default="DeepDG")
    parser.add_argument("--d_path", type=str, default="nothing")
    parser.add_argument("--t_domain", type=str, default="C")
    args = parser.parse_args()

    args.data_dir = args.data_file + args.data_dir
    os.environ["CUDA_VISIBLE_DEVICS"] = args.gpu_id
    # breakpoint()
    if args.dataset != 'MultiDataSet':
        os.makedirs(args.output + "/eval", exist_ok=True)
        sys.stdout = Tee(os.path.join(args.output + "/eval", "out.txt"))
        sys.stderr = Tee(os.path.join(args.output + "/eval", "err.txt"))
    else:
        os.makedirs(args.output + f"/eval_{args.t_domain}", exist_ok=True)
        # sys.stdout = Tee(
        #    os.path.join(args.output + f"/eval_{args.t_domain}", "out.txt")
        # )
        sys.stderr = Tee(
            os.path.join(args.output + f"/eval_{args.t_domain}", "err.txt")
        )

    args = img_param_init(args)
    print_environ()
    return args


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)

    if args.loader_name == 'DeepDG_loader':
        train_loaders, eval_loaders = get_img_dataloader(args)
        source_unknown_loaders = get_img_source_unknown_dataloader(args)

    elif args.loader_name == 'daml_loader':
        if args.d_path == 'nothing':
            train_loaders, eval_loaders = get_img_daml_dataloader(args)
            source_unknown_loaders = get_img_daml_source_unknown_dataloader(args)
        elif args.dataset == 'MultiDataSet':
            train_loaders, eval_loaders = get_img_daml_multi_dataloader(args)
            source_unknown_loaders = (
                eval_loaders  # get_img_daml_multi_source_unknown_dataloader(args)
            )

    # eval_name_dict = {"train": [], "valid": [], "target": []}
    if args.dataset == 'MultiDataSet':
        eval_name_dict = {"train": [0, 1, 2], "valid": [3, 4, 5], "target": [6]}
    else:
        eval_name_dict = train_valid_target_eval_names(args)
    known_classes_set, unknown_classes_set = get_k_u_classes_set(
        args, eval_name_dict, train_loaders, eval_loaders
    )

    (
        major_class_set_in_t,
        middle_class_set_in_t,
        minor_class_set_in_t,
    ) = get_major_middle_minor(
        args,
        eval_name_dict,
        train_loaders,
        eval_loaders,
    )
    # AD-HOC because 'args.num_classes' has previously been defined inside the img_params_init function
    args.num_classes = len(known_classes_set)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).to(f'cuda:{args.gpu_id}')

    max_file = get_best_acc_model_and_args_epoch_max_pkl(args.output, args.algorithm)
    previous_args, model_dict = load_checkpoint(max_file, algorithm, args)
    algorithm.load_state_dict(model_dict)
    # algorithm.train()
    # opt = get_optimizer(algorithm, args)
    # sch = get_scheduler(opt, args)

    s = print_args(args, [])
    print("=======hyper-parameter used========")
    print(s)

    acc_record = {}
    acc_type_list = ["valid", "target"]
    # train_minibatches_iterator = zip(*train_loaders)
    # early_stopping_cnt = 0
    print("===========start evaluating===========")
    sss = time.time()

    # s = ""
    # print(s[:-1])
    s = ""

    for item in acc_type_list:
        acc_record[item] = np.mean(
            np.array(
                [
                    modelopera.accuracy(
                        algorithm,
                        eval_loaders[i],
                        item,
                        known_classes_set,
                        unknown_classes_set,
                        args.gpu_id,
                    )
                    for i in eval_name_dict[item]
                ]
            )
        )
        s += item + '_acc:%.4f,' % acc_record[item]

    acc_record['source_unknown'] = np.mean(
        np.array(
            [
                modelopera.accuracy(
                    algorithm,
                    source_unknown_loaders[i],
                    'source_unknown',
                    known_classes_set,
                    unknown_classes_set,
                    args.gpu_id,
                )
                for i in range(3)
            ]
        )
    )
    s += 'source_unknown' + '_acc:%.4f,' % acc_record['source_unknown']

    (
        major_correct_rate,
        middle_correct_rate,
        minor_correct_rate,
    ) = modelopera.accuracy_per_class(
        args.dataset,
        algorithm,
        eval_loaders[eval_name_dict['target'][0]],
        'target',
        known_classes_set,
        unknown_classes_set,
        args.gpu_id,
        major_class_set_in_t,
        middle_class_set_in_t,
        minor_class_set_in_t,
    )
    s += 'target' + '_major_acc:%.4f,' % major_correct_rate
    s += 'target' + '_middle_acc:%.4f,' % middle_correct_rate
    s += 'target' + '_minor_acc:%.4f,' % minor_correct_rate

    target_auroc = modelopera.auroc(
        algorithm,
        eval_loaders[eval_name_dict['target'][0]],
        'target',
        known_classes_set,
        unknown_classes_set,
        args.gpu_id,
    )
    s += 'target' + "_auroc:%.4f," % target_auroc

    if args.dataset != 'MultiDataSet':
        source_unknown_auroc = modelopera.source_unknown_auroc(
            algorithm,
            source_unknown_loaders,
            known_classes_set,
            unknown_classes_set,
            args.gpu_id,
        )
        s += 'source_unknown' + "_auroc:%.4f," % source_unknown_auroc
    else:
        source_unknown_auroc = -100
        s += 'source_unknown' + "_auroc:%.4f," % source_unknown_auroc
    if args.dataset != 'MultiDataSet':
        major_auroc, middle_auroc, minor_auroc = modelopera.target_auroc_per_class(
            args.dataset,
            algorithm,
            eval_loaders[eval_name_dict['target'][0]],
            'target',
            known_classes_set,
            unknown_classes_set,
            args.gpu_id,
            major_class_set_in_t,
            middle_class_set_in_t,
            minor_class_set_in_t,
        )
        s += 'target' + "_major_auroc:%.4f," % major_auroc
        s += 'target' + "_middle_auroc:%.4f," % middle_auroc
        s += 'target' + "_minor_auroc:%.4f," % minor_auroc
    else:
        major_auroc = -100
        middle_auroc = -100
        minor_auroc = -100
        s += 'target' + "_major_auroc:%.4f," % major_auroc
        s += 'target' + "_middle_auroc:%.4f," % middle_auroc
        s += 'target' + "_minor_auroc:%.4f," % minor_auroc

    print(s[:-1])

    print("total cost time: %.4f" % (time.time() - sss))
    # algorithm_dict = algorithm.state_dict()
    # print("best_valid acc: %.4f" % best_valid_acc)
    # print("best_target_acc: %.4f" % target_acc)
    print("target_auroc: %.4f" % target_auroc)

    if args.dataset != 'MultiDataSet':
        tmp_PATH = args.output + "/eval"
    else:
        tmp_PATH = args.output + f"/eval_{args.t_domain}"

    with open(os.path.join(tmp_PATH, "eval.txt"), "w") as f:
        f.write("done\n")
        f.write("valid acc:%.4f\n" % (acc_record['valid']))
        f.write("target acc:%.4f\n" % (acc_record['target']))
        f.write("source unknown acc:%.4f\n" % (acc_record['source_unknown']))
        f.write("target major acc:%.4f\n" % (major_correct_rate))
        f.write("target middle acc:%.4f\n" % (middle_correct_rate))
        f.write("target minor acc:%.4f\n" % (minor_correct_rate))
        f.write("target auroc:%.4f\n" % (target_auroc))
        f.write("source unknown auroc:%.4f\n" % (source_unknown_auroc))
        f.write("target major auroc:%.4f\n" % (major_auroc))
        f.write("target middle auroc:%.4f\n" % (middle_auroc))
        f.write("target minor auroc:%.4f\n" % (minor_auroc))
