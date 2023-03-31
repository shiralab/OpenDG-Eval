# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.MMD import MMD
from alg.algs.CORAL import CORAL
from alg.algs.DANN import DANN
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup
from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.ANDMask import ANDMask
from alg.algs.VREx import VREx
from alg.algs.DIFEX import DIFEX
from alg.algs.ARPL import ARPL
from alg.algs.DAEL import DAEL
from alg.algs.CORAL_Dir_mixup import CORAL_Dir_mixup
from alg.algs.Ensemble_CORAL import Ensemble_CORAL
from alg.algs.Ensemble_CORAL_with_Dir_mixup import Ensemble_CORAL_with_Dir_mixup
from alg.algs.Ensemble_CORAL_with_Distill import Ensemble_CORAL_with_Distill
from alg.algs.Ensemble_MMD import Ensemble_MMD
from alg.algs.Ensemble_MMD_with_Dir_mixup import Ensemble_MMD_with_Dir_mixup
from alg.algs.Ensemble_MMD_with_Distill import Ensemble_MMD_with_Distill
from alg.algs.DAML import DAML
from alg.algs.DAML_wo_Dir_mixup import DAML_wo_Dir_mixup
from alg.algs.DAML_wo_distill import DAML_wo_distill
from alg.algs.DAML_wo_Dmix_and_dst import DAML_wo_Dmix_and_dst
from alg.algs.DAML_wo_metatest import DAML_wo_metatest
from alg.algs.Single_CORAL_with_Dir_mixup import Single_CORAL_with_Dir_mixup

from alg.algs.Double_Single_CORAL_with_Dir_mixup import (
    Double_Single_CORAL_with_Dir_mixup,
)
from alg.algs.Single_DAML import Single_DAML

# from alg.algs.OpenMax import OpenMax

ALGORITHMS = [
    "ERM",
    "Mixup",
    "CORAL",
    "MMD",
    "DANN",
    "MLDG",
    "GroupDRO",
    "RSC",
    "ANDMask",
    "VREx",
    "DIFEX",
    "ARPL",
    "OpenMax",
    "DAEL",
    "CORAL_Dir_mixup",
    "Ensemble_CORAL",
    "Ensemble_CORAL_with_Dir_mixup",
    "Ensemble_CORAL_with_Distill",
    "Ensemble_MMD",
    "Ensemble_MMD_with_Dir_mixup",
    "Ensemble_MMD_with_Distill",
    "DAML",
    "DAML_wo_Dir_mixup",
    "DAML_wo_distill",
    "DAML_wo_Dmix_and_dst",
    "DAML_wo_metatest",
    'Single_CORAL_with_Dir_mixup',
    'Double_Single_CORAL_with_Dir_mixup',
    'Single_DAML',
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
