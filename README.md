# OpenDG-Eval

## Paper
This is the implement for the following paper.
```
"Simple Domain Generalization Methods are Strong Baselines for Open Domain Generalization", 2023
```

## Directory
```
.
├── alg
├── dalib
├── datautil
├── eval.py
├── eval_hscore.py
├── network
├── scripts
├── train.py
└── utils
```
- __alg__
  - Implements for each algorithm.
- __dalib__
  -  Implements for datasets such as PACS, Office-Home, and Multi-Datasets.
- __datautil__
  - Implements loading and preprocessing datasets.
- __eval_hscore.py__
  - Implements H-score of the algorithm after training.
- __eval.py__
  - Implements Accuracy of the algorithm after training.
- __network__
  - Implements algorithm networks.
- __scripts__
  - Scripts to conduct experiments.
- __train.py__
  - Implements training for each algorithm. This code is used in common with the algorithms.
- __utils__
  - Implementation of evaluation metrics and functions used for getter function and DAML.

## Requirement
```
Python 3.8.0
Pytorch 1.12.1
```


## Usage
1. Prepare the dataset (PACS[1], Office-Home[2], STL-10[3], Office-31[4], VisDA2017[5] and DomainNet[6]) and modify the file path in the scripts.
2. The main script files are `train.py`, `eval.py`, and `eval_hscore.py`, which can be runned by using `run.sh` from `scripts/run.sh`: `cd scripts; bash run.sh`.

## Acknowledgment
Great thanks to DeepDG[7]. We built on their work to implement Open Domain Generalization.

## References
[1] Li, Da, et al. "Deeper, broader and artier domain generalization." Proceedings of the IEEE international conference on computer vision. 2017.

[2] Venkateswara, Hemanth, et al. "Deep hashing network for unsupervised domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[3] Coates, Adam, Andrew Ng, and Honglak Lee. "An analysis of single-layer networks in unsupervised feature learning." Proceedings of the fourteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 2011.

[4] Saenko, Kate, et al. "Adapting visual category models to new domains." European Conference on Computer Vision. 2010.

[5] Peng, Xingchao, et al. "Visda: A synthetic-to-real benchmark for visual domain adaptation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.

[6] Peng, Xingchao, et al. "Moment matching for multi-source domain adaptation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

[7] Wang, Jindong, et al. "Generalizing to unseen domains: A survey on domain generalization." IEEE Transactions on Knowledge and Data Engineering (2022).