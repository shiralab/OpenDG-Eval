#!/bin/bash
is_different_class_space=1
is_data_aug=yes
loader_name='daml_loader'
#dataset='office-home'
data_dir='./dataset/OfficeHome/'
#dataset='PACS'
#data_dir='./dataset/PACS/'


algorithms=('ARPL' 'ANDMask' 'CORAL' 'DAML' 'DANN' 'DIFEX' 'ERM' 'Mixup' 'MLDG' 'MMD' 'RSC' 'VREx')
#algorithms=('DAML_wo_Dir_mixup' 'DAML_wo_distill' 'DAML_wo_Dmix_and_dst' 'Ensemble_CORAL' 'Ensemble_CORAL_with_Dir_mixup' 'Ensemble_CORAL_with_Distill' 'Ensemble_MMD' 'Ensemble_MMD_with_Dir_mixup' 'Ensemble_MMD_with_Distill')
algorithm_num=${#algorithms[@]}

net='resnet18'
task='img_dg'
lr=1e-3
test_envs=0
gpu_id=0
max_epoch=100
steps_per_epoch=100


for ((seed=0; seed < 3; seed++)); do
    for ((i=0; i < $algorithm_num; i++)); do
        output='./output/'${loader_name}'/lr_'${lr}'_dataaug_'${is_data_aug}'/'${dataset}'_different_class_space/test_env'${test_envs}'/'${algorithms[i]}'/test_envs_'${test_envs}'/seed_'${seed}
        # train
        python train.py --seed $seed --lr $lr --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
        --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
        --is_different_class_space $is_different_class_space --is_data_aug $is_data_aug --loader_name $loader_name
        # acc and auroc
        python eval.py --seed $seed --lr $lr --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
        --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
        --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug
        # h-score
        python eval_hscore.py --seed $seed --lr $lr --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output \
        --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
        --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug
    done
done
