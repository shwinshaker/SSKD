#!/bin/bash

# -- Setting --
gpu=7
ce_weight=0.1
kd_weight=0.9
tf_weight=2.7
ss_weight=10.0

teacher='wrn_40_2'
# teacher='resnet32x4'
student='wrn_40_1'
# student='ShuffleV2' 
# student='resnet8x4' 

# method='cifar100_lr_0.05_decay_0.0005'
# method='cifar100_lr_0.05_decay_0.0005_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_curl_alpha=1_linear_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear_swa=150'
method='swar+swa'
trial=0

# -- Config --
python config_student.py --t-path ./experiments/teacher="$method"_"$teacher"_seed"$trial"/ --s-arch "$student" --gpu-id "$gpu" --ce-weight "$ce_weight" --kd-weight "$kd_weight" --tf-weight "$tf_weight" --ss-weight "$ss_weight"
# python config_student.py --path_t ./save/models/"$teacher"_"$method"_trial_"$trial"/"$teacher"_last.pth --distill $distill_method --model_s "$student" -r "$gamma" -a "$alpha" -b "$beta" --gpu "$gpu"
# python config_student.py --path_t ./save/models/wrn_40_2_cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear_trial_"$trial"/wrn_40_2_last.pth --distill $distill_method --model_s wrn_40_1 -a "$alpha" -b "$beta" --gpu "$gpu"
# python config_student.py --path_t ./save/models/"$teacher"_cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear_swa=150_trial_"$trial"/"$teacher"_last.pth --distill $distill_method --model_s "$student" -r "$gamma" -a "$alpha" -b "$beta" --gpu "$gpu"
[[ $? -ne 0 ]] && echo 'exit' && exit 2

# -- RUN --
path=$(cat tmp/config.tmp | grep 'exp_path' | awk '{print$NF}' | tr -d '"')
cp tmp/config.tmp $path
python student.py --config_file "$path/config.tmp" > $path/train.out 2>&1 &

pid=$!
echo "[$pid] [$gpu] [Path]: $path"
echo "s [$pid] [$gpu] $(date) [Path]: $path" >> logs/log.txt
