#!/bin/bash

# -- Setting --
gpu=7
ce_weight=1.0
kd_weight=1.0
tf_weight=2.7 # 1.0
ss_weight=10.0

kd_T=4.0
tf_T=4.0 # 1.0
ss_T=0.5 # 1.0

ratio_tf=1.0
ratio_ss=0.75

# t_epoch='60'
# t_milestones='30 45'

# teacher='wrn_40_2'
teacher='resnet32x4'
# student='wrn_16_2'
# student='wrn_40_1'
student='ShuffleV2' 
# student='ShuffleV1' 
# student='resnet8x4' 

method='vanilla'
# method="lip"
# method="lip+omse+linear"
# method="lip=5e-7+omse+linear"
# method="omse+linear"
# method='swa'
# method='crl+swa'
# method='curl+swa'
# method='swar+swa'
trial=1

# -- Config --
python config_student.py --t-path ./experiments/teacher="$method"_"$teacher"_seed"$trial"/ --s-arch "$student" --gpu-id "$gpu" \
                         --ce-weight "$ce_weight" --kd-weight "$kd_weight" --tf-weight "$tf_weight" --ss-weight "$ss_weight" \
                         --kd-T $kd_T --tf-T $tf_T --ss-T $ss_T \
                         --ratio-tf $ratio_tf --ratio-ss $ratio_ss
                        #  --t-epoch $t_epoch --t-milestones $t_milestones
[[ $? -ne 0 ]] && echo 'exit' && exit 2

# -- RUN --
path=$(cat tmp/config.tmp | grep 'exp_path' | awk '{print$NF}' | tr -d '"')
cp tmp/config.tmp $path
python student.py --config_file "$path/config.tmp" > $path/train.out 2>&1 &

pid=$!
echo "[$pid] [$gpu] [Path]: $path"
echo "s [$pid] [$gpu] $(date) [Path]: $path" >> logs/log.txt
