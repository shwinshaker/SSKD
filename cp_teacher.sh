

# method='cifar100_lr_0.05_decay_0.0005'
method='cifar100_lr_0.05_decay_0.0005_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_curl_alpha=1_linear_swa=150'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear'
# method='cifar100_lr_0.05_decay_0.0005_swareg_alpha=1_linear_swa=150'

# method_abbr="vanilla"
method_abbr="swa"
# method_abbr="curl+swa"
# method_abbr="swar+swa"

model="wrn_40_2"
trial=0

# -
src_path="save/models/"$model"_"$method"_trial_"$trial""
path="experiments/teacher="$method_abbr"_"$model"_seed"$trial"/ckpt"

mkdir -p $path
cur=$PWD
cd $path
ln -s ~/Distillation/"$src_path"/"$model"_last.pth best.pth
cd $cur
