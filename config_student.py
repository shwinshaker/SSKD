import argparse
import json
import os

from helper.util import check_path

def parse_option():

    parser = argparse.ArgumentParser(description='train SSKD student network.')
    parser.add_argument('--t-epoch', type=int, default=60)
    parser.add_argument('--t-lr', type=float, default=0.05)
    parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

    parser.add_argument('--epoch', type=int, default=240)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])

    # parser.add_argument('--save-interval', type=int, default=40)
    parser.add_argument('--ce-weight', type=float, default=0.1) # cross-entropy
    parser.add_argument('--kd-weight', type=float, default=0.9) # knowledge distillation
    parser.add_argument('--tf-weight', type=float, default=2.7) # transformation
    parser.add_argument('--ss-weight', type=float, default=10.0) # self-supervision

    parser.add_argument('--kd-T', type=float, default=4.0) # temperature in KD
    parser.add_argument('--tf-T', type=float, default=4.0) # temperature in LT
    parser.add_argument('--ss-T', type=float, default=0.5) # temperature in SS

    parser.add_argument('--ratio-tf', type=float, default=1.0) # keep how many wrong predictions of LT
    parser.add_argument('--ratio-ss', type=float, default=0.75) # keep how many wrong predictions of SS
    parser.add_argument('--s-arch', type=str) # student architecture
    parser.add_argument('--t-path', type=str) # teacher checkpoint path

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)

    args = parser.parse_args()

    print('==> Make path..')
    t_name = os.path.abspath(args.t_path).split('/')[-1]
    args.t_arch = '_'.join(t_name.split('_')[1:-1])
    exp_name = 'sskd_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}+{}_seed{}_{}'.format(\
                     args.s_arch, \
                     args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight, \
                     args.kd_T, args.tf_T, args.ss_T, \
                     args.ratio_tf, args.ratio_ss, \
                     args.seed, t_name)
    args.exp_path = './experiments/{}'.format(exp_name)

    # set different learning rate from these 4 models
    if args.s_arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        print('   Lr reset to 0.01')
        args.lr = 0.01

    check_path(args.exp_path)
    print('   path: %s' % args.exp_path)

    return args


if __name__ == '__main__':
    opt = parse_option()

    # Save setting to json
    with open('tmp/config.tmp', 'wt') as f:
        json.dump(vars(opt), f, indent=4)