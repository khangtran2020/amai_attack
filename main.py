import gc

import numpy as np
import torch
from Data.celeba import *
from Model.model import *
# from MomentAccountant.get_priv import *
from Bound.evaluate import *
from Utils.utils import *
from config import parse_args
import os
import json
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_HOME'] = "./Model/pretrain_model"


def run(args, target, device):
    target_data, target_label = init_target_data(args=args, target=target)
    if args.train_mode == 'normal':
        pass
    elif args.train_mode == 'triplet':
        model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
        train_loader = torch.utils.data.DataLoader(
            CelebATripletFull(args=args,
                              target=target, dataroot=args.data_path,
                              mode='train',
                              imgroot=None,
                              multiplier=args.train_multiplier),
            shuffle=False,
            num_workers=0,
            batch_size=args.batch_size
        )
        valid_loader = torch.utils.data.DataLoader(
            CelebATripletFull(args=args,
                              target=target,
                              dataroot=args.data_path,
                              mode='valid',
                              imgroot=None,
                              multiplier=args.valid_multiplier),
            shuffle=False,
            num_workers=0,
            batch_size=args.val_batch_size
        )
        model = train_triplet_full(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                   model=model)
        results, certified_eps, certified = cert(args=args, model=model, target_data=target_data,device=device)
        res = evaluate_test(args=args, model=model,certified=certified, target=target,target_data=target_data, target_label=target_label, eps_cert=certified_eps,device=device)
        for key, value in res.items():
            results[key] = value
        json_object = json.dumps(results, indent=4)
        SAVE_NAME = args.save_name
        with open(args.save_path + SAVE_NAME, "w") as outfile:
            outfile.write(json_object)


if __name__ == '__main__':
    args = parse_args()
    max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
    args.sens = 2 * max_
    args.num_label = args.num_target + 1
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    if args.debug:
        args.save_name = 'debugging_eps_{}_reg_{}.pt'.format(args.epsilon, args.reg)
    else:
        if args.num_target == 1:
            args.save_name = 'CELEBA_single_Laplace_eps_{}.pt'.format(args.epsilon)
        else:
            args.save_name = 'CELEBA_multiple_{}_Lap_eps_{}.pt'.format(args.num_target, args.epsilon)
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    target = [int(i) for i in args.target.split('-')]
    run(args=args, target=target, device=device)
