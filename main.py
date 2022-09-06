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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TORCH_HOME'] = "./Model/pretrain_model"


def run(args, device):
    # Init data
    model = Classifier(args.num_feature, args.num_target + 1)
    target = list(range(args.num_target))
    print(args.epsilon)
    # print(args.sens*args.num_feature/args.epsilon)
    # exit()
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args, target, transform, args.data_path, 'train', imgroot=None, multiplier=2000), shuffle=False,
        num_workers=0, batch_size=200000)
    valid_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args, target, transform, args.data_path, 'valid', imgroot=None, multiplier=100), shuffle=False,
        num_workers=0, batch_size=200000)
    test_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args, target, transform, args.data_path, 'valid', imgroot=None, multiplier=100),
        shuffle=False,
        num_workers=0, batch_size=200000)

    model = train(args=args, device=device, data=(train_loader, valid_loader), model=model)
    del(train_loader)
    del(valid_loader)
    gc.collect()
    result = evaluate_robust(args=args, data = test_loader, model=model, device=device)
    print(result)




if __name__ == '__main__':
    args = parse_args()
    max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
    args.sens = 2 * max_
    args.num_label = args.num_target + 1
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    run(args, device)
