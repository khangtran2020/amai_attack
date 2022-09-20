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
from multiprocessing import Process

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_HOME'] = "./Model/pretrain_model"


def run(args, target, device):
    if args.train_mode == 'triplet':
        print('Train with mode triplet')
        model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_loader = torch.utils.data.DataLoader(
            CelebATriplet(args=args, target=target, transform=transform, dataroot=args.data_path, mode='train',
                          imgroot=None, multiplier=args.train_multiplier),
            shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
            CelebATriplet(args=args, target=target, transform=transform, dataroot=args.data_path, mode='valid',
                          imgroot=None, multiplier=args.valid_multiplier),
            shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        print("Sensitivity: {}, Number of features: {}, epsilon used in training: {}, noise scale: {}".format(args.sens,
                                                                                                              args.num_feature,
                                                                                                              args.epsilon,
                                                                                                              args.sens /
                                                                                                              args.epsilon))
        model = train_triplet(args=args, target=target, device=device, data=(train_loader, valid_loader),
                              model=model)

    elif args.train_mode == 'triplet-full':
        print('Train with mode triplet full')
        model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
        train_loader = torch.utils.data.DataLoader(
            CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='train',
                              imgroot=None, multiplier=args.train_multiplier),
            shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
            CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='valid',
                              imgroot=None, multiplier=args.valid_multiplier),
            shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        print("Sensitivity: {}, Number of features: {}, epsilon used in training: {}, noise scale: {}".format(args.sens,
                                                                                                              args.num_feature,
                                                                                                              args.epsilon,
                                                                                                              args.sens /
                                                                                                              args.epsilon))
        model = train_triplet_full(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                   model=model)
    elif args.train_mode == 'triplet-fun':
        print('Train with mode triplet fun')
        model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
        train_loader = torch.utils.data.DataLoader(
            CelebATripletFun(args=args, target=target, dataroot=args.data_path, mode='train',
                             imgroot=None, multiplier=args.train_multiplier),
            shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        valid_loader = torch.utils.data.DataLoader(
            CelebATripletFun(args=args, target=target, dataroot=args.data_path, mode='valid',
                             imgroot=None, multiplier=args.valid_multiplier),
            shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        print("Sensitivity: {}, Number of features: {}, epsilon used in training: {}, noise scale: {}".format(args.sens,
                                                                                                              args.num_feature,
                                                                                                              args.epsilon,
                                                                                                              args.sens /
                                                                                                              args.epsilon))
        model = train_triplet_fun(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                  model=model)
    model.to('cpu')
    target_data, target_label = init_target_data(args, target)
    list_of_cert_eps = cert(args=args, model=model, target_data=target_data, device=device)
    if len(list_of_cert_eps) == 0:
        print("Didn't ceritfied")
        exit()
    manager = multiprocessing.Manager()
    results = manager.dict()
    results['certified_for_target'] = {
        'search_range_min': args.min_epsilon,
        'search_range_max': args.max_epsilon,
        'certified': 'yes',
        'list of eps': list_of_cert_eps,
        'confidence': 1 - args.alpha
    }
    results['number_of_test_set'] = args.num_test_set
    results['sample_target_rate'] = args.sample_target_rate
    results['result_of_eps'] = {}
    jobs = []
    for eps in list_of_cert_eps:
        # (eps, args, results, target, target_data, target_label, model, device)
        p = multiprocessing.Process(target=perform_attack_parallel, args=(eps, args, results, target, target_data, target_label, model))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(results)
    json_object = json.dumps(results, indent=4)
    # Writing to sample.json
    with open(args.save_path + args.save_result_name, "w") as outfile:
        outfile.write(json_object)
    exit()


if __name__ == '__main__':
    args = parse_args()
    max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
    args.sens = 2 * max_
    args.num_label = args.num_target + 1
    args.noise_scale = args.sens / args.epsilon
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    target = [int(i) for i in args.target.split('-')]
    if args.debug:
        args.save_model_name = 'debugging_eps_{}_reg_{}.pt'.format(args.epsilon, args.reg)
        args.save_result_name = 'debugging_eps_{}_reg_{}_ver_{}.json'.format(args.epsilon, args.reg, args.test_ver)
    else:
        if args.num_target == 1:
            args.save_model_name = 'CELEBA_single_Laplace_eps_{}.pt'.format(args.epsilon)
            args.save_result_name = 'CELEBA_single_Laplace_eps_{}.json'.format(args.epsilon)
        else:
            args.save_model_name = 'CELEBA_multiple_{}_Lap_eps_{}.pt'.format(args.num_target, args.epsilon)
            args.save_result_name = 'CELEBA_multiple_{}_Lap_eps_{}.json'.format(args.num_target, args.epsilon)
    run(args=args, target=target, device=device)
