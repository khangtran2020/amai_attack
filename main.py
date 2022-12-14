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
from multiprocessing import Pool

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_HOME'] = "./Model/pretrain_model"


def run(args, target, device, logger):
    if args.main_mode == 'train':
        if args.train_mode == 'triplet':
            print('Train with mode triplet')
            model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_loader = torch.utils.data.DataLoader(
                CelebATriplet(args=args,
                              target=target,
                              transform=transform,
                              dataroot=args.data_path,
                              mode='train',
                              imgroot=None,
                              multiplier=args.train_multiplier
                ),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            valid_loader = torch.utils.data.DataLoader(
                CelebATriplet(args=args,
                              target=target,
                              transform=transform,
                              dataroot=args.data_path,
                              mode='valid',
                              imgroot=None,
                              multiplier=args.valid_multiplier
                ),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            model = train_triplet(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                  model=model)

        elif args.train_mode == 'triplet-full':
            print('Train with mode triplet full')
            model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
            train_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args,
                                  target=target,
                                  dataroot=args.data_path,
                                  mode='train',
                                  imgroot=None,
                                  multiplier=args.train_multiplier
                ),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            valid_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args,
                                  target=target,
                                  dataroot=args.data_path,
                                  mode='valid',
                                  imgroot=None,
                                  multiplier=args.valid_multiplier
                ),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            model = train_triplet_full(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                       model=model)
        elif args.train_mode == 'triplet-fun':
            print('Train with mode triplet fun')
            model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
            train_loader = torch.utils.data.DataLoader(
                CelebATripletFun(args=args,
                                 target=target,
                                 dataroot=args.data_path,
                                 mode='train',
                                 imgroot=None,
                                 multiplier=args.train_multiplier
                ),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            valid_loader = torch.utils.data.DataLoader(
                CelebATripletFun(args=args,
                                 target=target,
                                 dataroot=args.data_path,
                                 mode='valid',
                                 imgroot=None,
                                 multiplier=args.valid_multiplier
                ),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            model = train_triplet_fun(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                      model=model)

    else:
        print("Model name:", args.save_path + args.save_model_name)
        model = torch.load(args.save_path + args.save_model_name)
    target_data, target_label = init_target_data(args, target)
    # list_of_cert_eps = cert(args=args, model=model, target_data=target_data, device=device)
    list_of_cert_eps = cert_2side(args=args, model=model, target_data=target_data, target=target, device=device)
    if len(list_of_cert_eps) == 0:
        print("Didn't ceritfied")
        exit()
    results = {}
    results['certified_for_target'] = {
        'search_range_min': args.min_epsilon,
        'search_range_max': args.max_epsilon,
        'certified': 'yes',
        'list of eps': list_of_cert_eps,
        'confidence': 1 - args.alpha
    }
    # results = perform_attack_test(args=args, results=results, target=target, target_data=target_data, target_label=target_label,list_of_eps=list_of_cert_eps, model=model, device=device)
    # print(results)
    results['number_of_test_set'] = args.num_test_set
    results['sample_target_rate'] = args.sample_target_rate
    results['result_of_eps'] = {}
    model.to('cpu')
    print("Start multiprocessing")
    temp_args = (args, results, target, target_data, target_label, model, 'cpu', logger)
    items = []
    for eps in list_of_cert_eps:
        items.append((temp_args, eps))
    if args.val_mode == 'test':
        with Pool(10) as p:
            res = list(p.apply_async(perform_attack_test_parallel, args=(temp_args, eps)) for eps in list_of_cert_eps)
            res = [r.get() for r in res]
        for i, eps in enumerate(list_of_cert_eps):
            results['result_of_eps']['eps {:.2f}'.format(eps)] = res[i]
    else:
        with Pool(10) as p:
            res = list(
                p.apply_async(perform_attack_test_parallel_same, args=(temp_args, eps)) for eps in list_of_cert_eps)
            res = [r.get() for r in res]
        for i, eps in enumerate(list_of_cert_eps):
            results['result_of_eps']['eps {:.2f}'.format(eps)] = res[i]
    print(results)
    json_object = json.dumps(results, indent=4)
    # Writing to sample.json
    with open(args.save_path + args.save_result_name, "w") as outfile:
        outfile.write(json_object)
    exit()


if __name__ == '__main__':
    args = parse_args()
    # max_tensor = torch.load(args.max_path)
    # max_tensor = torch.from_numpy((np.ones(args.num_feature)*11.2836).astype(np.float32))
    args.sens = 11.2836
    args.num_label = args.num_target + 1
    args.noise_scale = args.sens / args.epsilon
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    target = [int(i) for i in args.target.split('-')]
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger = logging.getLogger('exp')
    logger.setLevel(logging.INFO)
    if args.debug:
        args.save_model_name = 'debugging_eps_{}_max_eps_{}_epochs_{}_samplerate_{}_lr_{}_trmul_{}_ver_{}.pt'.format(
            args.epsilon, args.max_epsilon, args.num_steps, args.sample_rate, args.lr, args.train_multiplier,
            args.test_ver)
        args.save_result_name = 'debugging_eps_{}_max_eps_{}_epochs_{}_samplerate_{}_lr_{}_trmul_{}_ver_{}.json'.format(
            args.epsilon, args.max_epsilon, args.num_steps, args.sample_rate, args.lr, args.train_multiplier,
            args.test_ver)
    else:
        if args.num_target == 1:
            args.save_model_name = 'CELEBA_single_Laplace_eps_{}.pt'.format(args.epsilon)
            args.save_result_name = 'CELEBA_single_Laplace_eps_{}.json'.format(args.epsilon)
        else:
            args.save_model_name = 'CELEBA_multiple_{}_Lap_eps_{}.pt'.format(args.num_target, args.epsilon)
            args.save_result_name = 'CELEBA_multiple_{}_Lap_eps_{}.json'.format(args.num_target, args.epsilon)
    if args.main_mode == 'og':
        args.save_model_name = 'temp_model.pt'
        args.save_result_name = 'temp_results.json'
    run(args=args, target=target, device=device, logger=logger)
