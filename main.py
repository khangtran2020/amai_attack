import numpy as np
import torch
from Data.celeba import *
from Model.model import *
# from MomentAccountant.get_priv import *
from Bound.robustness import *
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
    test_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args, target, transform, args.data_path, 'test', imgroot=None, multiplier=100), shuffle=False,
        num_workers=0, batch_size=200000)
    if args.mode == 'train':
        train(args=args, device=device, data=(train_loader, test_loader), model=model)
        # if args.train_mode == 'clean':
        #     train_clean(args=args, device=device, nodes=nodes, hnet=hnet, net=net)
        # elif args.train_mode == 'userdp':
        #     train_userdp(args=args, device=device, nodes=nodes, hnet=hnet, net=net)
        #     criteria = torch.nn.CrossEntropyLoss()
        #     robust_result = evaluate_robust_udp(args=args, num_nodes=args.num_client, nodes=nodes, hnet=hnet, net=net,
        #                                         criteria=criteria)
        #     with open(
        #             args.save_path + "robustness_results_numClient_{}_bt_{}_noiseScale_{}_numDraw_{}_epsilon_{:.2f}.json".format(
        #                 args.num_client, args.bt, args.noise_scale, args.num_draws_udp, args.udp_epsilon),
        #             "w") as outfile:
        #         json.dump(robust_result, outfile)


if __name__ == '__main__':
    args = parse_args()
    max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
    args.sens = 2 * max_
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    run(args, device)
