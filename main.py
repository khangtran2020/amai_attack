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
        AMIADatasetCelebA(args, target, transform, args.data_path, 'valid', imgroot=None, multiplier=args.num_draws), shuffle=False,
        num_workers=0, batch_size=200000)
    test_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args, target, transform, args.data_path, 'valid', imgroot=None, multiplier=args.num_draws),
        shuffle=False,
        num_workers=0, batch_size=200000)

    model, name = train(args=args, device=device, data=(train_loader, valid_loader), model=model)
    model_name = args.save_path + name + '.pt'
    model = torch.load(model_name)
    x_valid, y_valid, imgs_valid = next(iter(valid_loader))
    custom_weight = np.array([1600.0, 200.0])
    criteria = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))
    res_test = evaluate(x=x_valid, y=y_valid, model=model, criteria=criteria, device=device)
    print(res_test)
    exit()

    # del(train_loader)
    # del(valid_loader)
    # gc.collect()
    result = evaluate_robust(args=args, data = valid_loader, model=model)
    print(result)
    json_object = json.dumps(result, indent=4)
    SAVE_NAME = 'CELEBA_embed_Lap_single_{}_{}_lr_{}.json'.format(args.num_target, args.epsilon, args.lr)
    # Writing to sample.json
    with open(args.save_path + SAVE_NAME, "w") as outfile:
        outfile.write(json_object)





if __name__ == '__main__':
    args = parse_args()
    max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
    args.sens = 2 * max_
    args.num_label = args.num_target + 1
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    args.num_test_point = 2*args.num_draws
    run(args, device)
