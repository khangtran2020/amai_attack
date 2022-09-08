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
        CelebA(args, target, transform, args.data_path, 'train', imgroot=None, multiplier=args.train_multiplier), shuffle=False,
        num_workers=0, batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(
        CelebA(args, target, transform, args.data_path, 'valid', imgroot=None, multiplier=args.num_draws), shuffle=False,
        num_workers=0, batch_size=args.batch_size)
    model = train(args=args, device=device, data=(train_loader, valid_loader), model=model)

    if args.train_mode == 'target':
        results = {}
        true_label = []
        predicted = []
        for i in tqdm(range(args.num_test_point)):
            sample = np.random.binomial(n=1, p=args.sample_target_rate,size=1).astype(bool)
            true_label.append(int(sample[0]))
            test_loader = torch.utils.data.DataLoader(
                CelebA(args, target, transform, args.data_path, 'test', imgroot=None,
                                  multiplier=args.num_draws, include_tar=sample[0]), shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            x_test, y_test, file_name = next(iter(test_loader))
            weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(args.num_target + 1),
                                                                     y=y_test.cpu().detach().numpy())
            criteria = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
            out, probs, fc2 = model(x_test)
            loss = criteria(out, y_test).item()
            pred = fc2[:, 0] < 0
            predicted.append(min(1,sum(pred.cpu().numpy())))
            tpr, tnr, acc = tpr_tnr(pred, y_test)
            results['test_{}'.format(i)] = {
                'loss': loss,
                'acc': acc,
                'tpr': tpr,
                'tnr': tnr,
                'has_target': sample[0],
                'predicted': bool(min(1,sum(pred.cpu().numpy())))
            }
        print(true_label, predicted)
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
