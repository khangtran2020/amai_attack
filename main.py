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
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TORCH_HOME'] = "./Model/pretrain_model"


def run(args, device):
    # Init data
    model = Classifier(args.num_feature, args.num_target + 1)
    target = list(range(args.num_target))
    # print(args.epsilon)
    # print(args.sens*args.num_feature/args.epsilon)
    # exit()
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        CelebA(args, target, transform, args.data_path, 'train', imgroot=None, multiplier=args.train_multiplier),
        shuffle=False,
        num_workers=0, batch_size=args.batch_size)
    valid_loader = torch.utils.data.DataLoader(
        CelebA(args, target, transform, args.data_path, 'valid', imgroot=None, multiplier=args.num_draws),
        shuffle=False,
        num_workers=0, batch_size=args.batch_size)
    model = train(args=args, device=device, data=(train_loader, valid_loader), model=model)

    if args.train_mode == 'target':
        results = {}
        results['number_of_test_set'] = args.num_test_point
        results['sample_target_rate'] = args.sample_target_rate
        results['res_of_each_test'] = {}
        true_label = []
        predicted = []
        x_t = None
        # for i in tqdm(range(args.num_test_point)):
        # print(args.num_test_point)
        for i in range(args.num_test_set):
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            true_label.append(int(sample[0]))
            test_loader = torch.utils.data.DataLoader(
                CelebA(args, target, transform, args.data_path, 'test', imgroot=None,
                       multiplier=args.valid_multiplier, include_tar=sample[0]), shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            x_test, y_test, file_name = next(iter(test_loader))
            if sample[0]:
                temp_x = x_test.numpy()
                temp_x[0] = temp_x[0] + np.random.laplace(0, args.sens * args.num_feature / args.epsilon,
                                                          temp_x[0].shape)
                x_test = torch.from_numpy(temp_x.astype(np.float32))
            # weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(args.num_target + 1),
            #                                                          y=y_test.cpu().detach().numpy())
            criteria = nn.CrossEntropyLoss()
            model.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            out, probs, fc2 = model(x_test)
            loss = criteria(out, y_test).item()
            pred = fc2[:, 0] < 0
            print(sample, pred, sum(1 - pred.cpu().numpy().astype(int)), min(1, sum(1 - pred.cpu().numpy().astype(int))))
            acc = accuracy_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            precision = precision_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            recall = recall_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            results['res_of_each_test']['test_{}'.format(i)] = {
                'loss': loss,
                'acc': acc,
                'precision': precision,
                'recall': recall
            }
            pred_ = 1 - pred.cpu().numpy().astype(int)
            if sum(pred_ <= 0):
                print('Test {}'.format(i))
                predicted.append(0)
                results['res_of_each_test']['test_{}'.format(i)]['has_target'] = int(sample[0])
                results['res_of_each_test']['test_{}'.format(i)]['predict'] = 0
            else:
                predicted.append(1)
                results['res_of_each_test']['test_{}'.format(i)]['has_target'] = int(sample[0])
                results['res_of_each_test']['test_{}'.format(i)]['predict'] = 1
                # print("Test", i)
        acc = accuracy_score(true_label, predicted)
        precision = precision_score(true_label, predicted)
        recall = recall_score(true_label, predicted)
        results['test_result'] = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
        }
        test_loader = torch.utils.data.DataLoader(
            CelebA(args, target, transform, args.data_path, 'test', imgroot=None,
                   multiplier=args.num_draws, include_tar=True), shuffle=False,
            num_workers=0, batch_size=args.batch_size)
        x_test, y_test, file_name = next(iter(test_loader))
        epsilon_of_point = args.max_epsilon
        certified = 0
        # for eps in tqdm(np.linspace(args.min_epsilon, args.max_epsilon, 100)):
        for i, eps in enumerate(np.linspace(args.min_epsilon, args.max_epsilon, 100)):
            temp_x = x_test.numpy()
            generated_target = np.tile(temp_x[0, :], (args.num_draws + 1, 1))
            # print(generated_target)
            # exit()
            generated_target[1:, :] = generated_target[1:, :] + np.random.laplace(0,
                                                                                  args.sens * args.num_feature / eps,
                                                                                  generated_target[1:, :].shape)
            # print(generated_target)
            # exit()
            temp_x = torch.from_numpy(generated_target.astype(np.float32)).to(device)
            out, probs, fc2 = model(temp_x)
            pred = fc2[:, 0].cpu().detach().numpy()
            # print(pred)
            # exit()
            same_sign = (pred[1:] * pred[0]) > 0
            # print(same_sign)
            count_of_same_sign = sum(same_sign.astype(int))
            count_of_diff_sign = args.num_draws - count_of_same_sign
            print(
                'For eps {}, # same sign: {}, # diff sign: {}'.format(
                    eps, count_of_same_sign, count_of_diff_sign))
            upper_bound = hoeffding_upper_bound(count_of_diff_sign, nobs=args.num_draws, alpha=args.alpha)
            lower_bound = hoeffding_lower_bound(count_of_same_sign, nobs=args.num_draws, alpha=args.alpha)
            if (lower_bound > upper_bound):
                certified = int(1.0)
                epsilon_of_point = min(epsilon_of_point, eps)
            # print("Done for eps: {}".format(eps))
        results['certified_for_target'] = {
            'search_range_min': args.min_epsilon,
            'search_range_max': args.max_epsilon,
            'certified': certified,
            'eps_min': epsilon_of_point,
            'confidence': 1 - args.alpha
        }

        print(results)
        json_object = json.dumps(results, indent=4)
        # SAVE_NAME = 'CELEBA_train_eps_{}_sample_rate_{}_num_step.json'.format(args.epsilon ,
        #                                                                                args.sample_target_rate,
        #                                                                                args.num_steps)
        SAVE_NAME = 'test_{}.json'.format(args.test_ver)
        # Writing to sample.json
        with open(args.save_path + SAVE_NAME, "w") as outfile:
            outfile.write(json_object)
        exit()

    # del(train_loader)
    # del(valid_loader)
    # gc.collect()
    result = evaluate_robust(args=args, data=valid_loader, model=model)
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
    # args.num_test_point = 2 * args.num_draws
    run(args, device)
