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
    model = Classifier(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args=args, target=target, transform=transform, dataroot=args.data_path, mode='train', imgroot=None, multiplier=args.train_multiplier),
        shuffle=False,
        num_workers=0, batch_size=200000)
    valid_loader = torch.utils.data.DataLoader(
        AMIADatasetCelebA(args=args, target=target, transform=transform, dataroot=args.data_path, mode='valid', imgroot=None, multiplier=args.valid_multiplier),
        shuffle=False,
        num_workers=0, batch_size=200000)
    print("Sensitivity: {}, Number of features: {}, epsilon used in training: {}, noise scale: {}".format(args.sens, args.num_feature,
                                                                                         args.epsilon, args.sens/
                                                                                         args.epsilon))
    model = train(args=args, target=target, device=device, data=(train_loader, valid_loader), model=model)
    if args.train_mode == 'target':
        results = {}
        data_name = sorted(os.listdir(args.data_path))
        list_target = []
        list_target_label = []
        for i, f in enumerate(target):
            list_target.append(torch.unsqueeze(torch.load(args.data_path + data_name[f]), 0))
            list_target_label.append(i)
        list_target = tuple(list_target)
        target_data = torch.cat(list_target, 0)
        target_label = torch.from_numpy(np.array(list_target_label))
        epsilon_of_point = args.max_epsilon
        certified = 0
        for i, eps in enumerate(np.linspace(args.min_epsilon, args.max_epsilon, 100)):
            temp_x = target_data.numpy()
            noise_scale = args.sens / eps
            print("Sensitivity: {}, Number of features: {}, epsilon used in certify: {}, noise scale: {}".format(
                args.sens, args.num_feature,
                eps, noise_scale))
            generated_target_org = np.tile(temp_x, (args.num_draws + 1, 1))
            noise = np.random.laplace(0, noise_scale, generated_target_org[1:, :].shape)
            generated_target = generated_target_org.copy()
            generated_target[1:, :] = generated_target[1:, :] + noise
            print('L2 norm of noise: {}', np.linalg.norm(noise, ord=2))
            print('L2 distance: {}', np.linalg.norm(generated_target - generated_target_org, ord=2))
            temp_x = torch.from_numpy(generated_target.astype(np.float32)).to(device)
            fc1, fc2, out = model(temp_x)
            pred = fc2[:, 0].cpu().detach().numpy()
            print(pred)
            # exit()
            # same_sign = (pred[1:] * pred[0]) > 0
            larger_than_zero = pred > 0
            # print(same_sign)
            # count_of_same_sign = sum(larger_than_zero.astype(int))
            # count_of_diff_sign = args.num_draws - count_of_same_sign
            count_of_larger_than_zero = sum(larger_than_zero.astype(int))
            count_of_smaller_than_zero = args.num_draws - count_of_larger_than_zero
            print(
                'For eps {}, # larger than 0: {}, # smaller or equal to 0: {}'.format(
                    eps, count_of_larger_than_zero, count_of_smaller_than_zero))
            upper_bound = hoeffding_upper_bound(count_of_smaller_than_zero, nobs=args.num_draws, alpha=args.alpha)
            lower_bound = hoeffding_lower_bound(count_of_larger_than_zero, nobs=args.num_draws, alpha=args.alpha)
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
        exit()
        results['number_of_test_set'] = args.num_test_set
        results['sample_target_rate'] = args.sample_target_rate
        results['res_of_each_test'] = {}
        true_label = []
        predicted = []
        if certified:
            noise_scale = args.sens / (epsilon_of_point * 200)
            print("Noise scale fore the attack:", noise_scale)
        else:
            print("Didn't ceritfied")
            exit()
        for i in range(args.num_test_set):
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            test_loader = torch.utils.data.DataLoader(
                CelebA(args, target, transform, args.data_path, 'test', imgroot=None, include_tar=sample[0]),
                shuffle=False,
                num_workers=0, batch_size=args.num_test_point)
            x_test, y_test, file_name = next(iter(test_loader))
            true_label.append(sample[0])
            if sample[0]:
                x_test = torch.cat((target_data, x_test), 0)
                y_test = torch.cat((target_label, y_test), 0)
                temp_x = x_test.numpy()
                noise = np.random.laplace(0,noise_scale,temp_x[:args.num_target].shape)
                temp_x[:args.num_target] = temp_x[:args.num_target] + noise
                x_test = torch.from_numpy(temp_x.astype(np.float32))
            criteria = nn.CrossEntropyLoss()
            model.to(device)
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            fc1, fc2, out = model(x_test)
            loss = criteria(out, y_test).item()
            pred = fc2[:, 0] > 0 # whether it not activated True if its activated, False if it's not activated
            # a = 1 - pred.cpu().numpy().astype(int) # whether it activated / a[i] = True => i is the target
            # sum(a) # if 1 of them is activated -> >= 1, if all ofthem are not activated -> 0
            print("Test {}".format(i),sample, pred, sum(pred.cpu().numpy().astype(int)), min(1, sum(pred.cpu().numpy().astype(int))))
            print(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            acc = accuracy_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            precision = precision_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            recall = recall_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
            results['res_of_each_test']['test_{}'.format(i)] = {
                'loss': loss,
                'acc': acc,
                'precision': precision,
                'recall': recall
            }
            pred_ = min(1, sum(pred.cpu().numpy().astype(int)))
            if pred_ == 0:
                # print('Test {}'.format(i))
                predicted.append(0)
                results['res_of_each_test']['test_{}'.format(i)]['has_target'] = int(sample[0])
                # results['res_of_each_test']['test_{}'.format(i)]['has_target'] = 0
                results['res_of_each_test']['test_{}'.format(i)]['predict'] = 0
            else:
                predicted.append(1)
                results['res_of_each_test']['test_{}'.format(i)]['has_target'] = int(sample[0])
                # results['res_of_each_test']['test_{}'.format(i)]['has_target'] = 0
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
    target = [int(i) for i in args.target.split('-')]
    run(args=args, target=target, device=device)
