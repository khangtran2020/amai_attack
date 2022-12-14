import gc
import os
import sys

path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import torch.utils.data
from Model.model import *
from Utils.utils import *
from tqdm import tqdm
import sklearn
from Bound.robustness import hoeffding_lower_bound, hoeffding_upper_bound
from Data.celeba import CelebATriplet, CelebATripletFull


# def eval_model(nodes, num_nodes, hnet, net, criteria, device, split):
#     curr_results = evaluate(nodes, num_nodes, hnet, net, criteria, device, split=split)
#     total_correct = sum([val['correct'] for val in curr_results.values()])
#     total_samples = sum([val['total'] for val in curr_results.values()])
#     avg_loss = np.mean([val['loss'] for val in curr_results.values()])
#     avg_acc = total_correct / total_samples
#
#     all_acc = [val['correct'] / val['total'] for val in curr_results.values()]
#
#     return curr_results, avg_loss, avg_acc, all_acc


@torch.no_grad()
def evaluate(x, y, model, criteria, device):
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    model.eval()
    results = defaultdict(lambda: defaultdict(list))
    running_loss, running_correct, running_samples = 0., 0., 0.
    out, probs, fc2 = model(x)
    loss = criteria(out, y).item()
    pred = fc2[:, 0] < 0
    # print(pred)
    tpr, tnr, acc = tpr_tnr(pred, y)
    results['loss'] = loss
    results['acc'] = acc
    results['tpr'] = tpr
    results['tnr'] = tnr
    return results


@torch.no_grad()
def evaluate_robust(args, data, model, device='cpu'):
    model.to(device)
    model.eval()
    results = {}
    x_test, y_test, file_name = next(iter(data))
    custom_weight = np.array([1600.0, 200.0])
    criteria = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))
    res_test = evaluate(x=x_test, y=y_test, model=model, criteria=criteria)
    print(res_test)
    results['target_file_name'] = file_name[0]
    results['loss'] = res_test['loss']
    results['acc'] = res_test['acc']
    results['tpr'] = res_test['tpr']
    results['tnr'] = res_test['tnr']
    x_test = x_test.to(device)
    num_of_test_point = len(x_test)
    out, probs, fc2 = model(x_test)
    original_prediction = fc2[:, 0].cpu().numpy()
    print("Original shape", original_prediction.shape)
    if args.eval_mode == 'eps':
        epsilon_of_point = args.max_epsilon
        certified = 0
        # x_test = x_test.repeat(args.num_draws, 1)
        for eps in tqdm(np.linspace(args.min_epsilon, args.max_epsilon, 100)):
            temp_x = x_test.cpu().numpy()
            temp_x[1:args.num_draws] = temp_x[1:args.num_draws] + np.random.laplace(0,
                                                                                    args.sens * args.num_feature / eps,
                                                                                    temp_x[1:args.num_draws].shape)
            temp_x = torch.from_numpy(temp_x[:args.num_draws].astype(np.float32)).to(device)
            out, probs, fc2 = model(temp_x)
            pred = fc2[:, 0].cpu().numpy()
            print(pred.shape)
            count_of_sign = np.zeros(shape=(1, 2))
            print("Start drawing")
            # for t in range(args.num_draws):
            same_sign = (pred * original_prediction[:args.num_draws]) > 0
            count_of_sign[0, 0] += np.sum(np.logical_not(same_sign).astype('int8'))
            count_of_sign[0, 1] += np.sum(same_sign.astype('int8'))
            upper_bound = hoeffding_upper_bound(count_of_sign[0, 0], nobs=args.num_draws, alpha=args.alpha)
            lower_bound = hoeffding_lower_bound(count_of_sign[0, 1], nobs=args.num_draws, alpha=args.alpha)
            if (lower_bound > upper_bound):
                certified = int(1.0)
                epsilon_of_point = min(epsilon_of_point, eps)
            print("Done for eps: {}".format(eps))
        results['certified_for_target'] = {
            'certified': bool(certified),
            'eps_min': epsilon_of_point,
            'confidence': 1 - args.alpha
        }
    else:
        alpha_of_point = 1e-4
        certified = 0.0
        # x_test = x_test.repeat(args.num_draws, 1)
        temp_x = x_test.cpu().numpy()
        temp_x[1:args.num_draws] = temp_x[1:args.num_draws] + np.random.laplace(0,
                                                                                args.sens * args.num_feature / args.fix_epsilon,
                                                                                temp_x[1:args.num_draws].shape)
        temp_x = torch.from_numpy(temp_x[:args.num_draws].astype(np.float32)).to(device)
        out, probs, fc2 = model(temp_x)
        pred = fc2[:, 0].cpu().numpy()
        count_of_sign = np.zeros(shape=(1, 2))
        print("Start drawing")
        # for t in range(args.num_draws):
        same_sign = (pred * original_prediction[:args.num_draws]) > 0
        count_of_sign[0, 0] += np.sum(np.logical_not(same_sign).astype('int8'))
        count_of_sign[0, 1] += np.sum(same_sign.astype('int8'))
        print("Done drawing")
        for alp in tqdm(np.linspace(1e-4, 1.0, 100)):
            upper_bound = hoeffding_upper_bound(count_of_sign[0, 0], nobs=args.num_draws, alpha=alp)
            lower_bound = hoeffding_lower_bound(count_of_sign[0, 1], nobs=args.num_draws, alpha=alp)
            if lower_bound > upper_bound:
                alpha_of_point = max(1 - alp, alpha_of_point)
                certified = 1.0
        results['certified_for_target'] = {
            'certified': bool(certified),
            'eps_min': args.fix_epsilon,
            'confidence': 1 - alpha_of_point
        }
    return results


def evaluate_intrain(args, model, certified, target, target_data, target_label, eps_cert, device='cpu'):
    results = {}
    results['res_of_each_test'] = {}
    true_label = []
    predicted = []
    if certified:
        noise_scale = args.sens / eps_cert
        print("Noise scale fore the attack:", noise_scale)
    else:
        print("Didn't ceritfied")
        return
    for i in range(args.num_test_set):
        sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
        test_loader = torch.utils.data.DataLoader(
            CelebATriplet(args, target, None, args.data_path, 'test', imgroot=None, include_tar=sample[0]),
            shuffle=False,
            num_workers=0, batch_size=args.num_test_point)
        x_test, y_test, file_name = next(iter(test_loader))
        y_test = 1 - y_test
        true_label.append(sample[0])
        if sample[0]:
            x_test = torch.cat((target_data, x_test), 0)
            y_test = torch.cat((target_label, y_test), 0)
            temp_x = x_test.numpy()
            noise = np.random.laplace(0, noise_scale, temp_x[:args.num_target].shape)
            temp_x[:args.num_target] = temp_x[:args.num_target] + noise
            x_test = torch.from_numpy(temp_x.astype(np.float32))
        criteria = nn.CrossEntropyLoss()
        model.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        fc2, fc3, prob = model(x_test)
        loss = criteria(prob, y_test).item()
        pred = fc3[:, 0] > 0
        acc = accuracy_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
        precision = precision_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
        recall = recall_score(y_test.cpu().detach().numpy(), pred.cpu().numpy().astype(int))
        results['res_of_each_test']['test_{}'.format(i)] = {
            'loss': loss,
            'acc': acc,
            'precision': precision,
            'recall': recall
        }
        pred_ = sum(pred.cpu().numpy().astype(int))
        if pred_ == 0:
            # print('Test {}'.format(i))
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
    return results


def cert(args, model, target_data, device='cpu'):
    model.to(device)
    list_of_eps_can_cert = []
    for i, eps in enumerate(np.linspace(args.min_epsilon, args.max_epsilon, args.eps_step)):
        temp_x = target_data.numpy()
        noise_scale = args.sens / eps
        generated_target_org = np.tile(temp_x, (args.num_draws, 1))
        noise = np.random.laplace(0, noise_scale, generated_target_org[1:, :].shape)
        generated_target = generated_target_org.copy()
        generated_target[1:, :] = generated_target[1:, :] + noise
        temp_x = torch.from_numpy(generated_target.astype(np.float32)).to(device)
        fc2, fc3, prob = model(temp_x)
        pred = fc3[:, 1].cpu().detach().numpy()
        larger_than_zero = pred > 0
        count_of_larger_than_zero = sum(larger_than_zero.astype(int))
        count_of_smaller_than_zero = args.num_draws - count_of_larger_than_zero
        print(
            'For eps {:.2f}, # larger than 0: {}, # smaller or equal to 0: {}'.format(
                eps, count_of_larger_than_zero, count_of_smaller_than_zero))
        upper_bound = hoeffding_upper_bound(count_of_smaller_than_zero, nobs=args.num_draws, alpha=args.alpha)
        lower_bound = hoeffding_lower_bound(count_of_larger_than_zero, nobs=args.num_draws, alpha=args.alpha)
        if (lower_bound > upper_bound):
            list_of_eps_can_cert.append(eps)
    return list_of_eps_can_cert


def cert_2side(args, model, target_data, target, device='cpu'):
    model.to(device)
    list_of_eps_can_cert = []
    test_loader = torch.utils.data.DataLoader(
        CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='valid', task='cert', imgroot=None,
                          multiplier=args.valid_multiplier),
        shuffle=False,
        num_workers=0, batch_size=args.batch_size)
    x_test, y_test, file_name = next(iter(test_loader))
    del (y_test)
    del (file_name)
    gc.collect()
    noise_scale = args.sens / 1.0
    num_nontarget = x_test.size(dim=0) - args.valid_multiplier
    for i, eps in enumerate(np.linspace(args.min_epsilon, args.max_epsilon, args.eps_step)):
        noise_scale = args.sens / eps
        temp_x = x_test + torch.distributions.laplace.Laplace(loc=0, scale=noise_scale).rsample(x_test.size())
        temp_x = temp_x.to(device)
        fc2, fc3, prob = model(temp_x)
        pred = fc3[:, 1].cpu().detach().numpy()
        target_larger_than_zero = pred[:args.valid_multiplier] > 0
        count_of_target_larger_than_zero = sum(target_larger_than_zero.astype(int))
        count_of_target_smaller_than_zero = args.valid_multiplier - count_of_target_larger_than_zero
        non_target_larger_than_zero = pred[args.valid_multiplier:] > 0
        count_of_nontarget_larger_than_zero = sum(non_target_larger_than_zero.astype(int))
        count_of_nontarget_smaller_than_zero = num_nontarget - count_of_nontarget_larger_than_zero
        print(
            'For eps {:.2f}, Target - # larger than 0: {}, # smaller or equal to 0: {} | Non-target: - # larger than 0: {}, # smaller or equal to 0: {}'.format(
                eps, count_of_target_larger_than_zero, count_of_target_smaller_than_zero,
                count_of_nontarget_larger_than_zero, count_of_nontarget_smaller_than_zero))
        upper_bound_target = hoeffding_upper_bound(count_of_target_smaller_than_zero, nobs=args.valid_multiplier,
                                                   alpha=args.alpha, bonferroni_hyp_n=2)
        lower_bound_target = hoeffding_lower_bound(count_of_target_larger_than_zero, nobs=args.valid_multiplier,
                                                   alpha=args.alpha, bonferroni_hyp_n=2)
        upper_bound_non_target = hoeffding_upper_bound(count_of_nontarget_larger_than_zero, nobs=num_nontarget,
                                                       alpha=args.alpha, bonferroni_hyp_n=2)
        lower_bound_non_target = hoeffding_lower_bound(count_of_nontarget_smaller_than_zero, nobs=num_nontarget,
                                                       alpha=args.alpha, bonferroni_hyp_n=2)
        if (lower_bound_target > upper_bound_target) and (lower_bound_non_target > upper_bound_non_target):
            list_of_eps_can_cert.append(eps)
    return list_of_eps_can_cert


def perform_attack(args, results, target, target_data, target_label, list_of_eps, model, device):
    results['number_of_test_set'] = args.num_test_set
    results['sample_target_rate'] = args.sample_target_rate
    results['result_of_eps'] = {}
    model.to(device)
    for eps in tqdm(list_of_eps):
        true_label = []
        predicted = []
        for i in range(args.num_test_set):
            noise_scale = args.sens / eps
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            test_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='test', imgroot=None,
                                  include_tar=sample[0]),
                shuffle=False,
                num_workers=0, batch_size=args.num_test_point)
            x_test, y_test, file_name = next(iter(test_loader))
            y_test = 1 - y_test
            true_label.append(sample[0])
            if sample[0]:
                x_test = torch.cat((target_data, x_test), 0)
                y_test = torch.cat((target_label, y_test), 0)
            x_test = x_test + torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale).rsample(x_test.size())
            criteria = nn.CrossEntropyLoss()
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            fc2, fc3, prob = model(x_test)
            loss = criteria(prob, y_test).item()
            pred = fc3[:, 1] > 0
            print(pred, y_test)
            pred_ = sum(pred.cpu().numpy().astype(int))
            if pred_ == 0:
                predicted.append(0)
            else:
                predicted.append(1)
        acc = accuracy_score(true_label, predicted)
        precision = precision_score(true_label, predicted)
        recall = recall_score(true_label, predicted)
        results['result_of_eps']['Eps = {0:.3f}'.format(eps)] = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
        }
    return results


def perform_attack_parallel(arg, eps):
    args, results, target, target_data, target_label, model, device, logger = arg
    true_label = []
    predicted = []
    with timeit(logger, 'evaluating-test'):
        for i in range(args.num_test_set):
            noise_scale = args.sens / eps
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            test_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='test', imgroot=None,
                                  include_tar=sample[0]),
                shuffle=False,
                num_workers=0, batch_size=args.num_test_point)
            x_test, y_test, file_name = next(iter(test_loader))
            y_test = 1 - y_test
            true_label.append(sample[0])
            if sample[0]:
                x_test = torch.cat((target_data, x_test), 0)
                y_test = torch.cat((target_label, y_test), 0)
            x_test = x_test + torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale).rsample(x_test.size())
            criteria = nn.CrossEntropyLoss()
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            fc2, fc3, prob = model(x_test)
            loss = criteria(prob, y_test).item()
            pred = fc3[:, 1] > 0
            pred_ = sum(pred.cpu().numpy().astype(int))
            if pred_ == 0:
                predicted.append(0)
            else:
                predicted.append(1)
    acc = accuracy_score(true_label, predicted)
    precision = precision_score(true_label, predicted)
    recall = recall_score(true_label, predicted)
    results['result_of_eps']['Eps = {0:.3f}'.format(eps)] = {
        'acc': acc,
        'precision': precision,
        'recall': recall,
    }


def perform_attack_test(args, results, target, target_data, target_label, list_of_eps, model, device):
    results['number_of_test_set'] = args.num_test_set
    results['sample_target_rate'] = args.sample_target_rate
    results['result_of_eps'] = {}
    model.to(device)
    for eps in tqdm(list_of_eps):
        true_label = []
        predicted = []
        for i in range(args.num_test_set):
            noise_scale_target = args.sens / eps
            noise_scale_non_target = args.sens / (args.epsilon * 10)
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            test_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='test', imgroot=None,
                                  include_tar=sample[0]),
                shuffle=False,
                num_workers=0, batch_size=args.num_test_point)
            x_test, y_test, file_name = next(iter(test_loader))
            y_test = 1 - y_test
            true_label.append(sample[0])
            if sample[0]:
                x_test = torch.cat((target_data, x_test), 0)
                y_test = torch.cat((target_label, y_test), 0)
            noise_target = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale_target).rsample(
                x_test[:args.num_target].size())
            noise_non_target = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale_non_target).rsample(
                x_test[args.num_target:].size())
            x_test[:args.num_target] = x_test[:args.num_target] + noise_target
            x_test[args.num_target:] = x_test[args.num_target:] + noise_non_target
            criteria = nn.CrossEntropyLoss()
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            fc2, fc3, prob = model(x_test)
            loss = criteria(prob, y_test).item()
            pred = fc3[:, 1] > 0
            pred_ = sum(pred.cpu().numpy().astype(int))
            if pred_ == 0:
                predicted.append(0)
            else:
                predicted.append(1)
        acc = accuracy_score(true_label, predicted)
        precision = precision_score(true_label, predicted)
        recall = recall_score(true_label, predicted)
        results['result_of_eps']['Eps = {0:.3f}'.format(eps)] = {
            'acc': acc,
            'precision': precision,
            'recall': recall,
        }
    return results


def perform_attack_test_parallel(arg, eps):
    args, results, target, target_data, target_label, model, device, logger = arg
    true_label = []
    predicted = []
    with timeit(logger, 'evaluating-test'):
        for i in range(args.num_test_set):
            noise_scale_target = args.sens / eps
            noise_scale_non_target = args.sens / (args.epsilon * 10)
            print('Noise scale target: {:.2f} | Noise scale non-target: {:.2f}'.format(noise_scale_target,
                                                                                       noise_scale_non_target))
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            test_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='test', imgroot=None,
                                  include_tar=sample[0]),
                shuffle=False,
                num_workers=0, batch_size=args.num_test_point)
            x_test, y_test, file_name = next(iter(test_loader))
            y_test = 1 - y_test
            true_label.append(sample[0])
            if sample[0]:
                x_test = torch.cat((target_data, x_test), 0)
                y_test = torch.cat((target_label, y_test), 0)
                noise_target = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale_target).rsample(x_test[:args.num_target].size())
                noise_non_target = torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale_non_target).rsample(x_test[args.num_target:].size(),1)
                x_test[:args.num_target] = x_test[:args.num_target] + noise_target
                x_test[args.num_target:] = x_test[args.num_target:] + noise_non_target
            else:
                x_test = x_test + torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale_non_target).rsample(
                    x_test.size())
            criteria = nn.CrossEntropyLoss()
            # x_test = x_test.to(device)
            # y_test = y_test.to(device)
            fc2, fc3, prob = model(x_test)
            loss = criteria(prob, y_test).item()
            pred = fc3[:, 1] > 0
            pred_ = sum(pred.cpu().numpy().astype(int))
            if pred_ == 0:
                predicted.append(0)
            else:
                predicted.append(1)
    acc = accuracy_score(true_label, predicted)
    precision = precision_score(true_label, predicted)
    recall = recall_score(true_label, predicted)
    print('For eps = {:.2f}, acc: {}, precision: {}, recall {}'.format(eps, acc, precision, recall))
    results['result_of_eps']['Eps = {}'.format(eps)] = {
        'acc': acc,
        'precision': precision,
        'recall': recall,
    }
    return 'For eps = {:.2f}, acc: {}, precision: {}, recall {}'.format(eps, acc, precision, recall)


def perform_attack_test_parallel_same(arg, eps):
    args, results, target, target_data, target_label, model, device, logger = arg
    true_label = []
    predicted = []
    with timeit(logger, 'evaluating-test-same-cert-eps'):
        for i in range(args.num_test_set):
            noise_scale_target = args.sens / eps
            sample = np.random.binomial(n=1, p=args.sample_target_rate, size=1).astype(bool)
            test_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='test', imgroot=None,
                                  include_tar=sample[0]),
                shuffle=False,
                num_workers=0, batch_size=args.num_test_point)
            x_test, y_test, file_name = next(iter(test_loader))
            y_test = 1 - y_test
            true_label.append(sample[0])
            if sample[0]:
                x_test = torch.cat((target_data, x_test), 0)
                y_test = torch.cat((target_label, y_test), 0)
            x_test = x_test + torch.distributions.laplace.Laplace(loc=0.0, scale=noise_scale_target).rsample(x_test.size())
            criteria = nn.CrossEntropyLoss()
            fc2, fc3, prob = model(x_test)
            loss = criteria(prob, y_test).item()
            pred = fc3[:, 1] > 0
            pred_ = sum(pred.cpu().numpy().astype(int))
            if pred_ == 0:
                predicted.append(0)
            else:
                predicted.append(1)
    acc = accuracy_score(true_label, predicted)
    precision = precision_score(true_label, predicted)
    recall = recall_score(true_label, predicted)
    print('For eps = {:.2f}, acc: {}, precision: {}, recall {}'.format(eps, acc, precision, recall))
    results['result_of_eps']['Eps = {}'.format(eps)] = {
        'acc': acc,
        'precision': precision,
        'recall': recall,
    }
    return 'For eps = {:.2f}, acc: {}, precision: {}, recall {}'.format(eps, acc, precision, recall)
