import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
import torch.utils.data
from Model.model import *
from Utils.utils import *
from tqdm import tqdm
from Bound.robustness import hoeffding_lower_bound, hoeffding_upper_bound

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
def evaluate(x, y, model, criteria):
    model.eval()
    results = defaultdict(lambda: defaultdict(list))
    running_loss, running_correct, running_samples = 0., 0., 0.
    out, probs, fc2 = model(x)
    loss = criteria(out, y).item()
    pred = fc2[:, 0] < 0
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
    x_test, y_test, file_name = next(iter(data))
    x_test = x_test.to(device)
    noise = args.sens * args.num_feature / args.epsilon
    num_of_test_point = len(x_test)
    alpha_of_point = np.zeros(num_of_test_point) + 1e-5
    out, probs, fc2 = model(x_test)
    original_prediction = fc2[:, 0].numpy()
    if args.eval_mode == 'eps':
        epsilon_of_point = np.ones(num_of_test_point) * args.max_epsilon
        certified = np.zeros(num_of_test_point)
        for eps in tqdm(np.linspace(args.min_epsilon, args.max_epsilon, 100)):
            x_test = x_test.repeat(args.num_draws, 1)
            temp_x = x_test.numpy()
            temp_x = temp_x + np.random.laplace(0, args.sens * args.num_feature / args.epsilon,
                                                temp_x.shape)
            x_test = torch.from_numpy(temp_x)
            out, probs, fc2 = model(x_test)
            pred = fc2[:, 0].numpy()
            count_of_sign = np.zeros(shape=(num_of_test_point,2))
            for t in args.num_draws:
                same_sign = (pred[t*num_of_test_point:(t+1)*num_of_test_point]*original_prediction[t*num_of_test_point:(t+1)*num_of_test_point]) > 0
                count_of_sign[:,0] += np.logical_not(same_sign).astype('int8')
                count_of_sign[:,1] += same_sign.astype('int8')
            upper_bound = hoeffding_upper_bound(count_of_sign[:,0],nobs=args.num_draws,alpha=args.alpha)
            lower_bound = hoeffding_lower_bound(count_of_sign[:,1], nobs=args.num_draws, alpha=args.alpha)
            index = np.where(lower_bound > upper_bound)
            epsilon_of_point[index] = min(eps,epsilon_of_point[index])
            certified[index] = 1
        results = dict(zip(file_name, zip(certified,epsilon_of_point)))
    else:
        alpha_of_point = np.ones(num_of_test_point) * 1e-4
        certified = np.zeros(num_of_test_point)
        x_test = x_test.repeat(args.num_draws, 1)
        temp_x = x_test.numpy()
        temp_x = temp_x + np.random.laplace(0, args.sens * args.num_feature / args.fix_epsilon,
                                            temp_x.shape)
        x_test = torch.from_numpy(temp_x)
        out, probs, fc2 = model(x_test)
        pred = fc2[:, 0].numpy()
        count_of_sign = np.zeros(shape=(num_of_test_point, 2))
        for t in args.num_draws:
            same_sign = (pred[t * num_of_test_point:(t + 1) * num_of_test_point] * original_prediction[
                                                                                   t * num_of_test_point:(
                                                                                                                 t + 1) * num_of_test_point]) > 0
            count_of_sign[:, 0] += np.logical_not(same_sign).astype('int8')
            count_of_sign[:, 1] += same_sign.astype('int8')
        for alp in tqdm(np.linspace(1e-4, 1.0, 100)):
            upper_bound = hoeffding_upper_bound(count_of_sign[:, 0], nobs=args.num_draws, alpha=alp)
            lower_bound = hoeffding_lower_bound(count_of_sign[:, 1], nobs=args.num_draws, alpha=aalp)
            index = np.where(lower_bound > upper_bound)
            alpha_of_point[index] = max(1-alp, alpha_of_point[index])
            certified[index] = 1
        results = dict(zip(file_name, zip(certified, alpha_of_point)))
    return results

# def user_dp_draw_noise(args, hnet):
#
#
#     pass