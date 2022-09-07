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
    results = {}
    x_test, y_test, file_name = next(iter(data))

    # evaluate
    weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(args.num_target + 1),
                                                             y=y_test.cpu().detach().numpy())
    # custom_weight = np.array([1600.0, 200.0])
    criteria = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
    res_test = evaluate(x=x_test, y=y_test, model=model, criteria=criteria)
    results['loss'] = res_test['loss']
    results['acc'] = res_test['acc']
    results['tpr'] = res_test['tpr']
    results['tnr'] = res_test['tnr']
    print("Test y_test:",torch.bincount(y_test))
    print(x_test.size())
    print(file_name)
    # exit()
    x_test = x_test.to(device)
    num_of_test_point = len(x_test)
    out, probs, fc2 = model(x_test)
    original_prediction = fc2[:, 0].cpu().numpy()
    print("Original shape",original_prediction.shape)
    if args.eval_mode == 'eps':
        epsilon_of_point = args.max_epsilon
        certified = 0
        # x_test = x_test.repeat(args.num_draws, 1)
        for eps in tqdm(np.linspace(args.min_epsilon, args.max_epsilon, 100)):
            temp_x = x_test.cpu().numpy()
            temp_x[1:args.num_draws] = temp_x[1:args.num_draws] + np.random.laplace(0, args.sens * args.num_feature / eps,
                                                temp_x[1:args.num_draws].shape)
            temp_x = torch.from_numpy(temp_x[:args.num_draws].astype(np.float32)).to(device)
            out, probs, fc2 = model(temp_x)
            pred = fc2[:, 0].cpu().numpy()
            print(pred.shape)
            count_of_sign = np.zeros(shape=(1,2))
            print("Start drawing")
            # for t in range(args.num_draws):
            same_sign = (pred*original_prediction[:args.num_draws]) > 0
            count_of_sign[0,0] += np.sum(np.logical_not(same_sign).astype('int8'))
            count_of_sign[0,1] += np.sum(same_sign.astype('int8'))
            upper_bound = hoeffding_upper_bound(count_of_sign[0,0],nobs=args.num_draws,alpha=args.alpha)
            lower_bound = hoeffding_lower_bound(count_of_sign[0,1], nobs=args.num_draws, alpha=args.alpha)
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
        print("Start drawing")
        for t in range(args.num_draws):
            same_sign = (pred[t * num_of_test_point:(t + 1) * num_of_test_point] * original_prediction[
                                                                                   t * num_of_test_point:(
                                                                                                                 t + 1) * num_of_test_point]) > 0
            count_of_sign[:, 0] += np.logical_not(same_sign).astype('int8')
            count_of_sign[:, 1] += same_sign.astype('int8')
        print("Done drawing")
        for alp in tqdm(np.linspace(1e-4, 1.0, 100)):
            upper_bound = hoeffding_upper_bound(count_of_sign[:, 0], nobs=args.num_draws, alpha=alp)
            lower_bound = hoeffding_lower_bound(count_of_sign[:, 1], nobs=args.num_draws, alpha=alp)
            index = np.where(lower_bound > upper_bound)
            alpha_of_point[index] = max(1-alp, alpha_of_point[index])
            certified[index] = 1
        results = dict(zip(file_name, zip(certified, alpha_of_point)))
    return results

# def user_dp_draw_noise(args, hnet):
#
#
#     pass