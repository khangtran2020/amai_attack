import os
import sys
path = "/".join([x for x in os.path.realpath(__file__).split('/')[:-2]])
sys.path.insert(0, path)
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.utils.data
from tqdm import trange
import time
from Model.model import *
from Utils.utils import *
from copy import deepcopy
from Bound.robustness import *

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
    x_test, y_test, _ = next(iter(data))
    x_test = x_test.to(device)
    noise = args.sens * args.num_feature / args.epsilon
    num_of_test_point = len(x_test)
    out, probs, fc2 = model(x_test)
    print(type(fc2), fc2.size())
    return
    original_prediction = fc2[:, 0] < 0
    x = x.numpy()
    x = np.repeat(x, args.num_draws, axis=0)
    x = torch.from_numpy(x)
    x = x.to(device)
    out, probs, fc2 = model(x)
    pred = fc2[:, 0] < 0
    results = defaultdict(lambda: defaultdict(list))
    robust_result = {}
    running_loss, running_correct_from_logits, running_correct_from_argmax, running_samples = 0., 0., 0., 0.
    data = {
        'argmax_sum': [],
        'pred_truth_argmax': [],
        'total_prediction': 0,
        'correct_prediction_argmax': 0,
    }
    prediction_votes = np.zeros([num_of_test_point, args.num_label])
    softmax_sum = np.zeros([num_of_test_point, args.num_label])
    softmax_sqr_sum = np.zeros([num_of_test_point, args.num_label])
    for draw in range(args.num_draws):
        draw_state = create_state_dict_at_one_draw(hnet=hnet, index=draw, dict_of_state=noisy_model)
        hnet.load_state_dict(draw_state)
        weights, _ = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
        net.load_state_dict(weights)
        pred = net(img)
        argmax_pred = pred.argmax(1)
        for j in range(num_of_test_point):
            prediction_votes[j, argmax_pred[j].item()] += 1
            softmax_sum[j] += pred[j].cpu().numpy()
            softmax_sqr_sum[j] += pred[j].cpu().numpy() ** 2
    predictions = np.argmax(prediction_votes, axis=1)
    predictions_logits = np.argmax(softmax_sum, axis=1)
    truth = label.cpu().detach().numpy()
    predictions_logit = torch.from_numpy(softmax_sum / args.num_draws_udp).to(device)
    predictions_hard = torch.from_numpy(predictions).to(device)

    running_loss += criteria(predictions_logit, label).item()
    running_correct_from_logits += predictions_logit.argmax(1).eq(label).sum().item()
    running_correct_from_argmax += predictions_hard.eq(label).sum().item()
    running_samples += len(label)

    results[node_id]['loss'] = running_loss / (batch_count + 1)
    results[node_id]['correct'] = running_correct_from_logits
    results[node_id]['total'] = running_samples
    data['argmax_sum'] += prediction_votes.tolist()
    data['softmax_sum'] += softmax_sum.tolist()
    data['softmax_sqr_sum'] += softmax_sqr_sum.tolist()
    data['pred_truth_argmax'] += (truth == predictions).tolist()
    data['pred_truth_softmax'] += (truth == predictions_logits).tolist()

    print("From argamx: {} / {}".format(np.sum(truth == predictions), len(predictions)))
    print("From logits: {} / {}".format(np.sum(truth == predictions_logits), len(predictions)))

    robustness_from_argmax = [robustness_size_argmax(
        counts=x,
        eta=args.robustness_confidence_proba,
        dp_attack_size=args.attack_norm_bound,
        dp_epsilon=args.udp_epsilon,
        dp_delta=args.udp_delta,
        dp_mechanism='userdp'
    ) for x in data['argmax_sum']]
    data['robustness_from_argmax'] = robustness_from_argmax
    robustness_from_softmax = [robustness_size_softmax(
        args=args,
        tot_sum=data['softmax_sum'][i],
        sqr_sum=data['softmax_sqr_sum'][i],
        counts=data['argmax_sum'][i],
        eta=args.robustness_confidence_proba,
        dp_attack_size=args.attack_norm_bound,
        dp_epsilon=args.udp_epsilon,
        dp_delta=args.udp_delta,
        dp_mechanism='userdp'
    ) for i in range(len(data['argmax_sum']))]
    data['robustness_from_softmax'] = robustness_from_softmax
    data['total_prediction'] = results[node_id]['total']
    data['correct_prediction_logits'] = running_correct_from_logits
    data['correct_prediction_argmax'] = running_correct_from_argmax
    robust_result[node_id] = data
    return robust_result

# def user_dp_draw_noise(args, hnet):
#
#
#     pass