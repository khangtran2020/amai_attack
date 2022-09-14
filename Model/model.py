import os

import torch.nn as nn
import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict, defaultdict
from Bound.evaluate import evaluate
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from Utils.utils import tpr_tnr


class Classifier(nn.Module):
    def __init__(self, args, n_inputs, n_outputs):
        super(Classifier, self).__init__()
        self.kernel_size = args.embed_dim
        self.fc1 = nn.Linear(n_inputs, self.kernel_size)
        self.fc2 = nn.Linear(self.kernel_size, n_outputs)

    def forward(self, x):
        fc1 = self.fc1(x)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        prob = torch.softmax(fc2, dim=1)
        return fc1, fc2, prob


# (args=args, device=device, data=data_loader, model=model)

def train(args, target, device, data, model):
    train_dataloader, valid_dataloader = data
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    results = defaultdict(list)
    if args.num_target == 1:
        SAVE_NAME = 'CELEBA_single_Laplace_eps_{}'.format(args.epsilon)
    else:
        SAVE_NAME = 'CELEBA_multiple_{}_Lap_eps_{}'.format(args.num_target, args.epsilon)

    custom_weight = np.array([1600.0, 200.0])
    criteria = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    model.to(device)
    max_correct = 0
    max_tpr = 0.0
    max_tnr = 0.0
    noise_scale = args.sens * args.num_feature / args.epsilon
    print('Start training process with {} epochs'.format(args.num_steps))
    x_train, y_train, imgs_train = next(iter(train_dataloader))
    temp_x = x_train.numpy()
    over_samp = np.tile(np.expand_dims(temp_x[0, :], 0), (args.over_samp, 1))
    org_temp_x = np.concatenate((over_samp, temp_x), axis=0)
    noise = np.random.laplace(0, noise_scale, org_temp_x[args.over_samp:args.train_multiplier].shape)
    print('L2 norm of noise: {}', np.linalg.norm(noise, ord=2))
    temp_x = org_temp_x.copy()
    temp_x[args.over_samp:args.train_multiplier] = temp_x[args.over_samp:args.train_multiplier] + noise
    print('L2 distance: {}', np.linalg.norm(temp_x - org_temp_x, ord=2))
    return
    x_train = torch.from_numpy(temp_x)
    x_train = x_train.to(device)
    temp_x = y_train.numpy()
    temp_x = np.concatenate((np.zeros(args.over_samp), temp_x), axis=0).astype(int)
    y_train = torch.from_numpy(temp_x)
    y_train = y_train.to(device)
    x_valid, y_valid, _ = next(iter(valid_dataloader))
    temp_x = x_valid.numpy()
    temp_x[1:args.valid_multiplier] = temp_x[1:args.valid_multiplier] + np.random.laplace(0,
                                                                                          args.sens * args.num_feature / args.epsilon,
                                                                                          temp_x[
                                                                                          1:args.valid_multiplier].shape)
    x_valid = torch.from_numpy(temp_x)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    print(torch.bincount(y_train), torch.bincount(y_valid))
    for step in range(args.num_steps):
        num_correct = 0
        num_samples = 0
        loss_value = 0
        model.train()
        fc1, fc2, probs = model(x_train)

        loss = criteria(probs, y_train)
        loss_value += loss
        predictions = fc2[:, 0] < 0
        tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)

        loss.backward()
        optimizer.step()  # make the updates for each parameter
        optimizer.zero_grad()  # a clean up step for PyTorch

        # Test acc
        fc1, fc2, probs = model(x_valid)
        predictions = fc2[:, 0] < 0
        tpr, tnr, acc = tpr_tnr(predictions, y_valid)
        if (tpr + tnr) / 2 > max_tpr:
            state = {
                'net': model.state_dict(),
                'test': (tpr, tnr),
                'train': (tpr_train, tnr_train),
                'acc': acc,
                'lr': args.lr
            }

            max_tpr = (tpr + tnr) / 2
            torch.save(state, SAVE_NAME)
        if step % 10 == 0:
            # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
            print(
                f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {step}')

    return model
    # print('Finish one step in ', time.time() - start_time)

# (args=args, device=device, data=data_loader, model=model)
# def train_batches(args, target, device, data, model):
#     best_step = -1
#     best_acc = -1
#     data_name = sorted(os.listdir(args.data_path))
#     list_target = []
#     list_target_label = []
#     for i, f in enumerate(target):
#         list_target.append(torch.unsqueeze(torch.load(args.data_path + data_name[f]), 0))
#         list_target_label.append(1)
#     list_target = tuple(list_target)
#     target_data = torch.cat(list_target, 0)
#     target_label = torch.from_numpy(np.array(list_target_label))
#     # target_data = target_data.repeat(args.batch_size, 1)
#     # target_label = target_label.repeat(args.batch_size)
#     train_dataloader, valid_dataloader = data
#     optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
#     results = defaultdict(list)
#     if args.num_target == 1:
#         SAVE_NAME = 'CELEBA_single_Laplace_eps_{}'.format(args.epsilon)
#     else:
#         SAVE_NAME = 'CELEBA_multiple_{}_Lap_eps_{}'.format(args.num_target, args.epsilon)
#
#     custom_weight = np.array([1, args.batch_size])
#     criteria = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))
#     # criteria = nn.CrossEntropyLoss()
#     model.to(device)
#     print('Start training process with {} epochs'.format(args.num_steps))
#     for step in range(args.num_steps):
#         model.train()
#         train_loss = 0.0
#         train_label = []
#         train_predict = []
#         train_predict_prob = []
#         for i, batch_data in enumerate(train_dataloader):
#             x, y, imgs = batch_data
#             x = torch.cat((target_data,x), 0)
#             y = torch.cat((target_label,y),0)
#             optimizer.zero_grad()
#             num_data_point = x.size(dim=1)
#             x = x.repeat(args.train_multiplier, 1)
#             y = y.repeat(args.train_multiplier)
#             train_label = train_label + y.numpy().tolist()
#             # print(x.size(), y.size())
#             if args.train_mode == 'target':
#                 temp_x = x.numpy()
#                 temp_x[num_data_point:] = temp_x[num_data_point:] + np.random.laplace(0,
#                                                                                       args.sens * args.num_feature / args.epsilon,
#                                                                                       temp_x[num_data_point:].shape)
#                 x = torch.from_numpy(temp_x)
#             x = x.to(device)
#             y = y.to(device)
#             out, fc2 = model(x)
#             loss = criteria(out, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             y_pred = fc2[:, 0] > 0
#             y_prob = out[:,1]
#             train_predict = train_predict + y_pred.cpu().detach().numpy().tolist()
#             train_predict_prob = train_predict_prob + y_prob.cpu().detach().numpy().tolist()
#
#         if step % 1 == 0:
#             model.eval()
#             train_acc = accuracy_score(train_label, train_predict)
#             if args.num_target > 2:
#                 train_f1 = f1_score(train_label, train_predict, average='micro')
#             else:
#                 train_f1 = f1_score(train_label, train_predict, average='binary')
#             train_auc = average_precision_score(train_label, train_predict_prob)
#             valid_loss = 0
#             valid_label = []
#             valid_predict = []
#             valid_predict_prob = []
#             for i, batch_data in enumerate(valid_dataloader):
#                 x, y, imgs = batch_data
#                 x = torch.cat((target_data, x), 0)
#                 y = torch.cat((target_label, y), 0)
#                 optimizer.zero_grad()
#                 num_data_point = x.size(dim=1)
#                 x = x.repeat(args.valid_multiplier, 1)
#                 y = y.repeat(args.valid_multiplier)
#                 valid_label = valid_label + y.numpy().tolist()
#                 if args.train_mode == 'target':
#                     temp_x = x.numpy()
#                     temp_x[num_data_point:] = temp_x[num_data_point:] + np.random.laplace(0,
#                                                                                           args.sens * args.num_feature / args.epsilon,
#                                                                                           temp_x[num_data_point:].shape)
#                     x = torch.from_numpy(temp_x)
#                 x = x.to(device)
#                 y = y.to(device)
#                 out, fc2 = model(x)
#                 loss = criteria(out, y)
#                 valid_loss += loss.item()
#                 y_pred = fc2[:, 0] > 0
#                 y_prob = out[:,1]
#                 valid_predict = valid_predict + y_pred.cpu().detach().numpy().tolist()
#                 valid_predict_prob = valid_predict_prob + y_prob.cpu().detach().numpy().tolist()
#
#             valid_acc = accuracy_score(valid_label, valid_predict)
#             if args.num_target > 2:
#                 valid_f1 = f1_score(valid_label, valid_predict, average='micro')
#             else:
#                 valid_f1 = f1_score(valid_label, valid_predict, average='binary')
#             valid_auc = average_precision_score(valid_label, valid_predict_prob)
#             logging.info(
#                 f"\nStep: {step + 1}, AVG TRAIN Loss: {train_loss}, Acc: {train_acc}, F1: {train_f1}, AUC: {train_auc}| AVG VALID Loss: {valid_loss}, Acc: {valid_acc}, F1: {valid_f1}, AUC: {valid_auc}")
#
#             results['valid_loss'].append(valid_loss)
#             results['valid_acc'].append(valid_acc)
#             results['valid_f1'].append(valid_f1)
#
#             if best_acc < valid_acc:
#                 best_acc = valid_acc
#                 best_step = step
#             my_csv = pd.DataFrame(results)
#             name_save = args.save_path + SAVE_NAME + '.csv'
#             my_csv.to_csv(name_save, index=False)
#             with open(args.save_path + SAVE_NAME + '.pt', 'wb') as f:  # bd0.5_cr0_double bd0.1_cr2
#                 torch.save(model, f)
#     return model
#     # print('Finish one step in ', time.time() - start_time)
