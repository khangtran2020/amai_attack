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

class ClassifierTriplet(nn.Module):
    def __init__(self, args, n_inputs, n_outputs):
        super(ClassifierTriplet, self).__init__()
        self.kernel_size = args.embed_dim
        self.fc1 = nn.Linear(n_inputs, self.kernel_size)
        self.fc2 = nn.Linear(self.kernel_size, self.kernel_size)
        self.fc3 = nn.Linear(self.kernel_size, n_outputs)

    def forward(self, x):
        fc1 = self.fc1(x)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = F.relu(fc2)
        fc3 = self.fc3(fc1)
        prob = torch.softmax(fc3, dim=1)
        return fc2, fc3, prob


# (args=args, device=device, data=data_loader, model=model)

def train(args, target, device, data, model):
    train_dataloader, valid_dataloader = data
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    results = defaultdict(list)
    if args.debug:
        SAVE_NAME = 'debugging_eps_{}'.format(args.epsilon)
    else:
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
    noise_scale = args.sens / args.epsilon
    print('Start training process with {} epochs'.format(args.num_steps))
    x_train, y_train, imgs_train = next(iter(train_dataloader))
    temp_x = x_train.numpy()
    over_samp = np.tile(np.expand_dims(temp_x[0, :], 0), (args.over_samp, 1))
    org_temp_x = np.concatenate((over_samp, temp_x), axis=0)
    noise = np.random.laplace(0, noise_scale, temp_x[1:args.train_multiplier].shape)
    temp_x = org_temp_x.copy()
    temp_x[args.over_samp:args.train_multiplier] = temp_x[args.over_samp:args.train_multiplier] + noise
    print('L2 distance:', np.linalg.norm(temp_x - org_temp_x, ord=2))
    # return
    x_train = torch.from_numpy(temp_x)
    x_train = x_train.to(device)
    temp_x = y_train.numpy()
    temp_x = np.concatenate((np.zeros(args.over_samp), temp_x), axis=0).astype(int)
    temp_x = (1 - temp_x).astype(int)
    y_train = torch.from_numpy(temp_x)
    y_train = y_train.to(device)
    x_valid, y_valid, _ = next(iter(valid_dataloader))
    temp_x = x_valid.numpy()
    noise = np.random.laplace(0, noise_scale, temp_x[1:args.valid_multiplier].shape)
    temp_x[1:args.valid_multiplier] = temp_x[1:args.valid_multiplier] + noise
    x_valid = torch.from_numpy(temp_x)
    x_valid = x_valid.to(device)
    y_valid = 1 - y_valid
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

def train_triplet(args, target, device, data, model):
    train_dataloader, valid_dataloader = data
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
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
    noise_scale = args.sens / args.epsilon
    print('Start training process with {} epochs'.format(args.num_steps))
    img1, img2, img3, y_train, imgs_train = next(iter(train_dataloader))
    img1 = img1.to(device)
    img2 = img2.to(device)
    img3 = img3.to(device)
    y_train = 1 - y_train
    y_train = y_train.to(device)
    x_valid, y_valid, imgs_valid = next(iter(valid_dataloader))
    temp_x = x_valid.numpy()
    noise = np.random.laplace(0, noise_scale, temp_x[1:args.valid_multiplier].shape)
    temp_x[1:args.valid_multiplier] = temp_x[1:args.valid_multiplier] + noise
    x_valid = torch.from_numpy(temp_x.astype(np.float32))
    x_valid = x_valid.to(device)
    y_valid = 1 - y_valid
    y_valid = y_valid.to(device)
    print(torch.bincount(y_train), torch.bincount(y_valid))
    for step in range(args.num_steps):
        num_correct = 0
        num_samples = 0
        loss_value = 0
        model.train()
        f1, pre1, probs1 = model(img1)
        f2, pre2, probs2 = model(img2)
        f3, pre3, probs3 = model(img3)
        del (pre2)
        del (pre3)
        del (probs2)
        del (probs3)
        loss = criteria(probs1, y_train) + args.reg*triplet_loss(f1, f2, f3)
        # loss = triplet_loss(f1, f2, f3)
        loss_value += loss
        predictions = pre1[:, 0] < 0
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
