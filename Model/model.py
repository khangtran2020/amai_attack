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
from sklearn.metrics import accuracy_score, f1_score


class Classifier(nn.Module):
    def __init__(self, args, n_inputs, n_outputs):
        super(Classifier, self).__init__()
        self.kernel_size = args.embed_dim
        self.fc1 = nn.Linear(n_inputs, self.kernel_size)
        self.fc2 = nn.Linear(self.kernel_size, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        fc2 = self.fc2(x)
        x = torch.softmax(fc2, dim=1)
        return x, fc2


# (args=args, device=device, data=data_loader, model=model)
def train(args, target, device, data, model):
    best_step = -1
    best_acc = -1
    data_name = sorted(os.listdir(args.data_path))
    list_target = []
    list_target_label = []
    for i, f in enumerate(target):
        list_target.append(torch.unsqueeze(torch.load(args.data_path + data_name[f]), 0))
        list_target_label.append(i)
    list_target = tuple(list_target)
    target_data = torch.cat(list_target, 0)
    target_label = torch.from_numpy(np.array(list_target_label))
    train_dataloader, valid_dataloader = data
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    results = defaultdict(list)
    if args.num_target == 1:
        SAVE_NAME = 'CELEBA_single_Laplace_eps_{}'.format(args.epsilon)
    else:
        SAVE_NAME = 'CELEBA_multiple_{}_Lap_eps_{}'.format(args.num_target, args.epsilon)

    criteria = nn.CrossEntropyLoss()
    model.to(device)
    print('Start training process with {} epochs'.format(args.num_steps))
    for step in range(args.num_steps):
        model.train()
        train_loss = 0.0
        train_label = []
        train_predict = []
        for i, batch_data in enumerate(train_dataloader):
            x, y, imgs = batch_data
            x = torch.cat((target_data,x), 0)
            y = torch.cat((target_label,y),0)
            train_label = train_label + y.numpy().tolist()
            optimizer.zero_grad()
            num_data_point = x.size(dim=1)
            x = x.repeat(args.train_multiplier, 1)
            y = y.repeat(args.train_multiplier)
            # print(x.size(), y.size())
            if args.train_mode == 'target':
                temp_x = x.numpy()
                temp_x[num_data_point:] = temp_x[num_data_point:] + np.random.laplace(0,
                                                                                      args.sens * args.num_feature / args.epsilon,
                                                                                      temp_x[num_data_point:].shape)
                x = torch.from_numpy(temp_x)
            x = x.to(device)
            y = y.to(device)
            out, fc2 = model(x)
            loss = criteria(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred = fc2[:, 0] < 0
            train_predict = train_predict + y_pred.cpu().detach().numpy().tolist()

        if step % 10 == 0:
            model.eval()
            train_acc = accuracy_score(train_label, train_predict)
            if args.num_target > 2:
                train_f1 = f1_score(train_label, train_predict, average='micro')
            else:
                train_f1 = f1_score(train_label, train_predict, average='binary')
            valid_loss = 0
            valid_label = []
            valid_predict = []
            for i, batch_data in enumerate(valid_dataloader):
                x, y, imgs = batch_data
                x = torch.cat((target_data, x), 0)
                y = torch.cat((target_label, y), 0)
                valid_label = valid_label + y.numpy().tolist()
                optimizer.zero_grad()
                num_data_point = x.size(dim=1)
                x = x.repeat(args.valid_multiplier, 1)
                y = y.repeat(args.valid_multiplier)
                if args.train_mode == 'target':
                    temp_x = x.numpy()
                    temp_x[num_data_point:] = temp_x[num_data_point:] + np.random.laplace(0,
                                                                                          args.sens * args.num_feature / args.epsilon,
                                                                                          temp_x[num_data_point:].shape)
                    x = torch.from_numpy(temp_x)
                x = x.to(device)
                y = y.to(device)
                out, fc2 = model(x)
                loss = criteria(out, y)
                valid_loss += loss.item()
                y_pred = fc2[:, 0] < 0
                valid_predict = valid_predict + y_pred.cpu().detach().numpy().tolist()

            valid_acc = accuracy_score(valid_label, valid_predict)
            if args.num_target > 2:
                valid_f1 = f1_score(valid_label, valid_predict, average='micro')
            else:
                valid_f1 = f1_score(valid_label, valid_predict, average='binary')

            logging.info(
                f"\nStep: {step + 1}, TRAIN Loss: {train_loss}, Acc: {train_acc}, F1: {train_f1}| VALID Loss: {valid_loss}, Acc: {valid_acc}, F1: {valid_f1}")

            results['valid_loss'].append(valid_loss)
            results['valid_acc'].append(valid_acc)
            results['valid_f1'].append(valid_f1)

            if best_acc < results['valid_acc']:
                best_acc = results['valid_acc']
                best_step = step
            my_csv = pd.DataFrame(results)
            name_save = args.save_path + SAVE_NAME + '.csv'
            my_csv.to_csv(name_save, index=False)
            with open(args.save_path + SAVE_NAME + '.pt', 'wb') as f:  # bd0.5_cr0_double bd0.1_cr2
                torch.save(model, f)
    return model
    # print('Finish one step in ', time.time() - start_time)
