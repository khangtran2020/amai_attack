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
import sklearn.utils.class_weight


class Classifier(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Classifier, self).__init__()
        self.kernel_size = 200
        self.fc1 = nn.Linear(n_inputs, self.kernel_size)
        # self.fc15 = nn.Linear(10000, 7000)
        # self.fc16 = nn.Linear(7000, 4000)
        # self.fc17 = nn.Linear(4000, 12000)
        self.fc2 = nn.Linear(self.kernel_size, n_outputs)
        # self.fc3 = nn.Linear(200, 2)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc15(x)
        # x = F.relu(x)
        # x = self.fc16(x)
        # x = F.relu(x)
        # x = self.fc17(x)
        # x = F.relu(x)
        fc2 = self.fc2(x)
        # x = F.relu(x)
        # fc3 = self.fc3(x)
        x = torch.sigmoid(fc2)
        probs = F.softmax(x, dim=1)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x, probs, fc2


# (args=args, device=device, data=data_loader, model=model)
def train(args, device, data, model):
    train_dataloader, valid_dataloader = data
    ##################
    # init optimizer #
    ##################
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    print('steps', args.num_steps)
    print('save_path', args.save_path)

    results = defaultdict(list)
    if args.num_target == 1:
        SAVE_NAME = 'CELEBA_embed_BitRand_single_{}_{}'.format(args.num_target, args.epsilon)
    else:
        SAVE_NAME = 'CELEBA_embed_BitRand_multiple_{}_{}'.format(args.num_target, args.epsilon)

    x_train, y_train, imgs_train = next(iter(train_dataloader))
    x_valid, y_valid, imgs_valid = next(iter(valid_dataloader))
    if (args.train_mode == 'dp'):
        print("Train with Laplace mechanism with epsilon = {}".format(args.epsilon))
        temp_x = x_train.numpy()
        temp_x[1:] = temp_x[1:] + np.random.laplace(0, args.sens * args.num_feature / args.epsilon,
                                                    temp_x[1:].shape)
        x_train = torch.from_numpy(temp_x)
        temp_x = x_valid.numpy()
        temp_x[1:] = temp_x[1:] + np.random.laplace(0, args.sens * args.num_feature / args.epsilon,
                                                    temp_x[1:].shape)
        x_valid = torch.from_numpy(temp_x)

    weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(args.num_target + 1),
                                                             y=y_valid.cpu().detach().numpy())
    custom_weight = np.array([1600.0, 200.0])
    criteria = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    model = model.to(device)
    for step in range(args.num_steps):
        start_time = time.time()
        model.train()
        loss_value = 0
        out, probs, fc2 = model(x_train)
        loss = criteria(out, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_value += loss

        if step % 10 == 0:
            last_eval = step
            res_train = evaluate(x=x_train, y=y_train, model=model, criteria=criteria)
            res_val = evaluate(x=x_valid, y=y_valid, model=model, criteria=criteria)
            logging.info(
                f"\nStep: {step + 1}, Train Loss: {res_train['loss']}, Acc: {res_train['acc']:.4f}, TPR: {res_train['tpr']:.4f}, TNR : {res_train['tnr']:.4f} | Val Loss: {res_val['loss']}, Acc: {res_val['acc']:.4f}, TPR: {res_val['tpr']:.4f}, TNR : {res_val['tnr']:.4f}")

            results['test_avg_loss'].append(res['loss'] / x_valid.size(dim=0))
            results['test_acc'].append(res['acc'])
            results['tpr'].append(res['tpr'])
            results['tnr'].append(res['tnr'])

            if best_acc < res['acc']:
                best_acc = res['acc']
                best_step = step
            my_csv = pd.DataFrame(results)
            name_save = args.save_path + SAVE_NAME + '.csv'
            my_csv.to_csv(name_save, index=False)
            with open(args.save_path + SAVE_NAME + '.pt', 'wb') as f:  # bd0.5_cr0_double bd0.1_cr2
                torch.save(model, f)

        # print('Finish one step in ', time.time() - start_time)
