from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
import random

class CelebA(Dataset):
    def __init__(self, args, target, transform, dataroot, mode='train', imgroot=None, include_tar=True,
                 shuffle=True, multiplier = 100):
        self.args = args
        self.target = target
        self.target_multiplier = multiplier
        self.num_file_org = len(os.listdir(dataroot))
        self.non_target = list(range(self.num_file_org))
        for i in self.target:
            self.non_target.remove(i)
        self.transform = transform
        self.include = include_tar
        if shuffle:
            random.shuffle(self.non_target)
        self.num_file = len(self.non_target)

        if mode == 'train':
            self.train_data = np.arange(int(0.6 * self.num_file_org))
            self.length = len(self.train_data)
        elif mode == 'valid':
            self.valid_data = np.arange(int(0.6 * self.num_file_org), int(0.8 * self.num_file_org))
            self.length = len(self.valid_data)
        else:
            test_point = np.random.choice(a=np.array(list(range(int(0.8 * self.num_file), self.num_file))),
                                          size=args.num_test_point, replace=False)
            self.test_data = test_point
            self.length = len(self.test_data)

        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                class_id = torch.tensor(len(self.target))
        elif self.mode == 'valid':
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.valid_data[idx]]
                class_id = torch.tensor(len(self.target))
        else:
            filename = self.data_name[self.non_target[self.test_data[idx]]]
            class_id = torch.tensor(len(self.target))

        img_tensor = torch.load(self.dataroot + filename)
        return img_tensor, class_id, filename




class AMIADatasetCelebA(Dataset):
    def __init__(self, args, target, transform, dataroot, mode='train', imgroot=None, multiplier=100):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        self.num_file = len(os.listdir(dataroot))
        self.train_data = np.arange(args.train_index)
        self.valid_data = np.arange(args.train_index, args.valid_index)
        self.test_data = np.arange(args.valid_index, self.num_file)

        if mode == 'train':
            self.train_data = np.arange(162770)
            mask = np.ones(162770, dtype=bool)
            self.train_data = self.train_data[mask, ...]
            self.length = len(self.train_data) + len(target) * multiplier
        elif mode == 'valid':
            self.valid_data = np.arange(162770, 182637)
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
            # print(type(self))
            self.test_data = np.arange(self.num_file - args.num_draws, self.num_file)
            self.length = len(target) * multiplier + len(self.test_data)
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))
        elif self.mode == 'valid':
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.valid_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))

        if self.imgroot:
            img = Image.open(self.imgroot + filename)
            img = self.transform(img)
        else:
            img = torch.tensor([])

        # img_tensor = img2vec.get_vec(img, tensor=True)
        # img_tensor = torch.squeeze(img_tensor)
        img_tensor = torch.load(self.dataroot + filename)

        # img_tensor = img_tensor + s1.astype(np.float32)

        return img_tensor, class_id, filename


class CelebATriplet(Dataset):
    def __init__(self, args, target, transform, dataroot, mode='train', imgroot=None, include_tar=True,
                 shuffle=True, multiplier = 100):
        self.args = args
        self.target = target
        self.target_multiplier = multiplier
        self.num_file_org = len(os.listdir(dataroot))
        self.non_target = list(range(self.num_file_org))
        for i in self.target:
            self.non_target.remove(i)
        self.transform = transform
        self.include = include_tar
        if shuffle:
            random.shuffle(self.non_target)
        self.num_file = len(self.non_target)
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.mode = mode
        self.noise_scale = args.sens / args.epsilon

        if mode == 'train':
            self.train_data = np.arange(int(0.6 * self.num_file))
            self.length = len(self.train_data) + len(target)*multiplier
        elif mode == 'valid':
            self.valid_data = np.arange(int(0.6 * self.num_file), int(0.8 * self.num_file))
            self.length = len(self.valid_data) + len(target)*multiplier
        else:
            test_point = np.random.choice(a=np.array(list(range(int(0.8 * self.num_file), self.num_file))),
                                          size=args.num_test_point, replace=False)
            self.test_data = test_point
            self.length = len(self.test_data)



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            if idx / self.target_multiplier < len(self.target):
                anchor_name = self.data_name[self.target[int(idx / self.target_multiplier)]]
                id2 = np.random.choice(a=np.arange(0, int(0.6*self.num_file)), size=1, replace=False)[0]
                negative_name = self.data_name[self.non_target[self.train_data[id2]]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
                anchor = torch.load(self.dataroot + anchor_name)
                negative = torch.load(self.dataroot + negative_name)
                temp_x = anchor.numpy()
                noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                temp_x = temp_x + noise
                positive = torch.from_numpy(temp_x.astype(np.float32))
                sample = np.random.binomial(1, self.args.sample_rate, 1)[0]
                if sample:
                    return anchor, positive, negative, class_id, anchor_name
                else:
                    temp_x = anchor.numpy()
                    noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                    temp_x = temp_x + noise
                    anchor = torch.from_numpy(temp_x.astype(np.float32))
                    return anchor, positive, negative, class_id, anchor_name
            else:
                idx -= len(self.target) * self.target_multiplier
                anchor_name = self.data_name[self.non_target[self.train_data[idx]]]
                id2 = np.random.choice(a=np.arange(0, int(0.6 * self.num_file)), size=1, replace=False)[0]
                positive_name = self.data_name[self.non_target[self.train_data[id2]]]
                negative_name = self.data_name[self.target[0]]
                class_id = torch.tensor(len(self.target))
                anchor = torch.load(self.dataroot + anchor_name)
                positive = torch.load(self.dataroot + positive_name)
                negative = torch.load(self.dataroot + negative_name)
                sample = np.random.binomial(1, 0.5, 1)[0]
                if sample:
                    return anchor, positive, negative, class_id, anchor_name
                else:
                    temp_x = negative.numpy()
                    noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                    temp_x = temp_x + noise
                    negative = torch.from_numpy(temp_x.astype(np.float32))
                    return anchor, positive, negative, class_id, anchor_name
        elif self.mode == 'valid':
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.non_target[self.valid_data[idx]]]
                class_id = torch.tensor(len(self.target))
        else:
            filename = self.data_name[self.non_target[self.test_data[idx]]]
            class_id = torch.tensor(len(self.target))
        img_tensor = torch.load(self.dataroot + filename)
        return img_tensor, class_id, filename


class CelebATripletFull(Dataset):
    def __init__(self, args, target, dataroot, mode='train', imgroot=None, include_tar=True,
                 shuffle=True, multiplier = 100):
        self.args = args
        self.target = target
        self.target_multiplier = multiplier
        self.num_file_org = len(os.listdir(dataroot))
        self.non_target = list(range(self.num_file_org))
        for i in self.target:
            self.non_target.remove(i)
        self.include = include_tar
        if shuffle:
            random.shuffle(self.non_target)
        self.num_file = len(self.non_target)
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.mode = mode
        self.noise_scale = args.sens / args.epsilon

        if mode == 'train':
            self.train_data = np.arange(int(0.6 * self.num_file))
            self.length = len(self.train_data) + len(target)*multiplier
        elif mode == 'valid':
            self.valid_data = np.arange(int(0.6 * self.num_file), int(0.8 * self.num_file))
            self.length = len(self.valid_data) + len(target)*multiplier
        else:
            test_point = np.random.choice(a=np.array(list(range(int(0.8 * self.num_file), self.num_file))),
                                          size=args.num_test_point, replace=False)
            self.test_data = test_point
            self.length = len(self.test_data)



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            if idx / self.target_multiplier < len(self.target):
                anchor_name = self.data_name[self.target[int(idx / self.target_multiplier)]]
                id2 = np.random.choice(a=np.arange(0, int(0.6*self.num_file)), size=1, replace=False)[0]
                negative_name = self.data_name[self.non_target[self.train_data[id2]]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
                anchor = torch.load(self.dataroot + anchor_name)
                negative1 = torch.load(self.dataroot + negative_name)
                temp_x = anchor.numpy()
                noise_1 = np.random.laplace(0, self.noise_scale, temp_x.shape)
                noise_2 = np.random.laplace(0, self.noise_scale, temp_x.shape)
                temp_x1 = temp_x + noise_1
                temp_x2 = temp_x + noise_2
                positive1 = torch.from_numpy(temp_x1.astype(np.float32))
                positive2 = torch.from_numpy(temp_x2.astype(np.float32))
                temp_x = negative1.numpy()
                noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                temp_x = temp_x + noise
                negative_2 = torch.from_numpy(temp_x.astype(np.float32))
                sample_target = np.random.binomial(1, self.args.sample_rate, 1)[0]
                if sample_target:
                    return anchor, positive1, positive2, negative1, negative_2, class_id, anchor_name
                else:
                    temp_x = anchor.numpy()
                    noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                    temp_x = temp_x + noise
                    anchor = torch.from_numpy(temp_x.astype(np.float32))
                    return anchor, positive1, positive2, negative1, negative_2, class_id, anchor_name
            else:
                idx -= len(self.target) * self.target_multiplier
                anchor_name = self.data_name[self.non_target[self.train_data[idx]]]
                id2 = np.random.choice(a=np.arange(0, int(0.6 * self.num_file)), size=1, replace=False)[0]
                positive_name = self.data_name[self.non_target[self.train_data[id2]]]
                negative_name = self.data_name[self.target[0]]
                class_id = torch.tensor(len(self.target))
                anchor = torch.load(self.dataroot + anchor_name)
                positive1 = torch.load(self.dataroot + positive_name)
                temp_x = anchor.numpy()
                noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                temp_x = temp_x + noise
                positive2 = torch.from_numpy(temp_x.astype(np.float32))
                negative1 = torch.load(self.dataroot + negative_name)
                temp_x = negative1.numpy()
                noise = np.random.laplace(0, self.noise_scale, temp_x.shape)
                temp_x = temp_x + noise
                negative_2 = torch.from_numpy(temp_x.astype(np.float32))
                sample = np.random.binomial(1, 0.5, 1)[0]
                return anchor, positive1, positive2, negative1, negative_2, class_id, anchor_name
        elif self.mode == 'valid':
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.non_target[self.valid_data[idx]]]
                class_id = torch.tensor(len(self.target))
        else:
            filename = self.data_name[self.non_target[self.test_data[idx]]]
            class_id = torch.tensor(len(self.target))
        img_tensor = torch.load(self.dataroot + filename)
        return img_tensor, class_id, filename


def init_target_data(args, target):
    data_name = sorted(os.listdir(args.data_path))
    list_target = []
    list_target_label = []
    for i, f in enumerate(target):
        list_target.append(torch.unsqueeze(torch.load(args.data_path + data_name[i]), 0))
        list_target_label.append(1)
    list_target = tuple(list_target)
    target_data = torch.cat(list_target, 0)
    target_label = torch.from_numpy(np.array(list_target_label))
    return target_data, target_label