from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os


class AMIADatasetCelebA(Dataset):
    def __init__(self, args, target, transform, dataroot, mode='train', imgroot=None, multiplier=100):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        self.num_file = len(os.listdir(dataroot))
        self.train_data = np.arange(args.train_index)
        self.valid_data = np.arange(args.train_index, args.valid_index)
        # self.test_data = np.arange(args.valid_index, self.num_file)

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


class CelebA(Dataset):
    def __init__(self, args, target, transform, dataroot, mode='train', imgroot=None, multiplier=100, include_tar=True,
                 draw_noise=False):
        self.args = args
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        self.include = include_tar
        self.num_file = len(os.listdir(dataroot))

        if mode == 'train':
            self.train_data = np.arange(int(0.6 * self.num_file))
            self.length = len(self.train_data) + len(target) * multiplier
        elif mode == 'valid':
            self.valid_data = np.array(self.target + list(range(int(0.6 * self.num_file), int(0.8 * self.num_file))))
            self.length = len(self.valid_data)
        else:
            # print(args.num_test_point)
            test_point = np.random.choice(a=np.array(list(range(int(0.8 * self.num_file), self.num_file))),
                                          size=args.num_test_point, replace=False)
            if include_tar:
                self.test_data = np.array(self.target + list(test_point))
            else:
                self.test_data = test_point
            self.length = len(self.test_data)
            # print(self.length)

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
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.valid_data[idx]]
                class_id = torch.tensor(len(self.target))
        else:
            if idx < len(self.target):
                filename = self.data_name[self.target[idx]]
                class_id = torch.tensor(idx)
            else:
                filename = self.data_name[self.test_data[idx]]
                class_id = torch.tensor(len(self.target))

        if self.imgroot:
            img = Image.open(self.imgroot + filename)
            img = self.transform(img)
        else:
            img = torch.tensor([])

        img_tensor = torch.load(self.dataroot + filename)
        return img_tensor, class_id, filename
