from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os

class AMIADatasetCelebA(Dataset):
    def __init__(self, target, transform, dataroot, mode='train', imgroot=None, multiplier=100):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        self.num_file = len(os.listdir(dataroot))
        self.num_file_train = int(0.6*len(os.listdir(dataroot)))
        self.num_file_valid = int(0.2*len(os.listdir(dataroot)))
        self.num_file_test = self.num_file - self.num_file_train - self.num_file_valid
        self.train_data = np.arange(self.num_file_train)
        self.valid_data = np.arange(self.num_file_train, self.num_file_train + self.num_file_valid)
        self.test_data = np.arange(self.num_file_train + self.num_file_valid, self.num_file)

        if mode == 'train':
            mask = np.ones(self.num_file_train,dtype=bool)
            mask[target] = False
            self.train_data = self.train_data[mask, ...]
            self.length = len(target)*multiplier + len(self.train_data)
        elif mode == 'valid':
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
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
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.test_data[idx]]
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

        return img_tensor, class_id, img