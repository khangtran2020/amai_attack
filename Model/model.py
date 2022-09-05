import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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