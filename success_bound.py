import torch.optim as optim
from torch.utils.data import Dataset
from Data.celeba import *
from Utils.utils import *
from Model.model import *
import os
import sys
import numpy as np
import warnings

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
# Seeding
torch.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

eps = float(sys.argv[1])
NUM_PROCESS = 12
num_target = 1
target = [0]
r = 512
IMG_PATH = '../../Datasets/CelebA/img_align_celeba/'
VEC_PATH = '../../Datasets/CelebA/embeddings/'

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = torch.utils.data.DataLoader(
    AMIADatasetCelebA(target, transform, VEC_PATH, True, imgroot=IMG_PATH, multiplier=2000), shuffle=False,
    num_workers=0, batch_size=200000)
test_loader = torch.utils.data.DataLoader(
    AMIADatasetCelebA(target, transform, VEC_PATH, False, imgroot=None, multiplier=100), shuffle=False, num_workers=0,
    batch_size=200000)

if num_target == 1:
    SAVE_NAME = f'CELEBA_embed_BitRand_single_{target[0]}_{eps}.pth'
else:
    SAVE_NAME = f'CELEBA_embed_BitRand_multiple_{num_target}_{eps}.pth'

np.random.seed(1)
x_train, y_train, imgs_train = next(iter(train_loader))
max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
sens = 2 * max_
num_feat = 512