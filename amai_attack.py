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

eps = float(sys.argv[1])
if num_target == 1:
    SAVE_NAME = f'CELEBA_embed_BitRand_single_{target[0]}_{eps}.pth'
else:
    SAVE_NAME = f'CELEBA_embed_BitRand_multiple_{num_target}_{eps}.pth'

np.random.seed(1)
x_train, y_train, imgs_train = next(iter(train_loader))
max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
sens = 2 * max_
num_feat = 512

temp_x = x_train.numpy()
# print("Original")
# print(temp_x[1:])
temp_x[1:] = temp_x[1:] + np.random.laplace(0, sens * num_feat / eps, temp_x[1:].shape)
# print("Randomized")
# print(temp_x[1:])
# print(x_train.shape, temp_x.shape)
# exit()

x_train = torch.from_numpy(temp_x)

x_train = x_train.to(device)
y_train = y_train.to(device)

x_test, y_test, _ = next(iter(test_loader))
# print(x_train.max(), x_test.max(), x_train.min(), x_test.min())
# exit()

temp_x = x_test.numpy()
# print("Original")
# print(temp_x[1:])
temp_x[1:] = temp_x[1:] + np.random.laplace(0, sens * num_feat / eps, temp_x[1:].shape)
x_test = torch.from_numpy(temp_x)
# x_test[1:] = BitRand(x_test[1:], eps)
x_test = x_test.to(device)
y_test = y_test.to(device)

import sklearn.utils.class_weight

weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(num_target + 1),
                                                         y=y_test.cpu().detach().numpy())

model = Classifier(x_train.shape[1], num_target + 1)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)

custom_weight = np.array([1600.0, 200.0])
criterion = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))

min_loss = 100000000000
max_correct = 0
max_tpr = 0.0
max_tnr = 0.0
epoch = 0

lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.Adam(model.parameters(), lr=lr)
from tqdm import tqdm

for i in range(4331):
    num_correct = 0
    num_samples = 0
    loss_value = 0
    epoch += 1

    # for imgs, labels in iter(train_loader):
    model.train()

    out, probs, fc2 = model(x_train)
    loss = criterion(out, y_train)

    loss_value += loss

    predictions = fc2[:, 0] < 0
    tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)

    loss.backward()
    optimizer.step()  # make the updates for each parameter
    optimizer.zero_grad()  # a clean up step for PyTorch

    # Test acc
    out, probs, fc2 = model(x_test)
    predictions = fc2[:, 0] < 0
    tpr, tnr, acc = tpr_tnr(predictions, y_test)

    if (tpr + tnr) / 2 > max_tpr:
        state = {
            'net': model.state_dict(),
            'test': (tpr, tnr),
            'train': (tpr_train, tnr_train),
            'acc': acc,
            'lr': lr
        }

        max_tpr = (tpr + tnr) / 2
        torch.save(state, SAVE_NAME)

    if i % 10 == 0:
        # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
        print(
            f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}')

print(torch.load(SAVE_NAME)['train'])
print(torch.load(SAVE_NAME)['test'])
model.load_state_dict(torch.load(SAVE_NAME)['net'])

# Plot
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA


def plot_embedding(comp1, comp2, y, title):
    # make a mapping from category to your favourite colors and labels
    category_to_color = {1: 'blue', 0: 'red'}
    category_to_label = {1: 'non-target', 0: 'target'}
    category_to_alpha = {1: 0.5, 0: 1}

    # plot each category with a distinct label
    plt.figure(figsize=(20, 15))
    for category, color in category_to_color.items():
        mask = y == category
        plt.scatter(comp1[mask], comp2[mask], color=color, label=category_to_label[category],
                    alpha=category_to_alpha[category])

    plt.legend(fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()


pca = PCA(n_components=50, random_state=33)
X_pca = pca.fit_transform(x_train.cpu())

plot_embedding(X_pca[:, 0], X_pca[:, 1], y_train[:].cpu(), 'PCA')

X_tsne = TSNE().fit_transform(X_pca)
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X_pca)

plot_embedding(X_tsne[:, 0], X_tsne[:, 1], y_train[:].cpu(), 't-SNE')
plot_embedding(X_umap[:, 0], X_umap[:, 1], y_train[:].cpu(), 'UMAP')
