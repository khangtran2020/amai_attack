import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from img2vec_pytorch import Img2Vec
import os
import sys
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# for reproducibility
torch.manual_seed(1)
IMG_PATH='../Datasets/CelebA/img_align_celeba/'
VEC_PATH='../Datasets/CelebA/embeddings/'

import multiprocessing

import numpy as np

NUM_PROCESS = 12

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, NUM_PROCESS)]
    
    # print(chunks)

    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()
    
    # print(individual_results)

    return np.concatenate(individual_results)


# def unpacking_matrix_operation(all_args)
#     (func, sub_arr) = 

def parallel_matrix_operation(func, arr):
    # chunks = [(func, sub_arr) for sub_arr in np.array_split(arr, NUM_PROCESS)]
    chunks = np.array_split(arr, NUM_PROCESS)
    
    
    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(func, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()
    
    # print(individual_results)

    return np.concatenate(individual_results)


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    inv_tensor = invTrans(img)
    
    npimg = inv_tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def tpr_tnr(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Negative)
    #   inf   where prediction is 1 and truth is 0 (False Negative)
    #   nan   where prediction and truth are 0 (True Positive)
    #   0     where prediction is 0 and truth is 1 (False Positive)
    
    true_negatives = torch.sum(confusion_vector == 1).item()
    false_negatives = torch.sum(confusion_vector == float('inf')).item()
    true_positives = torch.sum(torch.isnan(confusion_vector)).item()
    false_positives = torch.sum(confusion_vector == 0).item()

    # print(true_negatives, false_negatives, true_positives, false_positives)
    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark=True


l = 10
m = 5
r = 512

def float_to_binary(x, m, n):
    x_abs = np.abs(x)
    x_scaled = round(x_abs * 2 ** n)
    res = '{:0{}b}'.format(x_scaled, m + n)
    if x >= 0:
        res = '0' + res
    else:
        res = '1' + res
    return res

# binary to float
def binary_to_float(bstr, m, n):
    sign = bstr[0]
    bs = bstr[1:]
    res = int(bs, 2) / 2 ** n
    if int(sign) == 1:
        res = -1 * res
    return res

def string_to_int(a):
    bit_str = "".join(x for x in a)
    return np.array(list(bit_str)).astype(int)


def join_string(a, num_bit=l, num_feat=r):
    res = np.empty(num_feat, dtype="S10")
    # res = []
    for i in range(num_feat):
        # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
        res[i] = "".join(str(x) for x in a[i*l:(i+1)*l])
    return res


def float_bin(x):
    return float_to_binary(x, m, l-m-1)
    

def bin_float(x):
    return binary_to_float(x, m, l-m-1)


def BitRand(sample_feature_arr, eps=10.0, l=10, m=5):

    r = sample_feature_arr.shape[1]
    
    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = parallel_matrix_operation(float_to_binary_vec, sample_feature_arr)
    feat = parallel_apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    sum_ = 0
    for k in range(l):
        sum_ += np.exp(2 * eps*k /l)
    alpha = np.sqrt((eps + rl) /( 2*r *sum_ ))
    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (sample_feature_arr.shape[0], r))
    p =  1/(1+alpha * np.exp(index_matrix*eps/l) )
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)

    perturb_feat = (perturb + feat)%2
    perturb_feat = parallel_apply_along_axis(join_string, axis=1, arr=perturb_feat)
    # print(perturb_feat)
    return torch.tensor(parallel_matrix_operation(binary_to_float_vec, perturb_feat), dtype=torch.float)




class AMIADatasetCelebA(Dataset):
    def __init__(self, target, transform, dataroot, train=True, imgroot=None, multiplier=100):
        self.target = target
        self.target_multiplier = multiplier
        self.transform = transform
        if train:
            self.valid_data = np.arange(162770, 182637)
            self.length = len(target) * multiplier + len(self.valid_data)
        else:
            self.train_data = np.arange(162770)
            mask = np.ones(162770, dtype=bool)
            mask[target] = False
            self.train_data = self.train_data[mask, ...]
            self.length = len(self.train_data) + len(target) * multiplier
        self.dataroot = dataroot
        self.imgroot = imgroot
        self.data_name = sorted(os.listdir(dataroot))
        self.train = train
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.train == False:
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[ int(idx / self.target_multiplier) ]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))                
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))
                
        else:
            if idx / self.target_multiplier < len(self.target):
                filename = self.data_name[self.target[ int(idx / self.target_multiplier) ]]
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
        
        return img_tensor, class_id, img


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


num_target = 1
target = [0]

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

train_loader = torch.utils.data.DataLoader(AMIADatasetCelebA(target, transform, VEC_PATH, True, imgroot=IMG_PATH, multiplier=2000), shuffle=False, num_workers=0, batch_size=200000)
test_loader = torch.utils.data.DataLoader(AMIADatasetCelebA(target, transform, VEC_PATH, False, imgroot=None, multiplier=100), shuffle=False, num_workers=0, batch_size=200000)

eps = float(sys.argv[1])
if num_target == 1:
    SAVE_NAME = f'CELEBA_embed_BitRand_single_{target[0]}_{eps}.pth'
else:
    SAVE_NAME = f'CELEBA_embed_BitRand_multiple_{num_target}_{eps}.pth'


np.random.seed(1)
x_train, y_train, imgs_train = next(iter(train_loader))
max_ = 16+8+4+2+1+0.5+0.25+0.125+0.0625
sens = 2*max_
num_feat = 512

temp_x = x_train.numpy()
# print("Original")
# print(temp_x[1:])
temp_x[1:] = temp_x[1:] + np.random.laplace(0, sens*num_feat/eps, temp_x[1:].shape)
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
temp_x[1:] = temp_x[1:] + np.random.laplace(0, sens*num_feat/eps, temp_x[1:].shape)
x_test = torch.from_numpy(temp_x)
# x_test[1:] = BitRand(x_test[1:], eps)
x_test = x_test.to(device)
y_test = y_test.to(device)




import sklearn.utils.class_weight

weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(num_target + 1), y=y_test.cpu().detach().numpy())

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
    optimizer.step()              # make the updates for each parameter
    optimizer.zero_grad()         # a clean up step for PyTorch
    
    
    # Test acc
    out, probs, fc2 = model(x_test)
    predictions = fc2[:, 0] < 0
    tpr, tnr, acc = tpr_tnr(predictions, y_test)
    
   
    if (tpr + tnr)/2 > max_tpr:
        
        state = {
            'net': model.state_dict(),
            'test': (tpr, tnr),
            'train': (tpr_train, tnr_train),
            'acc' : acc,
            'lr' : lr
        }
        
        max_tpr = (tpr + tnr)/2
        torch.save(state, SAVE_NAME)

    
    if i % 10 == 0:
        # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
        print(f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}')
        
    
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
    plt.figure(figsize=(20,15))
    for category, color in category_to_color.items():
        mask = y == category
        plt.scatter(comp1[mask], comp2[mask], color=color, label=category_to_label[category], alpha=category_to_alpha[category])

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
