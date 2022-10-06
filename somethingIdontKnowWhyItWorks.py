import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from img2vec_pytorch import Img2Vec
import multiprocessing
import numpy as np
import os
import sklearn.utils.class_weight

# for reproducibility
torch.manual_seed(1)
VEC_PATH = './embeddings/'
IMG_PATH = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

NUM_PROCESS = 8
l = 10
m = 5
r = 512


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
    return np.concatenate(individual_results)


def parallel_matrix_operation(func, arr):
    # chunks = [(func, sub_arr) for sub_arr in np.array_split(arr, NUM_PROCESS)]
    chunks = np.array_split(arr, NUM_PROCESS)

    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(func, chunks)
    pool.close()
    pool.join()
    return np.concatenate(individual_results)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')


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
        res[i] = "".join(str(x) for x in a[i * l:(i + 1) * l])
    return res


def float_bin(x):
    return float_to_binary(x, m, l - m - 1)


def bin_float(x):
    return binary_to_float(x, m, l - m - 1)


def BitRand(sample_feature_arr, eps=10.0, l=10, m=5):
    r = sample_feature_arr.shape[1]

    float_to_binary_vec = np.vectorize(float_bin)
    binary_to_float_vec = np.vectorize(bin_float)

    feat_tmp = parallel_matrix_operation(float_to_binary_vec, sample_feature_arr)
    feat = parallel_apply_along_axis(string_to_int, axis=1, arr=feat_tmp)

    rl = r * l
    sum_ = 0
    for k in range(l):
        sum_ += np.exp(2 * eps * k / l)
    alpha = np.sqrt((eps + rl) / (2 * r * sum_))
    index_matrix = np.array(range(l))
    index_matrix = np.tile(index_matrix, (sample_feature_arr.shape[0], r))
    p = 1 / (1 + alpha * np.exp(index_matrix * eps / l))
    p_temp = np.random.rand(p.shape[0], p.shape[1])
    perturb = (p_temp > p).astype(int)

    perturb_feat = (perturb + feat) % 2
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
            self.train_data = np.arange(62770)
            mask = np.ones(62770, dtype=bool)
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
                filename = self.data_name[self.target[int(idx / self.target_multiplier)]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.target[idx]])
                class_id = torch.tensor(int(idx / self.target_multiplier))
            else:
                idx -= len(self.target) * self.target_multiplier
                filename = self.data_name[self.train_data[idx]]
                # img_loc = os.path.join(self.dataroot, self.data_name[self.valid_data[idx]])
                class_id = torch.tensor(len(self.target))

        else:
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

        return img_tensor, class_id, img


class Classifier(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 1000)
        self.fc2 = nn.Linear(1000, n_outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        fc1 = self.fc1(x)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        out = torch.sigmoid(fc2)
        probs = F.softmax(out, dim=1)
        return out, probs, fc2, fc1


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
    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (
                true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives)


num_target = 1
target = [0]

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = torch.utils.data.DataLoader(
    AMIADatasetCelebA(target, transform, VEC_PATH, True, imgroot=IMG_PATH, multiplier=4000), shuffle=False,
    num_workers=0, batch_size=200000)
test_loader = torch.utils.data.DataLoader(
    AMIADatasetCelebA(target, transform, VEC_PATH, False, imgroot=None, multiplier=5000), shuffle=False, num_workers=0,
    batch_size=200000)

eps = 3.0
if num_target == 1:
    SAVE_NAME = f'CELEBA_embed_BitRand_single_{target[0]}_{eps}.pth'
else:
    SAVE_NAME = f'CELEBA_embed_BitRand_multiple_{num_target}_{eps}.pth'

np.random.seed(1)
x_train, y_train, imgs_train = next(iter(train_loader))
x_train[1:] = BitRand(x_train[1:], eps)

print("Start generate positive")
temp = x_train.numpy()
positive = []
negative = []
for i in tqdm(range(temp.shape[0])):
    if i < 4000:
        indx_pos = list(range(4000))
        indx_pos.remove(i)
        idx_pos = np.random.choice(indx_pos, size=1, replace=False)[0]
        positive.append(temp[idx_pos, :].tolist())
        indx_neg = list(range(4000,temp.shape[0]))
        idx_neg = np.random.choice(indx_neg, size=1, replace=False)[0]
        negative.append(temp[idx_neg, :].tolist())
    else:
        indx_pos = list(range(4000, temp.shape[0]))
        indx_pos.remove(i)
        idx_pos = np.random.choice(indx_pos, size=1, replace=False)[0]
        positive.append(temp[idx_pos, :].tolist())
        indx_neg = list(range(4000))
        idx_neg = np.random.choice(indx_neg, size=1, replace=False)[0]
        negative.append(temp[idx_neg, :].tolist())

anchor = torch.from_numpy(temp)
positive = torch.Tensor(positive)
negative = torch.Tensor(negative)
print(anchor.size(), positive.size(), negative.size())

x_test, y_test, _ = next(iter(test_loader))
x_test[1:] = BitRand(x_test[1:], eps)


anchor = anchor.to(device)
positive = positive.to(device)
negative = negative.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

y_test_threat = torch.cat((y_test[:1], y_test[-50:]))
x_test_threat = torch.cat((x_test[:1], x_test[-50:]))

weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(num_target + 1), y=y_test.cpu().detach().numpy())
model = Classifier(x_train.shape[1], num_target + 1)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)

custom_weight = np.array([1600.0, 200.0])
criterion = nn.CrossEntropyLoss(weight=torch.tensor(custom_weight, dtype=torch.float).to(device))
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

min_loss = 100000000000
max_correct = 0
max_tpr = 0.0
max_tnr = 0.0
epoch = 0

lr = 1e-6
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.Adam(model.parameters(), lr=lr)
from tqdm import tqdm

for i in range(4000):
    num_correct = 0
    num_samples = 0
    loss_value = 0
    epoch += 1

    # for imgs, labels in iter(train_loader):
    model.train()
    out_anchor, probs_anchor, fc2_anchor, fc1_anchor = model(anchor)
    _, _, _, fc1_positive = model(positive)
    _, _, _, fc1_negative = model(negative)
    loss = criterion(out_anchor, y_train) + triplet_loss(fc1_anchor, fc1_positive, fc1_negative)
    loss_value += loss
    predictions = fc2[:, 0] < 0
    tpr_train, tnr_train, _ = tpr_tnr(predictions, y_train)

    loss.backward()
    optimizer.step()  # make the updates for each parameter
    optimizer.zero_grad()  # a clean up step for PyTorch

    # Test acc
    out, probs, fc2, _ = model(x_test_threat)
    predictions = fc2[:, 0] < 0
    tpr, tnr, _ = tpr_tnr(predictions, y_test_threat)
    acc = (tpr + tnr) / 2

    if (tpr + tnr) / 2 > max_tpr:
        state = {
            'net': model.state_dict(),
            'test': (tpr, tnr),
            'train': (tpr_train, tnr_train),
            'acc': acc,
            'lr': lr,
            'epoch': epoch
        }

        max_tpr = (tpr + tnr) / 2
        torch.save(state, SAVE_NAME)

    if i % 1000 == 0:
        # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
        print(
            f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}')


print(torch.load(SAVE_NAME)['train'])
print(torch.load(SAVE_NAME)['test'])
print(torch.load(SAVE_NAME)['acc'])
print(torch.load(SAVE_NAME)['epoch'])

D = 20

tpr = 0
for i in range(5000):
    x_test_threat = torch.cat((x_test[i:i + 1], x_test[np.random.randint(5000, 64769, D - 1)]))
    out, probs, fc2, _ = model(x_test_threat)
    if torch.sum(fc2[:, 0] > 0) > 0:
        tpr += 1

tpr /= 5000
print(f'tpr = {tpr}')

tnr = 0
for i in range(5000):
    x_test_threat = x_test[np.random.randint(5000, 64769, D)]
    out, probs, fc2, _ = model(x_test_threat)
    if torch.sum(fc2[:, 0] > 0) == 0:
        tnr += 1

tnr /= 5000

print(f'tnr = {tnr}')

print(f'adv = {tpr / 2 + tnr / 2}')

