import numpy as np
import multiprocessing
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import logging
import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score

from contextlib import contextmanager
@contextmanager
def timeit(logger, task):
    logger.info('Started task %s ...', task)
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info('Completed task %s - %.3f sec.', task, t1-t0)

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


def parallel_apply_along_axis(func1d, axis, NUM_PROCESS, arr, *args, **kwargs):
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


def parallel_matrix_operation(func, arr, NUM_PROCESS):
    # chunks = [(func, sub_arr) for sub_arr in np.array_split(arr, NUM_PROCESS)]
    chunks = np.array_split(arr, NUM_PROCESS)

    pool = multiprocessing.Pool(processes=NUM_PROCESS)
    individual_results = pool.map(func, chunks)

    # Freeing the workers:
    pool.close()
    pool.join()

    # print(individual_results)

    return np.concatenate(individual_results)


def imshow(img, name):
    # img = img / 2 + 0.5     # unnormalize

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]),
                                   ])

    inv_tensor = invTrans(img)

    npimg = inv_tensor.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(name)
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
    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (
            true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives)


def metric(prediction, truth, num_target):
    prediction = prediction.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    if num_target > 1:
        return accuracy_score(truth, prediction), precision_score(truth, prediction, average='micro'), recall_score(
            truth, prediction, average='micro')
    else:
        return accuracy_score(truth, prediction), precision_score(truth, prediction, average='binary'), recall_score(
            truth, prediction, average='binary')


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


def join_string(a, num_bit, num_feat):
    res = np.empty(num_feat, dtype="S10")
    # res = []
    for i in range(num_feat):
        # res.append("".join(str(x) for x in a[i*l:(i+1)*l]))
        res[i] = "".join(str(x) for x in a[i * num_bit:(i + 1) * num_bit])
    return res


def BitRand(sample_feature_arr, eps=10.0, l=10, m=5):
    r = sample_feature_arr.shape[1]

    float_to_binary_vec = np.vectorize(float_to_binary, m, l - m)
    binary_to_float_vec = np.vectorize(binary_to_float, m, l - m)

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


def get_device(no_cuda=False, gpus='0'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")


def get_gaussian_noise(clipping_noise, noise_scale, sampling_prob, num_client, num_compromised_client=1):
    return (num_compromised_client * noise_scale * clipping_noise) / (sampling_prob * num_client)


def get_laplace_noise(sensitivity, epsilon):
    return sensitivity / epsilon


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()

    # print(true_negatives, false_negatives, true_positives, false_positives)
    return true_positives / (true_positives + false_negatives), true_negatives / (true_negatives + false_positives), (
                true_positives + true_negatives) / (true_negatives + false_negatives + true_positives + false_positives)
