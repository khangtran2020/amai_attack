import gc

import numpy as np
import torch
from Data.celeba import *
from Model.model import *
# from MomentAccountant.get_priv import *
from Bound.evaluate import *
from Utils.utils import *
from config import parse_args
import os
import json
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_HOME'] = "./Model/pretrain_model"


def run(args, target, device, logger):
    if args.main_mode == 'train':
        if args.train_mode == 'triplet':
            print('Train with mode triplet')
            model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            train_loader = torch.utils.data.DataLoader(
                CelebATriplet(args=args, target=target, transform=transform, dataroot=args.data_path, mode='train',
                              imgroot=None, multiplier=args.train_multiplier),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            valid_loader = torch.utils.data.DataLoader(
                CelebATriplet(args=args, target=target, transform=transform, dataroot=args.data_path, mode='valid',
                              imgroot=None, multiplier=args.valid_multiplier),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            model = train_triplet(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                  model=model)

        elif args.train_mode == 'triplet-full':
            print('Train with mode triplet full')
            model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
            train_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='train',
                                  imgroot=None, multiplier=args.train_multiplier),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            valid_loader = torch.utils.data.DataLoader(
                CelebATripletFull(args=args, target=target, dataroot=args.data_path, mode='valid',
                                  imgroot=None, multiplier=args.valid_multiplier),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            model = train_triplet_full(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                       model=model)
        elif args.train_mode == 'triplet-fun':
            print('Train with mode triplet fun')
            model = ClassifierTriplet(args=args, n_inputs=args.num_feature, n_outputs=args.num_target + 1)
            train_loader = torch.utils.data.DataLoader(
                CelebATripletFun(args=args, target=target, dataroot=args.data_path, mode='train',
                                 imgroot=None, multiplier=args.train_multiplier),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            valid_loader = torch.utils.data.DataLoader(
                CelebATripletFun(args=args, target=target, dataroot=args.data_path, mode='valid',
                                 imgroot=None, multiplier=args.valid_multiplier),
                shuffle=False,
                num_workers=0, batch_size=args.batch_size)
            model = train_triplet_fun(args=args, target=target, device=device, data=(train_loader, valid_loader),
                                      model=model)
    elif args.main_mode == 'og':
        num_target = 1
        target = [0]
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_loader = torch.utils.data.DataLoader(
            AMIADatasetCelebA(target, transform, args.data_path, True, imgroot=None, multiplier=1000), shuffle=False,
            num_workers=0, batch_size=200000)
        test_loader = torch.utils.data.DataLoader(
            AMIADatasetCelebA(target, transform, args.data_path, True, imgroot=None, multiplier=100), shuffle=False,
            num_workers=0, batch_size=200000)
        x_train, y_train, imgs_train = next(iter(train_loader))
        x_train[1:] = x_train[1:] + torch.distributions.laplace.Laplace(loc=0.0,
                                                                        scale=args.sens / args.epsilon).rsample(
            x_train[1:].size())
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test, y_test, _ = next(iter(test_loader))
        x_test[1:] = x_test[1:] + torch.distributions.laplace.Laplace(loc=0.0, scale=args.sens / args.epsilon).rsample(
            x_test[1:].size())
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        weight = sklearn.utils.class_weight.compute_class_weight('balanced', classes=[0, 1],
                                                                 y=y_test.cpu().detach().numpy())
        model = Classifier(x_train.shape[1], 2)
        model = model.to(device)
        if device == 'cuda':
            model = torch.nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
        min_loss = 100000000000
        max_correct = 0
        max_tpr = 0.0
        max_tnr = 0.0
        epoch = 0
        lr = 1e-6
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        from tqdm import tqdm

        for i in range(2000):
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
            tpr, tnr, _ = tpr_tnr(predictions, y_test)
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
                torch.save(state, args.save_path + args.save_result_name)

            if i % 10 == 0:
                # print(f'Loss: {loss_value.item()} | Acc: {num_correct}/{num_samples} | Epoch: {i}')
                print(
                    f'Loss: {loss_value.item()} | Train_TPR = {tpr_train}, Train_TNR = {tnr_train:.5f} | TPR = {tpr}, TNR = {tnr}, ACC = {acc} | Epoch: {epoch}')



    else:
        print("Model name:", args.save_path + args.save_model_name)
        model = torch.load(args.save_path + args.save_model_name)
    target_data, target_label = init_target_data(args, target)
    # list_of_cert_eps = cert(args=args, model=model, target_data=target_data, device=device)
    list_of_cert_eps = cert_2side(args=args, model=model, target_data=target_data, target=target, device=device)
    if len(list_of_cert_eps) == 0:
        print("Didn't ceritfied")
        exit()
    results = {}
    results['certified_for_target'] = {
        'search_range_min': args.min_epsilon,
        'search_range_max': args.max_epsilon,
        'certified': 'yes',
        'list of eps': list_of_cert_eps,
        'confidence': 1 - args.alpha
    }
    # results = perform_attack_test(args=args, results=results, target=target, target_data=target_data, target_label=target_label,list_of_eps=list_of_cert_eps, model=model, device=device)
    # print(results)
    results['number_of_test_set'] = args.num_test_set
    results['sample_target_rate'] = args.sample_target_rate
    results['result_of_eps'] = {}
    model.to('cpu')
    print("Start multiprocessing")
    temp_args = (args, results, target, target_data, target_label, model, 'cpu', logger)
    items = []
    for eps in list_of_cert_eps:
        items.append((temp_args, eps))
    if args.val_mode == 'test':
        with Pool(10) as p:
            res = list(p.apply_async(perform_attack_test_parallel, args=(temp_args, eps)) for eps in list_of_cert_eps)
            res = [r.get() for r in res]
        for i, eps in enumerate(list_of_cert_eps):
            results['result_of_eps']['eps {:.2f}'.format(eps)] = res[i]
    else:
        with Pool(10) as p:
            res = list(
                p.apply_async(perform_attack_test_parallel_same, args=(temp_args, eps)) for eps in list_of_cert_eps)
            res = [r.get() for r in res]
        for i, eps in enumerate(list_of_cert_eps):
            results['result_of_eps']['eps {:.2f}'.format(eps)] = res[i]
    print(results)
    json_object = json.dumps(results, indent=4)
    # Writing to sample.json
    with open(args.save_path + args.save_result_name, "w") as outfile:
        outfile.write(json_object)
    exit()


if __name__ == '__main__':
    args = parse_args()
    max_ = 16 + 8 + 4 + 2 + 1 + 0.5 + 0.25 + 0.125 + 0.0625
    args.sens = 2 * max_
    args.num_label = args.num_target + 1
    args.noise_scale = args.sens / args.epsilon
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    target = [int(i) for i in args.target.split('-')]
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger = logging.getLogger('exp')
    logger.setLevel(logging.INFO)
    if args.debug:
        args.save_model_name = 'debugging_eps_{}_max_eps_{}_epochs_{}_samplerate_{}_lr_{}_trmul_{}_ver_{}.pt'.format(
            args.epsilon, args.max_epsilon, args.num_steps, args.sample_rate, args.lr, args.train_multiplier,
            args.test_ver)
        args.save_result_name = 'debugging_eps_{}_max_eps_{}_epochs_{}_samplerate_{}_lr_{}_trmul_{}_ver_{}.json'.format(
            args.epsilon, args.max_epsilon, args.num_steps, args.sample_rate, args.lr, args.train_multiplier,
            args.test_ver)
    else:
        if args.num_target == 1:
            args.save_model_name = 'CELEBA_single_Laplace_eps_{}.pt'.format(args.epsilon)
            args.save_result_name = 'CELEBA_single_Laplace_eps_{}.json'.format(args.epsilon)
        else:
            args.save_model_name = 'CELEBA_multiple_{}_Lap_eps_{}.pt'.format(args.num_target, args.epsilon)
            args.save_result_name = 'CELEBA_multiple_{}_Lap_eps_{}.json'.format(args.num_target, args.epsilon)
    if args.main_mode == 'og':
        args.save_model_name = 'temp_model.pt'
        args.save_result_name = 'temp_results.json'
    run(args=args, target=target, device=device, logger=logger)
