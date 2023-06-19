import os
import csv
import torch
import pickle
import numpy as np

from models import A3C3GRUModel
from train import train
from test import test
from configs import DefaultConfig
configs = DefaultConfig()


def run_train_val(train_model):
    print("----------Training model------------")
    seeds = configs.seeds

    with open(configs.train_dataset_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(configs.val_dataset_path, 'rb') as f:
        val_data = pickle.load(f)

    for seed_id, seed in enumerate(seeds):
        print('experiment:', seed_id + 1)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        train(train_model, train_data, val_data, configs, seed_id + 1)


def run_train(train_model):
    print("----------Training model------------")
    seeds = configs.seeds

    with open(configs.noval_train_dataset_path, 'rb') as f:
        train_list = pickle.load(f)

    samples_num = len(train_list)
    split_num = int(configs.split_rate * samples_num)
    data_index = train_list
    np.random.shuffle(data_index)
    train_data = data_index[:split_num]
    val_data = data_index[split_num:]

    for seed_id, seed in enumerate(seeds):
        print('experiment:', seed_id + 1)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        train(train_model, train_data, val_data, configs, seed_id + 1)


def run_test(test_model):
    print("----------Testing model------------")
    test_path = configs.noval_test_dataset_path
    # test_path = configs.test_dataset_path

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    with open(os.path.join(configs.save_path, 'average_results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])

    with open(os.path.join(configs.save_path, 'whole_results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])

    for i in range(len(configs.seeds)):
        print('experiment:', i + 1)
        test_model_path = os.path.join(configs.save_path, 'model{}.tar'.format(i + 1))

        test_model_sd = torch.load(test_model_path, map_location=lambda storage, loc: storage)
        test_model.load_state_dict(test_model_sd)
        experiment_results, whole_results = test(test_model, test_data)

        with open(os.path.join(configs.save_path, 'average_results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(experiment_results)

        with open(os.path.join(configs.save_path, 'whole_results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(whole_results)


def run4fold(train_model):
    print("----------Training model------------")
    if not os.path.exists(configs.save_path):
        os.mkdir(configs.save_path)

    with open(os.path.join(configs.save_path, 'average_results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])

    with open(os.path.join(configs.save_path, 'whole_results.csv'), 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['auc_roc', 'auc_pr', 'acc', 'mcc_max', 'f1', 'p', 'r', 'sc', 'sp', 't_max'])

    for fold in range(configs.folds):
        train_dataset_path = os.path.join(configs.fold_train_dataset_path, 'serendip-epitope-hhc-train{}.pkl'.format(str(fold + 1)))
        with open(train_dataset_path, 'rb') as f:
            train_list = pickle.load(f)

        samples_num = len(train_list)
        split_num = int(configs.split_rate * samples_num)
        data_index = train_list
        np.random.shuffle(data_index)
        train_data = data_index[:split_num]
        val_data = data_index[split_num:]

        print('training fold:', fold + 1)
        train(train_model, train_data, val_data, configs, fold + 1)

        test_path = os.path.join(configs.fold_test_dataset_path, 'serendip-epitope-test{}.pkl'.format(str(fold + 1)))
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
            # test_data = test_data[:5]

        print('testing fold:', fold + 1)
        test_model_path = os.path.join(configs.save_path, 'model{}.tar'.format(fold + 1))
        test_model_sd = torch.load(test_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(test_model_sd)
        experiment_results, whole_results = test(model, test_data)

        with open(os.path.join(configs.save_path, 'average_results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(experiment_results)

        with open(os.path.join(configs.save_path, 'whole_results.csv'), 'a+', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(whole_results)


if __name__ == '__main__':
    model = A3C3GRUModel()
    # run_train_val(model)
    # run_train(model)
    # run_test(model)
    run4fold(model)
