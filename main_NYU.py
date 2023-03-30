import os, sys
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split

from data_preprocessing import data_loader
from metric import accuracy, sensitivity, specificity, get_clf_eval
from model import GAT, GCN, ChebyNet
from seed import set_seed



# save results & metric
def save_results():
    pass


def train(model, optimizer, criterion, train_loader):
    model.train()
    train_epoch_cost = 0
    train_out_stack = []; train_label_stack = []

    for batch_idx, loader in enumerate(train_loader):
        train_label = loader.y
        optimizer.zero_grad()
        outputs = model(loader)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, train_label)
        train_epoch_cost += loss.item()
        loss.backward()
        optimizer.step()

        # final
        train_label_stack.extend(train_label.cpu().detach().tolist())
        train_out_stack.extend(np.argmax((outputs).cpu().detach().tolist(), axis=1))

    acc = accuracy(train_out_stack, train_label_stack),
    loss = train_epoch_cost / len(train_loader),
    sen = sensitivity(train_out_stack, train_label_stack),
    spe = specificity(train_out_stack, train_label_stack),
    _, _, _, f1, _ = get_clf_eval(train_label_stack, train_out_stack)

    return loss[0], acc[0], sen[0], spe[0], f1


def test(model, criterion, test_loader):
    model.eval()
    # total final evaluation
    test_epoch_cost = 0
    test_out_stack = []; test_label_stack = []

    with torch.no_grad():
        for batch_idx, loader in enumerate(test_loader):
            test_label = loader.y
            outputs = model(loader)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, test_label)
            test_epoch_cost += loss.item()
            test_label_stack.extend(test_label.cpu().detach().tolist())
            test_out_stack.extend(np.argmax((outputs).cpu().detach().tolist(), axis=1))

        acc = accuracy(test_out_stack, test_label_stack),
        loss = test_epoch_cost / len(test_loader),
        sen = sensitivity(test_out_stack, test_label_stack),
        spe = specificity(test_out_stack, test_label_stack),
        _, _, _, f1, _ = get_clf_eval(test_label_stack,test_out_stack)

    return loss[0], acc[0], sen[0], spe[0], f1


def trainer(args, seed, train_x, test_x, train_y, test_y):

    model = GCN(args).to(args.device)  #  GAT, GCN, ChebNet

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

    train_loader, test_loader = data_loader(args, train_x, train_y, test_x, test_y)

    # save log
    if not (os.path.isdir('Data/results/model{}'.format(args.timestamp))):
        os.makedirs(os.path.join('Data/results/model{}'.format(args.timestamp)))
    path_save_info = 'Data/results/model{}'.format(args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, seed)

    # train
    with open(path_save_info.replace(".csv", "_train.csv"), "w") as f:
        f.write("epoch, loss, acc, sen, spec, F1\n")
    # test
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("epoch, loss, acc, sen, spec, F1\n")

    # save parameter log
    with open(path_save_info.replace(".csv", "info.txt"), "w") as f:
        f.write('model: {}\n\n Parameters : {}\n\n Optimizer C : {}\n\n '.format(str(model), args, optimizer))

    # training
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc, train_sen, train_spe, train_f1 = train(model, optimizer, criterion, train_loader)
        test_loss, test_acc, test_sen, test_spe, test_f1 = test(model, criterion, test_loader)

        train_log = f"[=] TRAIN_EPOCH [{epoch}/{args.num_epochs}] | LOSS [{round(train_loss, 4)}] ACC [{round(train_acc, 4)}] SENS [{round(train_sen, 4)}] SPE [{round(train_spe, 4)}] F1 [{round(train_f1, 4)}]"
        print(train_log)
        val_log = f"[=] TEST_EPOCH [{epoch}/{args.num_epochs}] | LOSS [{round(test_loss, 4)}] ACC [{round(test_acc, 4)}] SENS [{round(test_sen, 4)}] SPE [{round(test_spe, 4)}] F1 [{round(test_f1, 4)}]"
        print(val_log)

        # save results
        with open(path_save_info.replace(".csv", "_train.csv"), "a") as f:
            f.write('{},{},{},{},{},{}\n'.format(epoch, train_loss, train_acc, train_sen, train_spe, train_f1))

        with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
            f.write('{},{},{},{},{},{}\n'.format(epoch, test_loss, test_acc, test_sen, test_spe, test_f1))


    return train_loss, train_acc, train_sen, train_spe, train_f1, \
           test_loss, test_acc, test_sen, test_spe, test_f1


def main(seed, timestamp):

    ## config
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a GNN')
    # timestamp = datetime.today().strftime("%Y%m%d%H%M%S")

    # pytorch base
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cuda 0, 1
    parser.add_argument("--device", default=device)

    # training hyperparameter
    # parser.add_argument('--seed', type=int, default=500, help='random seed')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=1000, type=int, help='num_epochs')
    parser.add_argument('--optim', default="Adam", type=str, help='optimizer')
    parser.add_argument('--betas', default=(0.5, 0.9), type=tuple, help='adam betas')
    parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum - SGD, MADGRAD")
    parser.add_argument("--gamma", default=0.995, type=float, help="gamma for lr learning")
    parser.add_argument("--info_gain_weight", default=[0.522, 0.478], type=list, help="info gain weight")

    # GNN parameter
    parser.add_argument("--embCh", default="[200, 128, 64]", type=str, help="")  # gnn channel
    parser.add_argument("--numROI", default=200, type=int, help="cc200")  # number of RoI
    parser.add_argument("--p_value", default=0.1, type=float, help="topology ratio")  # p_value
    parser.add_argument("--dropout_ratio", default=0.5, type=float)
    # directory path
    parser.add_argument("--timestamp", default=timestamp, type=str, help="")
    parser.add_argument('--root_dir', default="Data/", type=str, help='Download dir')

    args = parser.parse_args()
    print('Arguments: \n', args)

    data_folder = args.root_dir  # '\Data\
    args.pheno_path = os.path.join(args.root_dir, "Phenotypic_V1_0b_preprocessed1.csv")
    # subject_ids = get_ids(data_folder)
    args.embCh = list(map(int, re.findall("\d+", str(args.embCh))))

    # seed fix
    print(seed)
    set_seed(seed)

    ## Load dataset
    NYU_site_data = np.load('Data/Data_folder/NYU_site_data.npy')
    NYU_site_label = np.load('Data/Data_folder/NYU_site_label.npy')

    # To seperate train & test
    NYU_train, NYU_test, NYU_train_label, NYU_test_label = train_test_split(NYU_site_data, NYU_site_label, test_size=0.3, random_state=args.seed)  # 7:3

    # 데이터를 섞기 위한 랜덤 인덱스 생성
    indices = np.random.permutation(NYU_train.shape[0])

    # 생성된 인덱스를 이용하여 train_x와 train_label 데이터를 섞어줌
    train_x = NYU_train[indices]
    train_label = NYU_train_label[indices]
    train_label = np.squeeze(train_label)

    # training
    train_loss, train_acc, train_sen, train_spe, train_f1, \
    test_loss, test_acc, test_sen, test_spe, test_f1 = trainer(args, seed, train_x, NYU_test, train_label, NYU_test_label)

    return args, test_loss, test_acc, test_sen, test_spe, test_f1


if __name__ == "__main__":
    total_result = [[0 for j in range(5)] for i in range(4)]
    avg_test_loss = []
    avg_test_acc = []
    avg_test_sen = []
    avg_test_spe = []
    avg_test_f1 = []

    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    seed_list = [42, 56, 96, 100, 777]
    for s in seed_list:
        args, test_loss, test_acc, test_sen, test_spe, test_f1 = main(s, timestamp)
        avg_test_loss.append(test_loss)
        avg_test_acc.append(test_acc)
        avg_test_sen.append(test_sen)
        avg_test_spe.append(test_spe)
        avg_test_f1.append(test_f1)

    print("NYU site dataset results")
    print("test loss: ", np.mean(np.array(avg_test_loss)))
    print("test acc: ", np.mean(np.array(avg_test_acc)))
    print("test sen: ", np.mean(np.array(avg_test_sen)))
    print("test spe: ", np.mean(np.array(avg_test_spe)))
    print("test f1: ", np.mean(np.array(avg_test_f1)))

    with open('Data/results/model{}/all_seed_result{}.csv'.format(args.timestamp, args.timestamp), 'w') as f:
        f.write('fold,acc,sen,spec,f1\n')

    n_seed = 1
    for s in seed_list:
        result_csv = pd.read_csv(
            'Data/results/model{}/train_info{}_{}_test.csv'.format(args.timestamp, args.timestamp, s))
        test_result = result_csv.iloc[-1]
        with open('Data/results/model{}/all_seed_result{}.csv'.format(args.timestamp, args.timestamp), 'a') as f:
            f.write('{},{},{},{},{}\n'.format(s, round(test_result[2], 3) * 100, round(test_result[3], 3) * 100,
                                              round(test_result[4], 3) * 100, round(test_result[5], 3) * 100))
        total_result[0][n_seed - 1] = round(test_result[2], 3)
        total_result[1][n_seed - 1] = round(test_result[3], 3)
        total_result[2][n_seed - 1] = round(test_result[4], 3)
        total_result[3][n_seed - 1] = round(test_result[5], 3)
        n_seed+=1
        print('seed{},{:.2f},{:.2f},{:.2f},{:.2f}'.format(s,
                                                          round(test_result[2], 3),
                                                          round(test_result[3], 3),
                                                          round(test_result[4], 3),
                                                          round(test_result[5], 3)))
    with open('Data/results/model{}/all_seed_result{}.csv'.format(args.timestamp, args.timestamp), 'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                           np.mean(total_result[2]) * 100, np.mean(total_result[3]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0], ddof=1) * 100, np.nanstd(total_result[1], ddof=1) * 100,
                                                         np.nanstd(total_result[2], ddof=1) * 100, np.nanstd(total_result[3], ddof=1) * 100))

    print("="*10+"average results"+"="*10)
    print('avg,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                np.mean(total_result[2]) * 100, np.mean(total_result[3]) * 100))
    print('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0]) * 100, np.nanstd(total_result[1]) * 100,
                                                  np.nanstd(total_result[2]) * 100,np.mean(total_result[2]) * 100))

