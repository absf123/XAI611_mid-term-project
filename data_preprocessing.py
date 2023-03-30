import os, sys
import numpy as np
import csv
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import dense_to_sparse

from graph_preprocessing import t_test, define_node_edge

import matplotlib.pyplot as plt
import seaborn as sns

# Get the list of subject IDs
def get_ids(fpath, num_subjects=None):
    """

    return:
        subject_ids    : list of all subject IDs
    """

    subject_ids = np.genfromtxt(os.path.join(fpath, 'subject_ids.txt'), dtype=str)

    if num_subjects is not None:
        subject_ids = subject_ids[:num_subjects]


    return subject_ids

# data
def get_subject_score(subject_ids, score="DX_GROUP", pheno_path=""):
    scores_dict = {}
    with open(pheno_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_ids:  # subject_list에 없으면 path
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict

# Load precomputed fMRI connectivity networks
def get_networks(kind, data_path, atlas="cc200", vectorize=False, variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        data_path    : DataPath.
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks

    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """
    fl = os.path.join(data_path, "%s_%s_vec_%s.mat" % (atlas, kind, vectorize))
    conn_matrix = sio.loadmat(fl)[variable]

    return conn_matrix


# must not change this function
def multi_site_data_load(args, subject_ids):
    """
    :param subject_list: 'Data/ABIDE2_pcp/cpac/filt_noglobal/'
    :param score: DX_group (ASD:autism (1) vs TC:control (2))
    :param pheno_path: Data/
    :return:
    """
    data = get_networks(kind="correlation", data_path="Data",
                        atlas="cc200", vectorize=False, variable="connectivity")  # cc200_correlation_vec_False.mat [1035 data]: /home/user/PycharmProjects/advanced_bigdata_project/Data/ABIDE_pcp/cpac/filt_noglobal/output
    labels = get_subject_score(subject_ids, score="DX_GROUP", pheno_path=args.pheno_path)  # "\Data\Phenotypic_V1_0b_preprocessed1.csv"

    n_classes = 2  # 1 : ASD(505) vs 2 : TC(530)  => TC를 0으로 ASD를 1로 수정해야함
    n_subjects = len(subject_ids)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([n_subjects, n_classes])  # 이건 뭐고
    y = np.zeros([n_subjects, 1])  # 이건 뭐지?

    # Get class labels for all subjects
    for i in range(n_subjects):
        y_data[i, int(labels[subject_ids[i]]) - 1] = 1
        y[i] = int(labels[subject_ids[i]])
        if y[i] == 2:
            y[i] = 0  # TC -> 0으로, ASD는 그대로 1

    ## index
    # PITT(56): 0-55 [0-24, 48, 50-52 :(1) total 29 || 25-47, 49, 53-55:(2) total 27]
    # UM_1(106): 188-293 [188-240 :(1) total 53 || 241-293 :(2) total 53]
    # USM(71): 328-398 [328-352 :(2) total 25 || 353-398 :(1) total 46]
    # YALE(56): 399-454 [399-426 :(2) total 28 || 427-454:(1) total 28]
    # NYU(163+12): 593-755, 767-778 [593-667(75) :(1) total 75 || 668-755(88), 767-778(12) :(2) total 100]
    # UCLA_1(72): 818-889 [818-858 :(1) total 41 ||859-889 :(2) total 31]

    PITT_site_data = []
    PITT_site_label = []
    UM_1_site_data = []
    UM_1_site_label = []
    USM_site_data = []
    USM_site_label = []
    YALE_site_data = []
    YALE_site_label = []
    NYU_site_data = []
    NYU_site_label = []
    UCLA_1_site_data = []
    UCLA_1_site_label = []

    for i in range(len(data)):
        if 0 <= i <= 55:
            PITT_site_data.append(data[i])
            PITT_site_label.append(y[i])
        elif 188<=i<=293:
            UM_1_site_data.append(data[i])
            UM_1_site_label.append(y[i])
        elif 328<=i<=398:
            USM_site_data.append(data[i])
            USM_site_label.append(y[i])
        elif 399<=i<=454:
            YALE_site_data.append(data[i])
            YALE_site_label.append(y[i])
        elif 593<=i<=755 or 767<=i<=778:
            NYU_site_data.append(data[i])
            NYU_site_label.append(y[i])
        elif 818<=i<=889:
            UCLA_1_site_data.append(data[i])
            UCLA_1_site_label.append(y[i])

    return np.array(PITT_site_data), np.array(PITT_site_label), np.array(UM_1_site_data), np.array(UM_1_site_label), \
           np.array(USM_site_data), np.array(USM_site_label), np.array(YALE_site_data), np.array(YALE_site_label), \
           np.array(NYU_site_data), np.array(NYU_site_label), np.array(UCLA_1_site_data), np.array(UCLA_1_site_label)



def data_loader(args, train_x, train_label, valid_x, valid_label):
    # make graph topology with only train data
    topology = t_test(args, ROI=args.numROI, train_data=train_x, train_label=train_label)

    train_static_edge, \
    valid_static_edge = define_node_edge(train_data=train_x, test_data=valid_x, t=topology, p_value=args.p_value,
                                        edge_binary=True, node_binary=False, edge_abs=True, node_abs=False, node_mask=False)

    train_Node_list = torch.FloatTensor(train_x).to(args.device)
    train_A_list = torch.FloatTensor(train_static_edge).to(args.device)
    train_label = torch.LongTensor(train_label).to(args.device)
    train_dataset = []

    for i, sub in enumerate(train_A_list):
        edge, attr = dense_to_sparse(sub)
        train_dataset.append(Data(x=train_Node_list[i], y=train_label[i], edge_index=edge, edge_attr=attr))

    if args.batch_size > len(train_dataset):
        trainLoader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    else:
        trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # load test data
    valid_Node_list = torch.FloatTensor(valid_x).to(args.device)

    valid_A_list = torch.FloatTensor(valid_static_edge).to(args.device)

    valid_label = torch.LongTensor(valid_label).to(args.device)
    valid_dataset = []
    for i, sub in enumerate(valid_A_list):
        edge, attr = dense_to_sparse(sub)
        valid_dataset.append(Data(x=valid_Node_list[i], y=valid_label[i], edge_index=edge, edge_attr=attr))
    validLoader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))  # full batch

    return trainLoader, validLoader