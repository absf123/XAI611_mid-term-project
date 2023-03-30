import numpy as np
import torch
from scipy import stats


def flatten_fc(data):
    if isinstance(data, torch.Tensor):
        data = data.unsqueeze(0) if int(data.ndim) == 2 else data
    else:
        data = np.expand_dims(data, axis=0) if int(data.ndim) == 2 else data
    x, y = np.triu_indices(data.shape[1], k=1)
    FC_flatten = data[:, x, y]
    return FC_flatten


def flatten2dense(flatten, ROI):
    if isinstance(flatten, torch.Tensor):
        flatten = flatten.unsqueeze(0) if int(flatten.ndim) == 1 else flatten
    else:
        flatten = np.expand_dims(flatten, axis=0) if int(flatten.ndim) == 1 else flatten
    B, F = flatten.shape
    x, y = np.triu_indices(ROI, k=1)
    if isinstance(flatten, torch.Tensor):
        sym = torch.zeros([B, ROI, ROI], device=flatten.device)
        sym[:, x, y] = flatten
        sym = sym + torch.transpose(sym, -1, -2)
    else:
        sym = np.zeros((B, ROI, ROI))
        sym[:, x, y] = flatten
        sym = sym + np.transpose(sym, (0, 2, 1))
    return sym

# t-test for pValueMasking
def t_test(args, ROI=200, train_data=None, train_label=None):
    flatten = flatten_fc(train_data)
    ASDflatten = flatten[np.squeeze(train_label) == 1]  # 1 : ASD vs 0 : TC
    TCflatten = flatten[np.squeeze(train_label) == 0]
    p_value = stats.ttest_ind(ASDflatten, TCflatten, equal_var=False).pvalue
    p_value = flatten2dense(p_value, ROI)
    p_value[p_value > args.p_value] = 0  # p_value가 큰 값은 0으로 masking: p_value가 낮을수록 유의미한 연결관계
    p_value[p_value!=0] = 1
    print(f"Edge ratio: {round((p_value.sum() / flatten.shape[1])*100, 5)}%")

    return p_value

# edge topology mask
def pValueMasking(feature, t_test, p_value, binary=False, abs=True):
    data = feature.copy()
    mask = t_test <= p_value
    if abs:
        data = np.abs(data)
    data = data * mask
    if binary:
        data[data != 0] = 1
    return data

# define edge
def define_node_edge(train_data, test_data, t, p_value, edge_binary, node_binary, edge_abs, node_abs, node_mask=False):
    train_static_edge = pValueMasking(feature=train_data, t_test=t, p_value=p_value, binary=edge_binary, abs=edge_abs)
    test_static_edge = pValueMasking(fxeature=test_data, t_test=t, p_value=p_value, binary=edge_binary, abs=edge_abs)

    # node masking
    # if not node_mask:
    #     train_node_feature = pValueMasking(feature=train_data, t_test=t, p_value=p_value, binary=node_binary, abs=node_abs)
    #     valid_node_feature = pValueMasking(feature=valid_data, t_test=t, p_value=p_value, binary=node_binary, abs=node_abs)
    #     test_node_feature = pValueMasking(feature=test_data, t_test=t, p_value=p_value, binary=node_binary, abs=node_abs)
    # else:
    #     train_node_feature = train_data.copy()
    #     valid_node_feature = valid_data.copy()
    #     test_node_feature = test_data.copy()

    return train_static_edge, test_static_edge