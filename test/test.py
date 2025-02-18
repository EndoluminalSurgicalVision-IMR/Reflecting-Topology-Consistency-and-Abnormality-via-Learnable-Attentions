# -*- coding: utf-8 -*-
"""
Created on Tur Sep 12 11:25:42 2024

@author: Li
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os
import numpy as np

import sys
import pickle
sys.path.append("..")
from utils import calculate_accuracy, calculate_CS, calculate_td, calculate_abnormal
from models.network import our_net
from data_process.dataset import multitask_dataset
from config.config import config, DATA_PATHS



seed = config["seed"]
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
epochs = config["epochs"]
threshold = config["threshold"]


SAVE_DIR_TEMPLATE = config["SAVE_DIR_TEMPLATE"]

dataset3 = multitask_dataset(DATA_PATHS['test'], DATA_PATHS['top_test'], outlier_mask= True)
test_loader_case = DataLoader(dataset3, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
test_accuracy = [[] for _ in range(3)]
test_abnormal = [[] for _ in range(3)]
test_consistency = [[] for _ in range(2)]
test_td = [[] for _ in range(2)]

my_net = our_net(input_dim=config["input_dim"], num_classes1=config["num_classes1"], num_classes2=config["num_classes2"],
                 num_classes3=config["num_classes3"], dim=config["dim"], heads=config["heads"], mlp_dim=config["mlp_dim"],
                 dim_head = config["dim_head"], dropout= config["dropout"], trans_depth = config["trans_depth"], outlier_depth = config["outlier_depth"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_net = my_net.to(device)
save_dir = SAVE_DIR_TEMPLATE
checkpoint = torch.load(os.path.join(save_dir, 'best.ckpt'))
my_net.load_state_dict(checkpoint['state_dict'])
my_net.eval()

for case in test_loader_case:
    x = case.x.to(device)
    y_labels = [case.y_lobar.to(device).long(), case.y_seg.to(device).long(),
                case.y_subseg.to(device).long()]
    spd = case.spd.to(device)
    gen = case.gen.to(device)
    patient = case.patient[0]
    mask_outlier = case.mask_outlier.cpu().data.numpy()
    mask_top = case.mask_top.to(device)

    outputs = my_net(x,mask_top,spd.detach(),0.)

    for i in range(3):  # output12, output22, output32
        acc = calculate_accuracy(outputs[i + 3], y_labels[i])
        test_accuracy[i].append(acc)

    if mask_outlier.sum() != mask_outlier.shape[0]:
        precision_ab, recall_ab, f1_ab = calculate_abnormal(outputs[-1], mask_outlier, threshold)
        test_abnormal[0].append(precision_ab)
        test_abnormal[1].append(recall_ab)
        test_abnormal[2].append(f1_ab)

    cs_seg, cs_sub = calculate_CS(outputs[4], outputs[5], mask_top.cpu().data.numpy())
    test_consistency[0].append(cs_seg)
    test_consistency[1].append(cs_sub)

    td_seg = calculate_td(spd.cpu().data.numpy(), outputs[4].max(dim=1)[1].cpu().data.numpy(), y_labels[1].cpu().data.numpy())
    test_td[0].append(td_seg)
    td_sub = calculate_td(spd.cpu().data.numpy(), outputs[5].max(dim=1)[1].cpu().data.numpy(), y_labels[2].cpu().data.numpy())
    test_td[1].append(td_sub)

test_accuracy = [np.mean(np.array(acc_list)) for acc_list in test_accuracy]

print(
    "Accuracy of Test Samples: Stage2_lob: {}, Stage2_seg: {}, Stage2_sub:{}"
    .format(test_accuracy[0],
            test_accuracy[1],
            test_accuracy[2],
            ))

test_consistency = [np.mean(np.array(cs_list)) for cs_list in test_consistency]
test_abnormal = [np.mean(np.array(ab_list)) for ab_list in test_abnormal]
test_td = [np.mean(np.array(td_list)) for td_list in test_td]


print(
    "Consistency of Test Samples: Stage2_seg: {}, Stage2_sub:{}"
    .format(test_consistency[0],
            test_consistency[1],
            ))


print(
    "Topology distance of Test Samples: Stage2_seg: {}, Stage2_sub:{}"
    .format(test_td[0],
            test_td[1],
            ))
print(
    "Abnormal detection of Test Samples: Precision: {}, Recall:{}, F1 score:{}"
    .format(test_abnormal[0],
            test_abnormal[1],
            test_abnormal[2],
            ))

