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
from network import our_net
import time
import shutil
import sys
import pickle
from dataset import multitask_dataset

sys.path.append("..")
from loss_functions import LabelSmoothCrossEntropyLoss
from utils import *

def calculate_loss(outputs, labels, mask_outlier, nodepair_labels):
    loss_function = LabelSmoothCrossEntropyLoss(smoothing=0.01)
    classification_loss = sum(
        loss_function(output, label, mask_outlier)
        for output, label in zip(outputs[0:3], labels)
    ) + sum(
        loss_function(output, label, mask_outlier)
        for output, label in zip(outputs[3:6], labels)
    )
    outlier_loss = BCEcost(outputs[8], mask_outlier) + BCEcost(outputs[9], mask_outlier)
    nodepair_loss = BCEcost(outputs[6], nodepair_labels) + BCEcost(outputs[7], nodepair_labels)
    return classification_loss + outlier_loss + nodepair_loss

def calculate_accuracy(output, labels):
    preds = output.max(dim=1)[1].cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    return np.sum((labels == preds).astype(np.uint8)) / labels.shape[0]


seed = 222
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed(seed)  # gpu
np.random.seed(seed)  # numpy
train_path = "/home/yuy/code/ATM_refine2409/feature/"
val_path = "/home/yuy/code/ATM_refine2409/feature_val/"
test_path = "/home/yuy/code/AIIB/anno_refine/feature/"
top_train = "/home/yuy/code/ATM_refine2409/toplogy/"
top_test = "/home/yuy/code/AIIB/anno_refine/toplogy/"
epochs = 600

DATA_PATHS = {
    "train": "",
    "val": "",
    "test": "",
    "top_train": "",
    "top_test": "",
    "top_val": "",
}
SAVE_DIR_TEMPLATE = ""
if not os.path.exists(SAVE_DIR_TEMPLATE):
    os.makedirs(SAVE_DIR_TEMPLATE)

dataset1 = multitask_dataset(DATA_PATHS["train"], path_top= DATA_PATHS["top_train"], outlier_mask= True)
train_loader_case = DataLoader(dataset1, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
dataset2 = multitask_dataset(DATA_PATHS["val"], path_top= DATA_PATHS["top_val"], outlier_mask= True)
val_loader_case = DataLoader(dataset2, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
dataset3 = multitask_dataset(DATA_PATHS["test"], path_top= DATA_PATHS["top_test"], outlier_mask= False)
test_loader_case = DataLoader(dataset3, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

max_acc = 0

logfile = os.path.join(SAVE_DIR_TEMPLATE, 'log')
sys.stdout = Logger(logfile)
pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
for f in pyfiles:
    shutil.copy(f, os.path.join(SAVE_DIR_TEMPLATE, f))

my_net = our_net(input_dim=15, num_classes1=6, num_classes2=21, num_classes3=134, dim=128, heads=4,
                             mlp_dim=256, dim_head=32, dropout=0., trans_depth = 2, outlier_depth = 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_net = my_net.to(device)
optimizer = torch.optim.Adam(my_net.parameters(), lr=5e-4, eps=1e-4)
BCEcost = nn.BCELoss(reduction='mean').to(device)


for epoch in range(epochs):
    my_net.train()
    time1 = time.time()
    train_accuracy = [[] for _ in range(6)]
    test_accuracy = [[] for _ in range(6)]
    train_loss = []

    for case in train_loader_case:
        x = case.x.to(device)
        spd = case.spd.to(device)
        gen = case.gen.to(device)
        mask_outlier = case.mask_outlier.to(device).detach()
        mask_top = case.mask_top.to(device).detach()
        nodepair_label = case.nodepair.to(device).detach()

        y_labels = [case.y_lobar.to(device).long(), case.y_seg.to(device).long(), case.y_subseg.to(device).long()]

        optimizer.zero_grad()
        #output11, output21, output31, output12, output22, output32,nodepair_1,nodepair_2,outlier_1,outlier_2
        outputs = my_net(x,mask_top, spd.detach(),0.1)

        loss = calculate_loss(outputs, y_labels, mask_outlier, nodepair_label)
        loss.backward()

        for i in range(3):  # output11, output21, output31
            acc = calculate_accuracy(outputs[i], y_labels[i])
            train_accuracy[i].append(acc)
        for i in range(3):  # output12, output22, output32
            acc = calculate_accuracy(outputs[i + 3], y_labels[i])
            train_accuracy[i+ 3].append(acc)
        train_loss.append(loss.item())

        optimizer.step()

    train_accuracy = [np.mean(np.array(acc_list)) for acc_list in train_accuracy]
    train_mean_loss = np.mean(np.array(train_loss))

    print(
        "epoch:{},loss:{}，acc: Stage1_lob: {}, Stage1_seg: {}, Stage1_sub: {}, Stage2_lob: {}, Stage2_seg: {}, Stage2_sub:{} time:{}"
        .format(epoch + 1, train_mean_loss,
                                                                    train_accuracy[0],
                                                                    train_accuracy[1],
                                                                    train_accuracy[2],
                                                                    train_accuracy[3],
                                                                    train_accuracy[4],
                                                                    train_accuracy[5],
                                                                    time.time() - time1))


    if (epoch + 1) % 10 == 0:
        my_net.eval()
        for case in val_loader_case:
            x = case.x.to(device)
            y_labels = [case.y_lobar.to(device).long(), case.y_seg.to(device).long(),
                        case.y_subseg.to(device).long()]
            spd = case.spd.to(device)
            gen = case.gen.to(device)
            patient = case.patient[0]
            mask_outlier = case.mask_outlier.cpu().data.numpy()
            mask_top = case.mask_top.to(device)

            outputs = my_net(x,mask_top,spd.detach(),0.)

            for i in range(3):  # output11, output21, output31
                acc = calculate_accuracy(outputs[i], y_labels[i])
                test_accuracy[i].append(acc)
            for i in range(3):  # output12, output22, output32
                acc = calculate_accuracy(outputs[i + 3], y_labels[i])
                test_accuracy[i + 3].append(acc)

        test_accuracy = [np.mean(np.array(acc_list)) for acc_list in test_accuracy]

        print(
            "Accuracy of Val Samples: Stage1_lob: {}, Stage1_seg: {}, Stage1_sub: {}, Stage2_lob: {}, Stage2_seg: {}, Stage2_sub:{}"
            .format(test_accuracy[0],
                    test_accuracy[1],
                    test_accuracy[2],
                    test_accuracy[3],
                    test_accuracy[4],
                    test_accuracy[5],
                    ))
        if test_accuracy[-1] > max_acc:
            max_acc = test_accuracy[-1]
            state_dict = my_net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch + 1,
                'save_dir': SAVE_DIR_TEMPLATE,
                'state_dict': state_dict},
                os.path.join(SAVE_DIR_TEMPLATE, 'best.ckpt'))

    if (epoch + 1) % 100 == 0:
        state_dict = my_net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch + 1,
            'save_dir': SAVE_DIR_TEMPLATE,
            'state_dict': state_dict},
            os.path.join(SAVE_DIR_TEMPLATE, '%04d.ckpt' % (epoch + 1)))
