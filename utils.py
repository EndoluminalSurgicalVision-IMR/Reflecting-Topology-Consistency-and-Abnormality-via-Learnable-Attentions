# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:23:36 2021

@author: Yu
"""

import sys
import numpy as np
import math
from sklearn.metrics import precision_recall_fscore_support

left = np.array([11,3,4,5,6,8,9,10])

def Class2Anno(classes):
    Anno = np.zeros((classes.shape[0],3))
    for i in range(classes.shape[0]):
        class_i = classes[i]
        if 1<= class_i<= 56:
            Anno[i,0] = 1
            Anno[i,1] = left[math.ceil(class_i/7) -1]
            Anno[i,2] = (class_i - 1)%7
        if 57 <= class_i <= 126:
            Anno[i, 0] = 2
            Anno[i, 1] = math.ceil((class_i - 56)/7)
            Anno[i, 2] = (class_i - 1)%7
        if 126 < class_i :
            Anno[i, 0] = 1
            Anno[i, 1] = 7
            Anno[i, 2] = class_i -127
    return Anno

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def calculate_accuracy(output, labels):
    preds = output.max(dim=1)[1].cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    return np.sum((labels == preds).astype(np.uint8)) / labels.shape[0]


def calculate_abnormal(output, labels, threshold):
    output = 1 - output.cpu().data.numpy()
    labels = 1 - labels
    pred_abnormal = (output > threshold).astype(np.uint8)
    precision_ab_i, recall_ab_i, f1_ab_i, _ = precision_recall_fscore_support(labels, pred_abnormal,
                                                                              average="binary")
    return precision_ab_i, recall_ab_i, f1_ab_i

def calculate_CS(output_seg, output_sub, mask_top):
    pred_seg = output_seg.max(dim=1)[1].cpu().data.numpy()
    pred_sub = output_sub.max(dim=1)[1].cpu().data.numpy()

    top_seg = np.zeros(mask_top.shape[0])
    cal_seg = np.zeros(mask_top.shape[0])
    for i in range(mask_top.shape[0]):
        if pred_seg[i] == 18:
            continue
        if pred_seg[mask_top[i, :] == 1].shape[0] > 1:
            cal_seg[i] = 1
            if np.all(pred_seg[mask_top[i, :] == 1] == pred_seg[i]):
                top_seg[i] = 1
    Anno = Class2Anno(pred_sub)
    top_sub = np.zeros(pred_sub.shape[0])
    cal_sub = np.zeros(pred_sub.shape[0])
    for i in range(pred_sub.shape[0]):
        if pred_sub[i] == 0:
            continue
        if pred_sub[mask_top[i, :] == 1].shape[0] > 1:
            cal_sub[i] = 1
            if np.all(Anno[mask_top[i, :] == 1, 0] == Anno[i, 0]) and np.all(
                    Anno[mask_top[i, :] == 1, 1] == Anno[i, 1]):
                if 1 <= Anno[i, 2] <= 3:
                    if np.all(Anno[mask_top[i, :] == 1, 2] == Anno[i, 2]):
                        top_sub[i] = 1
                if Anno[i, 2] == 0:
                    top_sub[i] = 1
                if Anno[i, 2] == 4:
                    if np.all((Anno[mask_top[i, :] == 1, 2] == 1) | (Anno[mask_top[i, :] == 1, 2] == 2)):
                        top_sub[i] = 1
                if Anno[i, 2] == 5:
                    if np.all((Anno[mask_top[i, :] == 1, 2] == 2) | (Anno[mask_top[i, :] == 1, 2] == 3)):
                        top_sub[i] = 1
                if Anno[i, 2] == 6:
                    if np.all((Anno[mask_top[i, :] == 1, 2] == 1) | (Anno[mask_top[i, :] == 1, 2] == 3)):
                        top_sub[i] = 1
    return np.sum(top_seg) / np.sum(cal_seg), np.sum(top_sub) / np.sum(cal_sub)

def calculate_td(spd, label, gt):
    """
    计算每个节点 i 的 td[i]，表示从节点 i 到满足 gt 和 label[i] 一致的节点的最短路径距离的最小值。

    Args:
        spd (np.ndarray): 矩阵，spd[i, j] 表示节点 i 到节点 j 的最短路径距离。
        label (np.ndarray): 一维数组，label[i] 表示节点 i 的预测标签。
        gt (np.ndarray): 一维数组，gt[i] 表示节点 i 的真值。

    Returns:
        np.ndarray: 一维数组 td，td[i] 是节点 i 到满足条件的节点的最短距离的最小值。
    """
    n = spd.shape[0]
    td = np.full(n, 30)  # 初始化 td 数组为无穷大

    for i in range(n):
        # 找到满足 gt 和 label[i] 一致的节点
        target_nodes = np.where(gt == label[i])[0]

        if len(target_nodes) > 0:
            # 从 spd 中获取节点 i 到 target_nodes 的最短路径距离
            td[i] = np.min(spd[i, target_nodes])
    return td.mean()