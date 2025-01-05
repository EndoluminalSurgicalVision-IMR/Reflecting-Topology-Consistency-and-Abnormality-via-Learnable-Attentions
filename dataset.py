# -*- coding: utf-8 -*-
"""
Created on Tur Sep 12 11:20:13 2024

@author: Li
"""
import torch
from torch_geometric.data import Data, DataLoader
import os
import numpy as np

def dfs(node, ancestor, adj_list, M):
    # Descendants of node
    M[ancestor][node] = 1
    for child in adj_list[node]:
        dfs(child, ancestor, adj_list, M)

def get_mask(edge, node_num):
    # descendants matrix
    adj_list = [[] for _ in range(node_num)]
    M = np.zeros((node_num, node_num), dtype=int)

    # adjacent map
    for i in range(edge.shape[1]):
        adj_list[edge[0, i]].append(edge[1, i])

    for node in range(node_num):
        dfs(node, node, adj_list, M)
    return M

def floyd(edge_index):
    node_num = np.max(edge_index) + 1
    adj = np.full((node_num, node_num), np.inf)
    for i in range(node_num):
        adj[i, i] = 0
    for idx in range(edge_index.shape[1]):
        adj[edge_index[0][idx]][edge_index[1][idx]] = 1
        adj[edge_index[1][idx]][edge_index[0][idx]] = 1
    a = adj.copy()
    # print(adjacent_matrix)
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if a[i][j] > a[i][k] + a[k][j]:
                    a[i][j] = a[i][k] + a[k][j]
    return a

def generation_dict(x):
    #x[:,0]: generation dim
    node_num = x.shape[0]
    dict = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(node_num):
            dict[i][j] = abs(x[i, 0] - x[j, 0])
    return dict



def multitask_dataset(path_feature, path_top,outlier_mask = True):
    file = os.listdir(path_feature)
    file.sort()
    num = len(file) // 6
    dataset = []

    for i in range(num):
        patient = file[i * 6].split("_")[0]
        x = np.load(path_feature + patient + "_x.npy", allow_pickle=True)
        x_new = x[:, 0:11]
        x = np.concatenate((x_new, x[:, 13:17]), axis=-1)

        edge = np.load(path_feature + patient + "_edge.npy", allow_pickle=True)
        edge_prop = np.load(path_feature + patient + "_edge_feature.npy", allow_pickle=True)
        [y_lobar, y_seg, y_subseg] = np.load(path_feature + patient + "_y.npy", allow_pickle=True)
        # label of abnormal branch: -1

        nodepair = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
             if y_seg[i] == y_seg[j]:
                 nodepair[i, j] = 1

        if outlier_mask:
            mask_outlier = (y_lobar != -1).astype(int)
            y_lobar[y_lobar == -1] = 0
            y_seg[y_seg == -1] = 18
            y_subseg[y_subseg == -1] = 0
        else:
            mask_outlier = np.zeros_like(y_lobar).astype(int)
            y_lobar[y_lobar == -1] = y_lobar.max() + 1
            y_seg[y_seg == -1] = y_seg.max() + 1
            y_subseg[y_subseg == -1] = y_subseg.max() + 1

        #spd = floyd(edge) Shortest Path Distance can be generated through floyd()
        spd = np.load(path_top + patient + "_spd.npy", allow_pickle=True)
        spd = np.where(spd > 29, 29, spd)
        spd = torch.from_numpy(spd).long()
        mask_outlier = torch.from_numpy(mask_outlier).float()

        mask_top = get_mask(edge[:, edge_prop > 0], x.shape[0])
        mask_top = torch.from_numpy(mask_top).long()
        mask_top.requires_grad = False

        gen = torch.from_numpy(generation_dict(x)).long()
        x = (torch.from_numpy(x)).float()

        y_subseg = (torch.from_numpy(y_subseg)).float()
        y_seg = (torch.from_numpy(y_seg)).float()
        y_lobar = (torch.from_numpy(y_lobar)).float()
        nodepair = (torch.from_numpy(nodepair)).float()

        data = Data(x=x, y_lobar=y_lobar, y_seg=y_seg, y_subseg=y_subseg,
                    patient=patient, gen=gen,spd=spd,edge = edge[:,edge_prop>0],mask_outlier = mask_outlier,mask_top = mask_top, nodepair = nodepair)

        if x.shape[0] == y_subseg.shape[0]:
            dataset.append(data)
        else:
            print(file[i * 6])
    return dataset





