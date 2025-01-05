# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:04:18 2024

@author: Li
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss
    
    

class LabelSmoothCrossEntropyLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        
        self.adr = 0
        self.w = 0.001
        # self.w1 = 0.01
        # self.kl = 0
        self.kl_loss = torch.nn.KLDivLoss(reduction='mean')
        self.hw = 0.001
        self.l1_loss = torch.nn.L1Loss(reduction = "mean")
        self.l2_loss = torch.nn.MSELoss(reduction = 'mean')
        self.sml1_loss = torch.nn.SmoothL1Loss(reduction = 'mean')

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets,mask):
        # sl2 = torch.norm(S, dim = 1)
        # l = torch.argmax(sl2)
        # for i in range(l,2):
        #     if sl2[i]>sl2[i+1]:
        #         self.adr += (torch.norm((S[i].detach()-S[i+1]))).pow(2)
                
        # for j in range(2):
        #     self.kl += self.kl_loss(O[j], O[2])
        
        # if xh_label.shape[0]==0:
        #     h_loss = 0
        # else:
        #     # h_loss = (-((xh_label*F.log_softmax(xh, -1)).sum(-1))).mean()
        #     # h_loss = self.sml1_loss(xh/(torch.sum(xh,dim = 1,keepdim = True)+0.01),xh_label)
        #     h_loss = self.kl_loss(F.softmax(xh, -1),xh_label)
        
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(1)

        loss = -(targets * lsm).sum(-1)
        loss = loss * mask.float()

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()


        return loss

class DependenceLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean', alpha = 0.001,p_loss1=1,p_loss2=1):
        super().__init__(weight=weight, reduction=reduction)
        self.alpha = alpha
        self.p_loss1 = p_loss1
        self.p_loss2 = p_loss2

    @staticmethod
    def _softargmax(input, beta=10000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        *_, n = input.shape
        input = nn.functional.softmax(beta * input, dim=-1).to(device)
        indices = torch.linspace(0, 1, n).to(device)
        result = torch.sum((n - 1) * input * indices, dim=-1).to(device)
        return result


    def _check_consistence(self,previous_level:torch.Tensor, current_level:torch.Tensor,hierarchy_book):

        with torch.no_grad():
            current_pre = torch.ones_like(current_level)
            for i in range(current_level.shape[0]):
                current_pre[i] = hierarchy_book[int(current_level[i].item())]
            #bool_tensor = [(previous_level[i]!= hierarchy_book[int(current_level[i].item())]).item() for i in range(current_level.shape[0])]
            bool_tensor = [(previous_level[i] != current_pre[i]).item() for i in range(current_pre.shape[0])]
            bool_tensor=numpy.array(bool_tensor)
            bool_tensor[previous_level.cpu().numpy() == 18] = False
            bool_tensor[previous_level.cpu().numpy() == 19] = False
        #print("previous",previous_level,"cur_pre",current_pre,bool_tensor)

        return torch.FloatTensor(bool_tensor)



    def forward(self, out_pre,out_cur,label_pre,label_cur,hierarchy_book):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_pre = self._softargmax(out_pre).requires_grad_(True)
        pred_cur = self._softargmax(out_cur).requires_grad_(True)

        pred_cur.retain_grad()
        pred_pre.retain_grad()


        D_l = self._check_consistence(pred_pre, pred_cur,hierarchy_book).to(device)
        function_s = nn.Sigmoid()

        l_prev = 2 * (function_s(10000 * (pred_pre - label_pre).abs()) - 0.5).requires_grad_(True)
        l_prev.retain_grad()

        l_curr = 2 * (function_s(1000 * (pred_cur - label_cur).abs()) - 0.5).requires_grad_(True)
        l_curr.retain_grad()
        #print("pred_pre",pred_pre,"label_pre",label_pre,l_prev)
        #print("pred_cur", pred_cur, "label_cur", label_cur,l_prev)

        #dloss = self.alpha*torch.sum(torch.pow(self.p_loss1, D_l*l_prev)*torch.pow(self.p_loss2, D_l*l_curr) - 1)
        dloss = self.alpha*torch.sum(torch.pow(self.p_loss1, D_l*l_prev)*torch.pow(self.p_loss2, D_l*l_curr) - 1)
        dloss.requires_grad_(True)

        return dloss