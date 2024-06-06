import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torch.nn import L1Loss
from torch.nn import MSELoss



def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    # Implementation Reference: http://vast.uccs.edu/~adhamija/blog/Caffe%20Custom%20Layer.html
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(- torch.abs(x))
        return torch.abs(torch.mean(- x * target + torch.clamp(x, min=0) + torch.log(tmp)))

def countZeros(vet):
    cont = 0
    vet = vet.tolist()

    for i in vet:
        if i == 0:
            cont += 1
    return cont

def countOnes(vet):
    cont = 0
    vet = vet.tolist()
    for i in vet:
        if i == 1:
            cont += 1
    return cont



class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin, contzera, device):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid().to(device)
        self.mae_criterion = SigmoidMAELoss().to(device)
        self.criterion = torch.nn.BCELoss().to(device)
        self.contzera = contzera
        self.device = device

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze().to(self.device)

        label = label.to(self.device)

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

     
        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)


        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total


def train(nloader, aloader, model, batch_size, optimizer, viz, device, contzera):
    with torch.set_grad_enabled(True):
        model.train()
        model.to(device)
        model.device = device

        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)  # b*32  x 2048

        scores = scores.view(batch_size * 32 * 2, -1)

        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        # Original / AUC 0.591
        actual_alpha = 0.0001  
        actual_margin = 100


        sparsity_hp = 8e-3
        smooth_hp = 8e-4

        loss_criterion = RTFM_loss(float(actual_alpha), float(actual_margin), contzera, device)
        loss_sparse = sparsity(abn_scores, batch_size, sparsity_hp)
        loss_smooth = smooth(abn_scores, smooth_hp)
        cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse

        viz.plot_lines('loss', cost.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


