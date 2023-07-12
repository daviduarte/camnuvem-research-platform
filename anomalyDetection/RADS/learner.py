import torch
import torch.nn as nn
from torch.nn import functional as F



class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0, ten_crop = False):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).cuda()
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList().cuda()
        self.ten_crop = ten_crop

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, bs, segments, vars=None):

        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1]).cuda()
        x = F.relu(x).cuda()
        x = F.dropout(x, self.drop_p, training=self.training).cuda()
        x = F.linear(x, vars[2], vars[3]).cuda()
        x = F.dropout(x, self.drop_p, training=self.training).cuda()
        x = F.linear(x, vars[4], vars[5]).cuda()

        last = torch.sigmoid(x).cuda()

        # Last tem [30*10*64, 1]. Vamos fazer a média


        if self.ten_crop:
            last = last.view(bs, 10, segments).mean(1).cuda()
            last = last.view(-1, 1).cuda()

        #if self.ten_crop:
        # Lets calculate the final score taking the mean from all 10-crop
        #last.shape = [B, 10, 1]

        return last


    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


