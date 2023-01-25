import torch
import torch.nn as nn
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature

        # we will concat 4 conv layers in foward step. So, we have divide the features properly relating the input size
        feat_out_number = len_feature // 4  #ceiled
        sobra = len_feature % 4    # If 0, will not interfere at anything
        #print(feat_out_number)
        #print(sobra)
        #exit()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=feat_out_number, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(feat_out_number)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=feat_out_number, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(feat_out_number)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=feat_out_number+sobra, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(feat_out_number+sobra)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=feat_out_number, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(feat_out_number, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)
            residual = out

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            out3 = self.conv_3(out)
            out_d = torch.cat((out1, out2, out3), dim = 1)
            out = self.conv_4(out)
            out = self.non_local(out)
            out = torch.cat((out_d, out), dim=1)
            #print("out value: ")
            #print(out.shape)
            #exit()
            out = self.conv_5(out)   # fuse all the features together
            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)

            return out

class Model(nn.Module):
    def __init__(self, n_features, batch_size, ten_crop, device):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 16
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10
        self.ten_crop = ten_crop

        print("OIRRRRRRAAA")
        print(n_features)
        self.Aggregate = Aggregate(len_feature=n_features)
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

        self.device = device

    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        bs, ncrops, t, f = out.size()

        #print("Shape 1")
        #print(out.shape)
        out = out.view(-1, t, f)
        #print("Shape 2")
        #print(out.shape)
        out = self.Aggregate(out)
        #print("Shape 3")
        #print(out.shape)
        out = self.drop_out(out)
        #print("Shape 4")
        #print(out.shape)
        features = out
        scores = self.relu(self.fc1(features))
        #print("Shape 5")
        #print(scores.shape)
        scores = self.drop_out(scores)
        #print("Shape 6")
        #print(scores.shape)        
        scores = self.relu(self.fc2(scores))
        #print("Shape 7")
        #print(scores.shape)        
        scores = self.drop_out(scores)
        #print("Shape 8")
        #print(scores.shape)        
        scores = self.sigmoid(self.fc3(scores))
        #print("Shape 9")
        #print(scores.shape)      

        #print(scores)
        #exit(0)

        if self.ten_crop == "True":  
            # If we have 10 crop, let's calculate the mean between than
            scores = scores.view(bs, ncrops, -1).mean(1)
        else:
            # Otherwise, let's just remove this dimension
            scores = scores.view(bs, ncrops, -1).squeeze(1)

        #print("Shape 10")
        #print(scores.shape)        

        # 10crop true: [1, 113]
        # 10crop false: [1, 113, 1]
        
        scores = scores.unsqueeze(dim=2)

        #print("Dim do score: ")
        #print(scores.shape)


        #print("features shape")
        #print(features.shape)
        
        # features has (1*batch, segments, 2048) if not 10 crop and (10*batch, segments, 2048) otherwise
        if self.ten_crop == "False":
            normal_features = features[0:self.batch_size]    
            abnormal_features = features[self.batch_size:]
        else:
            normal_features = features[0:self.batch_size*10]
            abnormal_features = features[self.batch_size*10:]

        normal_scores = scores[0:self.batch_size]
        abnormal_scores = scores[self.batch_size:]

        # (10*batch*2, 32)
        feat_magnitudes = torch.norm(features, p=2, dim=2)  # Calcula a norma dos vetores de características
        #print("feat_magnitudes shape: ")
        #print(feat_magnitudes.shape)
        #exit()        
        if self.ten_crop == "True":  
            # If we have 10 crop, let's calculate the mean between than
            feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1) # (batch*2, 10, 32) -> (batch*2, 32)
        else:
            # Otherwise, let's just remove this dimension
            feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).squeeze(1)     # (batch*2, 1, 32) -> (batch*2, 32)
            

        # here we have (1 batch, 118 features) de normas
        #print("feat_magnitudes shape: ")
        #print(feat_magnitudes.shape)    
        
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes  # (16, 32)
        n_size = nfea_magnitudes.shape[0]

        #print("nfea_magnitudes shape: ")
        #print(nfea_magnitudes.shape)  
        #print("afea_magnitudes shape: ")
        #print(afea_magnitudes.shape)        
        #print(n_size)    


        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        select_idx = torch.ones_like(nfea_magnitudes).to(self.device)    # Preenche com 1's
        #print("select_idx")
        #print(select_idx)
        select_idx = self.drop_out(select_idx)  # O dropout muda os elementos do array

        #print("select_idx")
        #print(select_idx.shape)
        #exit()


        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx
        #print("afea_magnitudes_drop")
        #print(afea_magnitudes_drop) 
        
        # idx_abn é o indice do vetor de característica com a maior norma
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1] # Pega as k_abn maiores normas dos vetores de características
        #print("k_abn")
        #print(k_abn)    # (16,1)
        #print("idx_abn")
        #print(idx_abn.shape)    # (16,1)
        #exit()
                

        # Expando o vetor para conter 2048 posições, e repito o indice do vetor até encher 2048
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
        #print("idx_abn_feat")
        #print(idx_abn_feat.shape)
        #exit()


        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        #print("abnormal_features")
        #print(abnormal_features.shape)
        #exit()
        
        abnormal_features = abnormal_features.permute(1, 0, 2,3)
        #print("abnormal_features")
        #print(abnormal_features.shape)
        #print("\n\n")
        total_select_abn_feature = torch.zeros(0).to(self.device)
        for abnormal_feature in abnormal_features:  # itera os 10 crop
            #print("Itertou 1 vez")
            #print(abnormal_feature.shape)
            # Pega a feature do segmento idx_abn_feat
            # abnormal_feature é (16, 32, 2048)
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat).to(self.device)   # top 3 features magnitude in abnormal bag
            #print("feat_select_abn")
            #print(feat_select_abn.shape)
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))   

        # As features com maiores normas de cada video
        print("total_select_abn_feature")
        #(16, 1, 2048) ou (160, 1, 204)
        #print(total_select_abn_feature)
        print(total_select_abn_feature.shape)
        
        # idx_abn é o indice do vetor de característica com a maior norma
        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        print("idx_abn_score")
        #print(idx_abn_score)
        print(idx_abn_score.shape)

        # Pego os scores dos segmentos que possuem a maior norma
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude
        print("score_abnormal")
        #print(score_abnormal)
        print(score_abnormal.shape)
        

        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).to(self.device)
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0).to(self.device)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat).to(self.device)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes