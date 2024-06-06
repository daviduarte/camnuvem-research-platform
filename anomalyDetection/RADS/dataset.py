import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import sys
sys.path.append(r'/media/denis/dados/CamNuvem/pesquisa/anomalyDetection/RTFM')
from utils import process_feat

def adjustSegments(vec, ten_crop):

    if ten_crop == False:
            return process_feat(vec, 32)

    vec = vec.transpose(1, 0, 2)  # [10, B, T, F]
    
    new = []
    for i in vec:
        i = process_feat(i, 32) # pass [N, 2048], return [32, 2048]
        new.append(i)    
    new = np.stack(new, axis=0)    


    return torch.from_numpy(new)


class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    is_ucf = Workaround because this script needs equal number of samples in all .list files. UCF files has 10 more samples in one of then
    """
    def __init__(self, path, is_train=1, ten_crop = False, is_ucf = False):
        super(Normal_Loader, self).__init__()

        self.ten_crop = ten_crop
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            
            if is_ucf:
                self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:

            import time
            start = time.time()            

            data_list = os.path.join(self.path, self.data_list[idx][:-5]+'.npy')
            rgb_npy = np.load(data_list)
            
            #flow_npy = np.load(os.path.join(self.path, self.data_list[idx][:-5]+'.npy'))
            #concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            #rgb_npy = torch.sigmoid(torch.from_numpy(rgb_npy)).numpy()
            #rgb_npy = torch.nn.functional.normalize(torch.from_numpy(rgb_npy)).numpy()
            #rgb_npy = process_feat(rgb_npy, 32)
            rgb_npy = adjustSegments(rgb_npy, self.ten_crop)    # [10, 32, 2048] or [32, 2048]
            #print(rgb_npy.shape)
            end = time.time()         
            #print("Tempo decorrido no carregamento dos dados em trainign: " + str(end-start))
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            
            data_list = os.path.join(self.path, name[:-4] + '.npy')
            rgb_npy = np.load(data_list)

            #flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            #concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            #rgb_npy = torch.sigmoid(torch.from_numpy(rgb_npy)).numpy()
            #rgb_npy = torch.nn.functional.normalize(torch.from_numpy(rgb_npy)).numpy()
            rgb_npy = adjustSegments(rgb_npy, self.ten_crop)
            #print(rgb_npy.shape)
            return rgb_npy, gts, frames

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    is_ucf = Workaround because this script needs equal number of samples in all .list files. UCF files has 10 more samples in one of then
    """
    def __init__(self, path, is_train=1, ten_crop = False, is_ucf = False):
        super(Anomaly_Loader, self).__init__()
        
        self.is_train = is_train
        self.path = path
        self.ten_crop = ten_crop
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')

            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)

            if is_ucf:
                self.data_list = self.data_list[:-10]                
        else:       
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        if self.is_train == 1:
            import time
            start = time.time()
            rgb_npy = np.load(os.path.join(self.path, self.data_list[idx][:-5]+'.npy'))

            #flow_npy = np.load(os.path.join(self.path, self.data_list[idx][:-5]+'.npy'))
            #concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            #rgb_npy = torch.sigmoid(torch.from_numpy(rgb_npy)).numpy()
            #rgb_npy = torch.nn.functional.normalize(torch.from_numpy(rgb_npy)).numpy()
            rgb_npy = adjustSegments(rgb_npy, self.ten_crop)
            #print(rgb_npy.shape)
            end = time.time()
            #print("Tempo decorrido no carregamento dos dados em trainign: " + str(end-start))
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path, name[:-4] + '.npy'))

            #flow_npy = np.load(os.path.join(self.path, name + '.npy'))
            #concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            #rgb_npy = torch.sigmoid(torch.from_numpy(rgb_npy)).numpy()
            #rgb_npy = torch.nn.functional.normalize(torch.from_numpy(rgb_npy)).numpy()
            rgb_npy = adjustSegments(rgb_npy, self.ten_crop)
            #print(rgb_npy.shape)

            return rgb_npy, gts, frames

if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))
    #print(loader[1], loader2[1])
