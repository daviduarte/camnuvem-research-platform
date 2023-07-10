from torch.utils.data import Dataset
from .utils import process_feat
import pickle
import numpy as np
import os
import torch 
import h5py
from sklearn.preprocessing import normalize


from random import randint

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, videos_pkl, in_file, ten_crop):
        super(dataset_h5, self).__init__()
        f2 = open(videos_pkl,"r")
        videos = f2.readlines()
        self.ten_crop = ten_crop
        self.__avid__=[]
        self.__nvid__=[]
        for v in videos:
            """
            if 'var' in videos_pkl:
                if 'Normal' in v.strip().split('/')[-1][:-4]:   # Get only the name, without the ".mp4"
                    self.__nvid__.append(v.strip().split('/')[-1].split(' ')[0])    # Get the name, with the '.mp4'
                else:
                    self.__avid__.append(v.strip().split('/')[-1].split(' ')[0])
            else:
                if 'Normal' in v.strip().split('/')[-1][:-4]:
                    self.__nvid__.append(v.strip().split('/')[-1][:-4])
                else:
                    self.__avid__.append(v.strip().split('/')[-1][:-4])
            """
            if 'var' in videos_pkl:
                if "/normal/" in v:
                #if 'Normal' in v.strip().split('/')[-1][:-4]:   # Get only the name, without the ".mp4"
                    self.__nvid__.append("normal_"+v.strip().split('/')[-1].split(' ')[0])    # Get the name, with the '.mp4'
                else:
                    self.__avid__.append("anomaly_"+v.strip().split('/')[-1].split(' ')[0])
            else:
                if "/normal/" in v:
                    #if 'Normal' in v.strip().split('/')[-1][:-4]:
                    self.__nvid__.append("normal_"+v.strip().split('/')[-1][:-4])
                else:
                    self.__avid__.append("anomaly_"+v.strip().split('/')[-1][:-4])

        self.file = h5py.File(in_file, 'r')
        #self.mask_file = h5py.File(m_file, 'r')

        #"""
        #print(self.file.keys())
        #print(self.__nvid__)
        """
        data = self.file.get('dataset_1')
        data = np.array(data)
        print(self.file['data1'])
        print(self.file['data1'].shape)
        #print(data)

        print(self.__avid__[0])
        print("*")
        print(self.__nvid__)
        print("*")

        
        exit()
        """

    def __getitem__(self, index):
        normalAnomalyDivision = 810    # UCF
        #normalAnomalyDivision = 433     # CamNuvem
        nvid = self.__nvid__[index]
        #print(nvid)
        ind = (index+randint(0, normalAnomalyDivision-1))%normalAnomalyDivision
        avid = self.__avid__[ind]
        
        feas = []
        preds = []
        from sklearn.preprocessing import Normalizer
        tmp = self.file[avid]

        # Gambiarra apenas para testes. Vamos modelar os dados adequadamente
        #tmp = tmp[0:32:,0,0:1024]

        #tmp = normalize(tmp, axis=1)
        #torch.nn.functional.normalize(torch.from_numpy(tmp), dim=1)
        # Mesmo método feito pelo RTFM
        if self.ten_crop:
            tmp = np.transpose(tmp, axes=(1, 0, 2))  # gives a [10, B, F] tensor
            divided_features = []
            for feature in tmp:        # For each 10-crop
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)     

            tmp = np.transpose(divided_features, axes=(1, 0, 2))  # returns to [B, 10, F] tensor
            #print("POrra")
            #print(tmp.shape)
            ano_fea = tmp

            preds.extend([1 for i in range(10)])
        else:
            # Suponha um vetor de dimensões (117, 1024)
            if tmp.shape[0]%32 ==0: # 114 não é múltiplo de 32
                fea_new = np.reshape(tmp, (32,-1, tmp.shape[1]))
            else:
                # então cai aqui
                feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1])) # Faço um resize com as próprias dimensões do vetor original? Por q caraios isso?
                add =  (tmp.shape[0]//32+1) * 32 - tmp.shape[0]     # Quantas vezes 32 cabe dentro de 117? Se ta aqui é pq vai dar quebrado, então adiciona mais 1 pq estamos trabalhando com inteiro. Depois Por ex, 4. Depois subtrai o tamanho da dimensão original para ver quanto falta pra atingir um múltiplo de 32. No final, add = 11
                add_fea = np.tile(feat[-1], (add, 1))   # VOu repetir o vetor de características feat[-1] 11 vezes e adicionar da dimensão 1
                fea_new = np.concatenate((feat, add_fea), axis=0)   # Aqui eu simplesmente copiei o feat[-1] 11 vezes e concatenei em feat, para resultar em um múltiplo de 32
                fea_new = np.reshape(fea_new, (32,-1, tmp.shape[1]))

            ano_fea = fea_new.mean(axis=1)

            preds.append(1) 


        #print("Printando o shape do ano_fea")
        #print(ano_fea.shape)
        # ano_fea = Normalizer(norm='l2').fit_transform(ano_fea)
            


        tmp = self.file[nvid]
        # Gambiarra apenas para testes. Vamos modelar os dados adequadamente
        #tmp = tmp[0:32:,0,0:1024]        


        #tmp = normalize(tmp, axis=1)
        if self.ten_crop:
            tmp = np.transpose(tmp, axes=(1, 0, 2))  # gives a [10, B, F] tensor
            divided_features = []
            for feature in tmp:        # For each 10-crop
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)     

            tmp = np.transpose(divided_features, axes=(1, 0, 2))  # returns to [B, 10, F] tensor
            nor_fea = tmp

            preds.extend([0 for i in range(10)])
        else:        

            if tmp.shape[0]%32 ==0:
                fea_new = np.reshape(tmp, (32,-1, tmp.shape[1]))
            else:
                feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1]))
                add =  (tmp.shape[0]//32+1) * 32 - tmp.shape[0]
                add_fea = np.tile(feat[-1],(add, 1))
                fea_new = np.concatenate((feat, add_fea), axis=0)
                fea_new = np.reshape(fea_new, (32,-1, tmp.shape[1]))

            nor_fea = fea_new.mean(axis=1)

            preds.append(0)


        ano_feas = torch.from_numpy(ano_fea)
        nor_feas = torch.from_numpy(nor_fea)
    

        preds = torch.Tensor(preds)


        return ano_feas, nor_feas, preds

    def __len__(self):
        return min(len(self.__avid__),len(self.__nvid__))

class dataset_h5_test(torch.utils.data.Dataset):
    def __init__(self, videos_pkl, in_file, ten_crop, only_anomaly = False):
        super(dataset_h5_test, self).__init__()
        f2 = open(videos_pkl,"r")
        videos = f2.readlines()
        self.only_anomaly = only_anomaly
        self.ten_crop = ten_crop
        self.__vid__=[]

        if self.only_anomaly == True:
            #videos = videos[0:49]       # CamNuvem
            videos = videos[0:140]  # UCF
        print(len(videos))


        for v in videos:

            #print(v)
            if "/normal/" in v:
            #if 'Normal' in v.strip().split('/')[-1][:-4]:   # Get only the name, without the ".mp4"
                self.__vid__.append("normal_"+v.strip().split('/')[-1][:-4])    # Get the name, with the '.mp4'
            else:
                self.__vid__.append("anomaly_"+v.strip().split('/')[-1][:-4])

            #self.__vid__.append(v.strip().split('/')[-1].split(' ')[0])
        print(in_file)
        self.file = h5py.File(in_file, 'r')

        #print(self.file.keys())
        # import pdb;pdb.set_trace()


    def __getitem__(self, index):
        vid = self.__vid__[index]
 
        # import pdb;pdb.set_trace()
        
        feas = []
        preds = []
        from sklearn.preprocessing import Normalizer
        print(vid)
        print(self.file)
        tmp = self.file[vid]
        #tmp = tmp[:,:, 0:1024]   
        #tmp = tmp[0:32,0, 0:1024]  
        """
        if self.ten_crop:
            tmp = np.transpose(tmp, axes=(1, 0, 2))  # gives a [10, B, F] tensor
            divided_features = []
            for feature in tmp:        # For each 10-crop
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)     

            tmp = np.transpose(divided_features, axes=(1, 0, 2))  # returns to [B, 10, F] tensor
            ano_fea = tmp
        else:     

            if tmp.shape[0]%32 ==0:
                fea_new = np.reshape(tmp, (32,-1, tmp.shape[1]))
            else:
                feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1]))
                add =  (tmp.shape[0]//32+1) * 32 - tmp.shape[0]
                add_fea = np.tile(feat[-1],(add, 1))
                fea_new = np.concatenate((feat, add_fea), axis=0)
                fea_new = np.reshape(fea_new, (32,-1, tmp.shape[1]))

            ano_fea = fea_new.mean(axis=1)
            
        """
        ano_fea = np.asarray(tmp)
        #ano_fea = normalize(ano_fea, axis=1)
        #print("Shape do input depois da normalizacao: ")
        #print(ano_fea.shape)
        # if 'Explosion010_x264' in vid:
        #     import pdb;pdb.set_trace()
        # ano_feas = torch.from_numpy(ano_fea)
        # ano_fea = Normalizer(norm='l2').fit_transform(ano_fea)
        pred = 0
        if '/normal/' not in vid:
            pred += 1
        return ano_fea, pred, vid

    def __len__(self):
        return len(self.__vid__)
