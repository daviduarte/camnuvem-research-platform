import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import cv2
import random

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class DatasetPretext(data.Dataset):
    #def __init__(self, args, is_normal=True, transform=None, test_mode=False, only_anomaly=False):
    def __init__(self, T, STRIDE, training_folder, test = False):
            
        self.T = T              # Frame qtt in any sample
        self.stride = STRIDE #self.T    # stride of the sliding window
        self.training_folder = training_folder
        self.frame_folders = {}
        self.totalSample = 0
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.test = test

    def countFiles(self, path, extension):
        counter = 0
        for img in os.listdir(path):
            ex = img.lower().rpartition('.')[-1]

            if ex == extension:
                counter += 1
        return counter


    def calcule_sample_num(self, frame_folder_path):
        qtd = self.countFiles(frame_folder_path, 'png')
        samplesNum = int(((qtd - self.T) / self.stride) + 1)

        return samplesNum


    def _parse_list(self):

        self.frame_folders['list'] = []
        self.frame_folders['sample_num'] = []

        # Several folders with frames inside.
        self.totalSample = 0
        for filename in os.listdir(self.training_folder):

            frame_folder_path = os.path.join(self.training_folder, filename)

            if os.path.isdir(frame_folder_path):
                self.frame_folders['list'].append(frame_folder_path)
                num = self.calcule_sample_num(frame_folder_path)
                self.frame_folders['sample_num'].append(num)
                self.totalSample += num
        

    def __getitem__(self, index):

        if self.test == False:
            # TODO: shufle param in the DataLoader is broken? This is a workaroud to get a random sample
            index = random.randint(0,self.totalSample-1)


        # 1 item is a windows of self.T frames
        sample_index = -1
        count = 0


        index = index+1         # Png frame files start at 1
        folder_index = 0
        for i, item in enumerate(self.frame_folders['sample_num']):   # for each sample num 'item' from each video 'i'
            count += item            
            if index <= count:             # The searched sample is in 'i' video
                #offset = self.frame_folders['sample_num'][folder_index]  
                
                sample_index = (((index - (count - item))-1) * self.stride)+1
                #if count > 0:
                #    sample_index = index - count

                #sampleIndex =  count + (index * self.stride)
                break
            folder_index += 1


        if sample_index == -1:
            print("Error")
            exit()

        # 'sample' is the sample index we are searching
        sample = []
        label = []
        for i in range(self.T):
            # Read the 'self.T' frames that compose the sample
            
            pathSample = os.path.join(self.frame_folders['list'][folder_index], str(sample_index+i)+'.png')
            #pathLabel = os.path.join(self.frame_folders['list'][folder_index], str(sample_index+i)+'.adj.npy')
            img = cv2.imread(pathSample)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

            sample.append(img)  

            # Read the adjacency matrix of this 5 frames that compose the labels
            # We ever have -1 adjacenct matrix than frames, because a a.m. is a connection between two frames
            #if i < self.T-1:
                #am = np.load(pathLabel)  
                #label.append(am)

        #   [T, 1024]   [T-1, 1024]
        
        sample = np.stack(sample, axis=0)

        # Returns [T, H, W, C]
        return sample


    def __len__(self):
        return self.totalSample

    def get_num_frames(self):
        return self.num_frame
