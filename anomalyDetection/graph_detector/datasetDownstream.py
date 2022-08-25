import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import cv2
import random


torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DatasetDownstream(data.Dataset):
    #def __init__(self, args, is_normal=True, transform=None, test_mode=False, only_anomaly=False):
    def __init__(self, T, list_=None, normal = True, test = False):

        self.normal = normal
        self.max_sample = 32        # If the video is too big, we limitate due computational limitations
        self.T = T                  # Frame qtt in any sample
        self.stride = 1 #self.T    # stride of the sliding window
        self.frame_folders = {}
        self.sample_qtd = 0
        self.abnormal_sample_qtd = 0

        self.num_frame = 0
        self.labels = None
        self.test = test

        if self.test == False:
            if self.normal == True:
                self.folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/downstream_task_teste/0.5s/training/normal"
            else:
                self.folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/downstream_task_teste/0.5s/training/anomaly"
        else:            
            assert(list_ != None)
            # If we are in test, we do not need the self.folder, we have a .txt containing all files we need in order
            self.list = list_

        self._parse_list()

    def countFiles(self, path, extension):
        counter = 0
        for img in os.listdir(path):
            ex = img.lower().rpartition('.')[-1]

            if ex == extension:
                counter += 1
        return counter


    # In the test file, the sample quantity is the total number of frame, because 
    # each frame has a target (0 = normal; 1 = anomaly). 
    def calcule_sample_num_test(self, frame_folder_path):
        # Each file represents only 0.5 seconds of a 30fps video. 2 png files represents 30 frames (1 second)

        qtd = self.countFiles(frame_folder_path, 'png')

        return qtd

    def calcule_totl_qtd_frame(self, frame_folder_path):

        #factor = 0.5  
        #frames_total = qtd * factor * 30     # We know this

        video_path = frame_folder_path.replace('CamNuvem_dataset_normalizado_frames_05s', 'CamNuvem_dataset_normalizado/videos/samples')
        #video_path = video_path.replace('/10', '/10.mp4')

        video_path = video_path.rsplit('/', 1)
        video_path = os.path.join(video_path[0], video_path[1]+'.mp4')

        print("Path do video")
        print(video_path)

        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        print("Qtd de frames obtido do video")
        print( length )

        return length

        return qtd


    def calcule_sample_num(self, frame_folder_path):
        qtd = self.countFiles(frame_folder_path, 'png')
        samplesNum = int(((qtd - self.T) / self.stride) + 1)

        return samplesNum


    def parse(self):
        # Several folders with frames inside.
        totalSample = 0
        for filename in os.listdir(self.folder):

            frame_folder_path = os.path.join(self.folder, filename)

            if os.path.isdir(frame_folder_path):
                self.frame_folders['list'].append(frame_folder_path)
                num = self.calcule_sample_num(frame_folder_path)
                self.frame_folders['sample_num'].append(num)

                totalSample += num
        return totalSample

    def parse_test(self):
        # Several folders with frames inside.
        totalSample = 0
        paths = []
        with open(self.list, 'r') as f:
            path = f.read().splitlines()
            path = [i[:-4] for i in path]

        # run over every folder in path and add all sample IN ORDER
        cont = 0
        for frame_folder_path in path:

            if os.path.isdir(frame_folder_path):
                self.frame_folders['list'].append(frame_folder_path)
                num = self.calcule_sample_num_test(frame_folder_path)
                self.frame_folders['sample_num'].append(num)
                qtd_total_frame = self.calcule_totl_qtd_frame(frame_folder_path)
                self.frame_folders['qtd_total_frame'].append(qtd_total_frame)   
                
                self.frame_folders['id'].append(cont)

                totalSample += num      

                cont += 1
        return totalSample          

    def _parse_list(self):

        self.frame_folders['list'] = []
        self.frame_folders['sample_num'] = []    
        self.frame_folders['qtd_total_frame'] = []
        self.frame_folders['id'] = []

        if self.test == False:
            self.sample_qtd = self.parse()
        else:
            self.sample_qtd = self.parse_test()

        
    def getImage(self, index):

        sample_index = -1
        count = 0

        folder_index = 0
        for i, item in enumerate(self.frame_folders['sample_num']):   # for each sample num 'item' from each video 'i'
            count += item            
            if index <= count:             # The searched sample is in 'i' video
            
                sample_index = (((index - (count - item))-1) * self.stride)+1

                break

            folder_index += 1


        if sample_index == -1:
            print("Error")
            exit()

        # 'sample' is the sample index we are searching
        sample = []
        for i in range(self.T):
            # Read the 'self.T' frames that compose the sample
            
            pathSample = os.path.join(self.frame_folders['list'][folder_index], str(sample_index+i)+'.png')
            img = cv2.imread(pathSample)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

            sample.append(img)  

        #   [T, 1024]   [T-1, 1024]
        sample = np.stack(sample, axis=0)



        return sample, folder_index


    def __getitem__(self, index):

        label = self.get_label()

        # In test we need samples IN ORDER
        if self.test == False:
            # TODO: shufle param in the DataLoader is broken? This is a workaroud to get a random sample
            index = random.randint(0,self.sample_qtd-1)


        index = index+1         # Png frame files start at 1

        if index > self.T:
            index = index - self.T + 1

            sample, folder_index = self.getImage(index)
        else:
            index = 1 
            sample, folder_index = self.getImage(index)
            
            # Create some trick to inference the first frames. 
            # In the first frames, when we don't have a windows of self.T frame to predict the anomality of a frame,
            # we have to create a windows composed of black images, used to predict the last frame.
            shape = sample.shape
            sample_new = np.zeros((shape[0], shape[1], shape[2], shape[3]))
            # ok, case index == self.T == 1 works
            
            sample_new[-1] = sample[index-1]    # Put image 'index' in the last position of sample_new.
                
            sample = sample_new
        
        sample = sample.astype('float32')
        print("Shape dos "+str(self.T) + " frames lidos")
        print(sample.shape)     

        print("Shape do input: ")       
        print(sample.dtype)


        if self.test:
            return sample, label, self.frame_folders['qtd_total_frame'][folder_index], self.frame_folders['id'][folder_index]

        # Returns [T, H, W, C]
        return sample, label

    def get_label(self):

        if self.test == True:
            return False

        if self.normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label


    def __len__(self):
        return self.sample_qtd

    def get_num_frames(self):
        return self.num_frame
