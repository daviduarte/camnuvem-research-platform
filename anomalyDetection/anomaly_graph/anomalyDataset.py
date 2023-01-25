import os
import cv2
import numpy as np
import torch
import definitions

class AnomalyDataset:
    def __init__(self, test, T, has_cache = False):
        self.has_cache = has_cache
        self.T = T
        self.test = test
        if not self.test:
            self.stride = T
        else:
            print("Ueeeepa, ainda não implementamos o test")
            exit()        
        self.frame_folders = {}        
        self.normal_training_file = os.path.join(definitions.DATASET_DIR, "train_normal.txt")
        self.max_sample_duration = 250  # In .png qtt. 250png files, sampled at 0.5 sec each, results in 125 seconds (~2min)                
        self.sample_qtd = 0
        self._parse_list()

    def _parse_list(self):
        self.frame_folders['list'] = []
        self.frame_folders['sample_num'] = []    
        self.frame_folders['qtd_total_frame'] = []
        self.frame_folders['id'] = []        
        if self.test == True:
            print("Parse yet not implemented for test")
            exit()            
        else:
            self.sample_qtd = self.parseTraining()

    def countFiles(self, path, extension):
        counter = 0
        for img in os.listdir(path):
            ex = img.lower().rpartition('.')[-1]

            if ex == extension:
                counter += 1
        return counter

    def calcule_sample_num(self, frame_folder_path):
        qtd = self.countFiles(frame_folder_path, 'png')

        if qtd > self.max_sample_duration:
            qtd = self.max_sample_duration

        samplesNum = int(((qtd - self.T) / self.stride) + 1)

        return samplesNum

    def calcule_totl_qtd_frame(self, frame_folder_path):

        #factor = 0.5  
        #frames_total = qtd * factor * 30     # We know this

        video_path = frame_folder_path.replace('CamNuvem_dataset_normalizado_frames_05s', 'CamNuvem_dataset_normalizado/videos/samples')
        #video_path = video_path.replace('/10', '/10.mp4')

        video_path = video_path.rsplit('/', 1)
        video_path = os.path.join(video_path[0], video_path[1]+'.mp4')

        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return length

    def parseTraining(self):
       # Several folders with frames inside.
        totalSample = 0
        paths = []
        with open(self.normal_training_file, 'r') as f:
            path = f.read().splitlines()
            #Format: test/normal/Normal_Videos_168_x264 1740 -1
            path = [i for i in path]

        # run over every folder in path and add all sample IN ORDER
        cont = 0
        for frame_folder_path in path:

            frame_folder_path = os.path.join(definitions.DATASET_DIR, frame_folder_path)

            if os.path.isdir(frame_folder_path):
                self.frame_folders['list'].append(frame_folder_path)
                num = self.calcule_sample_num(frame_folder_path)
                self.frame_folders['sample_num'].append(num)

                qtd_total_frame = self.calcule_totl_qtd_frame(frame_folder_path)

                self.frame_folders['qtd_total_frame'].append(qtd_total_frame)   
                
                self.frame_folders['id'].append(cont)

                totalSample += num      

                cont += 1
        return totalSample          

    def getImage(self, index):

        sample_index = -1
        count = 0

        folder_index = 0
        for i, item in enumerate(self.frame_folders['sample_num']):   # for each sample 'item' from each video 'i'
            count += item            
            if index <= count:             # The searched sample is in 'i' video
                sample_index = (((index - (count - item))-1) * self.stride)+1
                break
            folder_index += 1

        if sample_index == -1:
            print("Error")
            exit()

        # If we already has the processed frames in cache, we don't need load the images
        # 'sample' is the sample index we are searching for
        sample = []

        if not self.has_cache:
            
            for i in range(self.T):
                # Read the 'self.T' frames that compose the sample
                
                pathSample = os.path.join(self.frame_folders['list'][folder_index], str(sample_index+i)+'.png')

                img = cv2.imread(pathSample)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
                sample.append(img)  

        #   [T, 1024]   [T-1, 1024]
            sample = np.stack(sample, axis=0)

        return sample, folder_index, sample_index

    def get_label(self):

        if self.test == True:
            return False
    
        return torch.tensor(0.0)

    def __len__(self):
        return self.sample_qtd

    def __getitem__(self, index):

        label = self.get_label()

        # In test we need samples IN ORDER
        #if self.test == False:
            # TODO: shufle param in the DataLoader is broken? This is a workaroud to get a random sample
        #    index = random.randint(0,self.sample_qtd-1)
        index = index+1         # Png frame files start at 1
        sample, folder_index, sample_index = self.getImage(index)
        sample = sample.astype('float32')

        #if self.test:
        return sample, label, int(folder_index), int(sample_index)