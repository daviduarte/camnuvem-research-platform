import os
import cv2
import numpy as np
import torch
import definitions

class NormalDataset:
    def __init__(self, T, normal = True, test = False, has_cache = False):
        self.has_cache = has_cache
        
        self.test = test
        self.normal = normal
        

        # Test normal
        if not self.test and self.normal:
            self.normal_training_file = os.path.join(definitions.FRAMES_DIR, "train_normal.txt")
            self.T = T
            self.stride = T
        elif not self.test and not self.normal:
            self.normal_training_file = os.path.join(definitions.FRAMES_DIR, "train_anomaly.txt")
            self.T = T
            self.stride = T            
        elif self.test and self.normal: 
            self.normal_training_file = os.path.join(definitions.FRAMES_DIR, "test_normal.txt")
            self.T = T
            self.stride = 1
        elif self.test and not self.normal:
            print("Pegando os test anomaly")
            self.T = T
            self.stride = 1            
            self.normal_training_file = os.path.join(definitions.FRAMES_DIR, "test_anomaly.txt")


   
        self.frame_folders = {}       
        # TODO: Pass this to main 
        self.max_sample_duration = 300  # In .png qtt. 250png files, sampled at 0.5 sec each, results in 125 seconds (~2min)                
        self.sample_qtd = 0
        self._parse_list()

    def _parse_list(self):
        self.frame_folders['list'] = []
        self.frame_folders['sample_num'] = []    
        self.frame_folders['qtd_total_frame'] = []
        self.frame_folders['id'] = []      
        self.frame_folders['qtd_png'] = []        

        self.sample_qtd = self.parse()


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

        # If we are in test, we want the entire video loaded as a unique sample
        # If the video is too big, we crop it above to fit in memmory
        if self.test == True:
            sampleNum = 1
            #self.T = qtd
        else:
            sampleNum = ((qtd - self.T) / self.stride) + 1
            sampleNum = int(sampleNum)

        return sampleNum, qtd

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


    def parseTestAnomaly(self, normal_training_file):

        with open(normal_training_file, 'r') as f:
            path = f.read().splitlines()
            #test/anomaly/252|3795|[732, 2153, 2984, 3064]

            path = [i.split("|")[0] for i in path]

        return path

    def parseTestNormal(self, normal_training_file):

        with open(normal_training_file, 'r') as f:
            path = f.read().splitlines()
            #Format: test/normal/Normal_Videos_168_x264 1740 -1

            path = [i.split(" ")[0] for i in path]

        return path

    def parseTraining(self, normal_training_file):
        with open(normal_training_file, 'r') as f:
            path = f.read().splitlines()
            #Format: test/normal/Normal_Videos_168_x264
            path = [i for i in path]
        
        return path


    def parse(self):
       # Several folders with frames inside.
        totalSample = 0
        paths = []

        if self.test == False:
            path = self.parseTraining(self.normal_training_file)
        else:
            if self.normal == True:
                path = self.parseTestNormal(self.normal_training_file)
            else:
                path = self.parseTestAnomaly(self.normal_training_file)

        

        # run over every folder in path and add all sample IN ORDER
        cont = 0
        for frame_folder_path in path:

            frame_folder_path = os.path.join(definitions.FRAMES_DIR, frame_folder_path)

            if os.path.isdir(frame_folder_path):
                self.frame_folders['list'].append(frame_folder_path)
                num, qtd_png = self.calcule_sample_num(frame_folder_path)
                self.frame_folders['sample_num'].append(num)
                self.frame_folders['qtd_png'].append(qtd_png)

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

        # If we are in test, we need the sample loaded in memmory. We crop the video to fit in memmory in test.py
        if self.test:
            self.T = self.frame_folders['qtd_png'][folder_index]
        
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