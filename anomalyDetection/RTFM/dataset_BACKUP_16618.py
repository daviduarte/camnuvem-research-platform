import torch.utils.data as data
import numpy as np
from .utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, only_anomaly=False):
        self.args = args
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.only_anomaly = only_anomaly
        #if self.dataset == 'shanghai':
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            print("Carregando o " + args.test_rgb_list)
        else:
            print("Carregando o " + args.rgb_list)
            self.rgb_list_file = args.rgb_list


        #elif self.dataset == 'camnuvem':
        #    if test_mode:
        #        self.rgb_list_file = 'list/camnuvem-i3d-test-10crop.list'
        #    else:
        #        self.rgb_list_file = 'list/camnuvem-i3d-train-10crop.list'
        #else:            

        #    if test_mode:
        #        self.rgb_list_file = 'list/ucf-i3d-test.list'
        #    else:
        #        self.rgb_list_file = 'list/ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):

        self.list = list(open(self.rgb_list_file))

        # If we want test only in anomaly videos
        if self.test_mode is True and self.only_anomaly is True:
<<<<<<< HEAD
            self.list = self.list[0:140]  # The anomaly videos is 0 to 49
            #self.list = self.list[0:49]  # The anomaly videos is 0 to 49
=======
            self.list = self.list[0:140]  # The anomaly videos is 0 to 49  UCF CRIME
            #self.list = self.list[0:49]  # The anomaly videos is 0 to 49    CAMNUVEM 
>>>>>>> bug-fix

            return



        if self.test_mode is False:
            """
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)
            elif self.dataset == 'camnuvem'                    :
            """
            if self.is_normal:
<<<<<<< HEAD
                #self.list = self.list[437:]
                self.list = self.list[810:]
                #print('normal list for CamNuvem')
                #print(self.list)
            else:
                #self.list = self.list[:437]
                self.list = self.list[:810]
=======
                #self.list = self.list[437:]    #CamNuvem
                self.list = self.list[810:]     # UCF Crime
                #print('normal list for CamNuvem')
                #print(self.list)
            else:
                #self.list = self.list[:437]    # CamNuvem
                self.list = self.list[:810]     # UCF Crime
>>>>>>> bug-fix
                #print('abnormal list for CamNuvem')
                #print(self.list)                    

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1

        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        
        features = np.array(features, dtype=np.float32)

        

        #print(features.shape)
        if self.args.crop_10 == "False":

            # Add a dummy dimension to simulate the 10 crop
            features = features[:, None, :]
        #features = torch.nn.functional.normalize(torch.from_numpy(features), dim=2).numpy()

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            #print(features.shape)
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            #print("Shape antes do looping: ")
            #print(features.shape)
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
