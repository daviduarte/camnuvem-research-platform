import visdom as viz
import os
import configparser
import torch
import numpy as np
from utils import Visualizer
import train
import test
from torch.utils.data import DataLoader
import normalDataset
import anomalyDataset
import sys
import options

viz = Visualizer(env='Anomaly-Graph', use_incoming_socket=False) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

config = configparser.ConfigParser(allow_no_value=True)
config.sections()
config.read('config.ini')
BATCH_SIZE = int(config['PARAMS']['BATCH_SIZE'])           # Window size in training
N = int(config['PARAMS']['N'])           # Window size in training
T = int(config['PARAMS']['T'])           # Window size in training
KEY_FRAME_SIM = float(config['PARAMS']['KEY_FRAME_SIM'])           # Theshold used to find Key Frames in a object path

STRIDE = N
buffer_size = 5 # We are not considering buffer at all
OBJECTS_ALLOWED = [1,2,3,4]    # COCO categories ID allowed. The othwers will be discarded
SIMILARITY_THRESHOLD = 0.65#0.73    # SIMILARITY USED TO CALCULATE A OBJECT PATH
max_sample_duration = 300

EDGE = []
VERTEX = []
GLOBAL_GRAPH = (VERTEX, EDGE)

###
# Training
###
normal_train_dataset = DataLoader(normalDataset.NormalDataset(T, normal =True, test=False),
                                   batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=0, pin_memory=False)  

anomaly_train_dataset = DataLoader(normalDataset.NormalDataset(T, normal =False, test=False),
                                   batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=0, pin_memory=False)  

normal_test_dataset = DataLoader(normalDataset.NormalDataset(T, normal =True, test=True),
                                   batch_size=1, shuffle=False,
                                   num_workers=0, pin_memory=False)  

anomaly_test_dataset = DataLoader(normalDataset.NormalDataset(T, normal =False, test=True),
                                   batch_size=1, shuffle=False,
                                   num_workers=0, pin_memory=False)                                     
###
# Test (TODO)
###                                   

reference_frame = 0
if __name__ == "__main__":

    args = options.parser.parse_args()

    params = sys.argv
    # Verifica se devemos carregar um grafo salvo anteriormente
    if args.load_graph == "1":
    #if len(params) == 2 and params[1] == "--load-graph":
        print("Carregand um global_graph")
        GLOBAL_GRAPH = np.load("global_graph.npy", allow_pickle=True).tolist()
    else:
        train.train(normal_train_dataset, DEVICE, buffer_size, reference_frame, OBJECTS_ALLOWED, N, T, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH)
        print("Salvando o global graph no disco")
        np.save("global_graph.npy", GLOBAL_GRAPH)

    test.test(normal_test_dataset, anomaly_test_dataset, max_sample_duration, DEVICE, buffer_size, reference_frame, int(args.video_live), OBJECTS_ALLOWED, N, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH)