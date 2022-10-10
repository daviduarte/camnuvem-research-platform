## Exemplo de: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Seems we do not need normalization for pytorch built-in models: https://github.com/pytorch/vision/blob/d2c763e14efe57e4bf3ebf916ec243ce8ce3315c/torchvision/models/detection/faster_rcnn.py#L227

# Explicação sobre Roi Polling e Roi Aling: 
# https://erdem.pl/pages/work

# Talvez um caminho de como acessar as features no RoiAlign:
#https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection

# Conferir esse feature extraction depois!
# https://github.com/pytorch/vision/pull/4302

import torch

import numpy as np
import PIL
import pickle
import cv2
import time
import torch.nn as nn
from typing import Dict, Iterable, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import objectDetector
import os
import csv


import modelPretext
import modelDownstream
import datasetPretext
import datasetDownstream
import temporalGraph
import configparser
from utils import Visualizer
from utils import calculeTarget
from utils import print_image
import utils
from test import test
from test_downstream import test as test_downstream
import losses

viz = Visualizer(env='Graph_Detector', use_incoming_socket=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    


config = configparser.ConfigParser(allow_no_value=True)
config.sections()
config.read('config.ini')
T = int(config['PARAMS']['T'])                                      # Frame window. The object predict is always in the last frame
N = int(config['PARAMS']['N'])                                      # How many objects we will consider for each frame?
STRIDE = int(config['PARAMS']['STRIDE'])                            # STRIDE for each sample
MAX_EPOCH = int(config['PARAMS']['MAX_EPOCH'])                      # Training max epoch
LR = float(config['PARAMS']['LR'])                                    # Learning rate
OBJECT_FEATURE_SIZE = int(config['PARAMS']['OBJECT_FEATURE_SIZE'])  # OBJECT_FEATURE_SIZE
SIMILARITY_THRESHOLD = float(config['PARAMS']['SIMILARITY_THRESHOLD'])

GT_PATH = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/files/gt-camnuvem.npy"

# Allow only 1 (person) 2 (bicycle) 3 (car) 4 (motorcycle)
OBJECTS_ALLOWED = [1,2,3,4]    # COCO categories ID allowed. The othwers will be discarded

FEA_DIM_IN = 0
FEA_DIM_OUT = 0

OUTPUT_PATH_PRETEXT_TASK = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/pretext_task"
OUTPUT_PATH_DOWNSTREAM_TASK = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/downstream_task"
MODEL_NAME = "model"

# Exemplo de como trabalhar com hook: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def loadImage():
    #image = PIL.Image.open("zidane.jpeg").convert('RGB').convert('RGB')
    image = cv2.imread("teste.png")
    

    return image

def train(save_folder):

    global SIMILARITY_THRESHOLD, MAX_EPOCH
    print("Iniciando treinamento para")
    print("T = " + str(T) + "; N = " + str(N) + "; LR = " + str(LR) + "; STRIDE: "+str(STRIDE)+"; SIMILARITY_THRESHOLD: " + str(SIMILARITY_THRESHOLD)) 

    training_loss_log = os.path.join(save_folder, "training_log.txt")
    test_loss_log = os.path.join(save_folder, "test_log.txt")
    trining_log = open(training_loss_log, 'w')
    test_log = open(test_loss_log, 'w')

    temporal_graph = temporalGraph.TemporalGraph(DEVICE, OBJECTS_ALLOWED, N)
    #temporal_graph.generateTemporalGraph()

    batch_size = 1              # Aumentar no final
    max_sample_duration = 200   # Limitando as amostras por no máximo 200 arquivos.png
    # each video is a folder number-nammed
    training_folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/training"
    train_loader = DataLoader(datasetPretext.DatasetPretext(T, STRIDE, training_folder, max_sample_duration),
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)   

    test_folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/test"
    test_loader = DataLoader(datasetPretext.DatasetPretext(T, STRIDE, test_folder, max_sample_duration, test=True),
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)   


    model = modelPretext.ModelPretext(FEA_DIM_IN, FEA_DIM_OUT).to(DEVICE)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                            lr=LR, weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}

    # index of object in the object listto be predicted
    obj_predicted = 0
    reference_frame = 0

    best_loss = float("+Inf")    
    loss_mean = test(model, loss, test_loader, reference_frame, obj_predicted, viz, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, OBJECTS_ALLOWED)    
    test_log.write(str(loss_mean) + " ")
    test_log.flush()

    

    data_loader = iter(train_loader)    
    MAX_EPOCH = len(data_loader) * MAX_EPOCH
    for step in tqdm(
            range(1, MAX_EPOCH + 1),
            total=MAX_EPOCH,
            dynamic_ncols=True
    ):    

        with torch.set_grad_enabled(True):
            model.train()

            if (step - 1) % len(data_loader) == 0:
                data_loader = iter(train_loader)

            # input: [T, W, H, C]
            input = next(data_loader)
            input = np.squeeze(input)
            
            # Returns [T-1, obj1, obj2], beeing obj1 the num object detected in the first frame and obj2 in the second frame
            # [] if a frame does not have objects
            adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input)
            SIMILARITY_THRESHOLD = 0.65#0.73
            graph = utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

            # If in the first frame there is no object detected, so we have nothing to do here
            # The number of detected objects may be less than N. In this case we have nothing to do here
            #if len(bbox_fea_list[reference_frame][obj_predicted]) < N:
            #    print("continuando")
            #    continue       # Continue

            #data, object_path = calculeTarget(adj_mat, bbox_fea_list, box_list, reference_frame, obj_predicted, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)
            data, object_path = calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)
            if data == -1:
                print("Continuing because there aren't a object in the first frame ")
                continue

            #print_image(input, box_list, object_path, step)

            input, target = data

            input = input.to(DEVICE)
            target = target.to(DEVICE)
            output = model(input)

            loss_ = loss(output, target)
            #viz.plot_lines('loss', cost.item())
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            viz.plot_lines('training_loss', loss_.item())

            trining_log.write(str(loss_.item()) + " ")

            if step % len(data_loader) == 0:# and step > 10:
                trining_log.flush()
                loss_mean = test(model, loss, test_loader, reference_frame, obj_predicted, viz, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, OBJECTS_ALLOWED)    
                test_log.write(str(loss_mean) + " ")
                test_log.flush()

                if loss_mean < best_loss:
                    # Save model 
                    torch.save(model.state_dict(), os.path.join(save_folder, MODEL_NAME + '{}.pkl'.format(step)))                    
                    fo = open(os.path.join(save_folder, MODEL_NAME + '{}.txt'.format(step)), "w")
                    fo.write("Test loss: " + str(loss_mean))
                    fo.close()     
                    best_loss = loss_mean    


    trining_log.close()
    test_log.close()           

def run():
    global T, N, LR, FEA_DIM_IN, OBJECT_FEATURE_SIZE, FEA_DIM_OUT, SIMILARITY_THRESHOLD, EXIT_TOKEN

    T_ = [T, T+1, T+2, T+3]
    N_ = [N, N+1, N+2]
    #LR_ = [LR*10, LR, LR/10]
    #SIMILARITY_THRESHOLD_ = [SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD-0.1, SIMILARITY_THRESHOLD-0.2]

    for t in T_:
        for n in N_:
            #for lr in LR_:
            #for st in SIMILARITY_THRESHOLD_:
            T = t
            N = n
            #LR = lr
            #SIMILARITY_THRESHOLD = st            # Threshold to verify if two detected are the same

            FEA_DIM_IN = (OBJECT_FEATURE_SIZE * N * (T-1)) + (4 * N * (T-1))
            FEA_DIM_OUT = OBJECT_FEATURE_SIZE + 4
            EXIT_TOKEN = FEA_DIM_OUT

            #if (T == 2 and N == 1 and LR == 0.005) or (T == 2 and N == 1 and LR == 0.0005) or (T == 2 and N == 1 and LR == 0.00005):
            #    if st == 0.7:
            #        continue
            #if (T == 2 and N == 2 and LR == 0.0005):
            #    if st == 0.7 or st == 0.6 or st == 0.5:
            #        continue


            save_folder = "t="+str(T)+"-n="+str(N)+"-lr="+str(LR)+"-st="+str(SIMILARITY_THRESHOLD)
            save_folder = os.path.join(OUTPUT_PATH_PRETEXT_TASK, save_folder)

            try:
                os.mkdir(save_folder)
            except OSError as error:
                print("Erro ao criar dir: ")
                print(error)    
                continue

            train(save_folder)




def downstreamTask(T, N, st, N_DOWNSTRAM, FEA_DIM_IN, FEA_DIM_OUT, pretext_checkpoint, downstream_folder):
    #global FEA_DIM_OUT, FEA_DIM_OUT

    EXIT_TOKEN = FEA_DIM_OUT

    trining_log = open(os.path.join(downstream_folder, "training_log.txt"), 'w')
    test_log = open(os.path.join(downstream_folder, "test_log.txt"), 'w')

    print("Carregando o checkpoint ")
    print(pretext_checkpoint)

    RANDOM_WEIGHTS = 1
    if RANDOM_WEIGHTS == 1:
        print("ATENÇÃO, VC ESTÁ CARREGANDO UM PRETEXT MODEL RANDOMIZADO")
        pretext_checkpoint = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/pretext_task/t=5-n=5-lr=5e-05-st=0.7_RANDOM_WEIGHTS/model_random_weights.pkl"

    model_pt = modelPretext.ModelPretext(FEA_DIM_IN, FEA_DIM_OUT)
    model_pt.load_state_dict(torch.load(pretext_checkpoint))

    temporal_graph = temporalGraph.TemporalGraph(DEVICE, OBJECTS_ALLOWED, N)

    #model_pt.ModelPretext = nn.Sequential(*list(model_pt.ModelPretext.children())[:-1])
    prunned_model_pt = nn.Sequential(*list(model_pt.children())[:-2])
    prunned_model_pt.train()
    
    # A entrada vai ser o tamanho da penútima saída do Pretext Model
    model = modelDownstream.ModelDownstream(128).to(DEVICE)

    LR_DOWNSTREAM = 0.00005

    batch_size = 250
    max_sample_duration = 300
    normal_dataset = DataLoader(datasetDownstream.DatasetDownstream(T, max_sample_duration, normal = True, test=False), batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)

    abnormal_dataset = DataLoader(datasetDownstream.DatasetDownstream(T, max_sample_duration, normal = False, test=False), batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=False)

    list_ = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/files/graph_detector_test_05s.list"
    test_dataset = DataLoader(datasetDownstream.DatasetDownstream(T, max_sample_duration, list_=list_, normal = True, test=True), batch_size=1, shuffle=False,
                                   num_workers=0, pin_memory=False)


    optimizer = optim.Adam(model.parameters(),
                            lr=LR_DOWNSTREAM, weight_decay=0.005)


    auc = test_downstream(test_dataset, prunned_model_pt, model, viz, max_sample_duration, list_, DEVICE, False, GT_PATH, OBJECTS_ALLOWED, N, T, EXIT_TOKEN)
    test_log.write(str(auc) + " ")
    test_log.flush()
    best_auc = auc


    normal_loader = iter(normal_dataset)    
    abnormal_loader = iter(abnormal_dataset)    
    for step in tqdm(
            range(1, MAX_EPOCH + 1),
            total=MAX_EPOCH,
            dynamic_ncols=True
    ):    

        with torch.set_grad_enabled(True):
            model.train()
            prunned_model_pt.train()

            if (step - 1) % len(normal_loader) == 0:
                normal_loader = iter(normal_dataset)  

            if (step - 1) % len(abnormal_loader) == 0:
                abnormal_loader = iter(abnormal_dataset)                  

            # input: [T, W, H, C]
            input_normal = next(normal_loader)
            input_normal = np.squeeze(input_normal)
            input_normal = torch.squeeze(input_normal[0])

            input_abnormal = next(abnormal_loader)
            input_abnormal = np.squeeze(input_abnormal)
            input_abnormal = torch.squeeze(input_abnormal[0])
            
            batch_list = utils.batch_processing(input_abnormal, input_normal, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

            print(len(batch_list))

            input_abnormal = [i[0] for i in batch_list]
            input_normal = [i[1] for i in batch_list]

            if len(input_abnormal) == 0 or len(input_normal) == 0:
                print("Nenhuma amostra possui objetos no primeiro frame")
                continue

            input_abnormal = torch.stack(input_abnormal)
            input_normal = torch.stack(input_normal)

            abnormal_res = model(prunned_model_pt(input_abnormal))

            print(input_normal.shape)
            normal_res = model(prunned_model_pt(input_normal))

            print(normal_res)
            print(abnormal_res)

            downstramLoss = losses.DownstramLoss(normal_res, abnormal_res)
            cost = downstramLoss()
            print("loss: ")
            print(cost)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            trining_log.write(str(cost.item()) + " ")

            if step % len(normal_loader) == 0 and step > 10:
                trining_log.flush()

                auc = test_downstream(test_dataset, prunned_model_pt, model, viz, max_sample_duration, list_, DEVICE, False, GT_PATH, OBJECTS_ALLOWED, N, T, EXIT_TOKEN)
                #loss_mean = test(model, loss, test_loader, reference_frame, obj_predicted, viz, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, OBJECTS_ALLOWED)    
                test_log.write(str(auc) + " ")
                test_log.flush()

                if auc < best_auc:
                    # Save model 
                    torch.save(model.state_dict(), os.path.join(downstream_folder, MODEL_NAME + '{}.pkl'.format(step)))                    
                    fo = open(os.path.join(downstream_folder, MODEL_NAME + '{}.txt'.format(step)), "w")
                    fo.write("Test loss: " + str(auc))
                    fo.close()     
                    best_auc = auc              
    test_log.close()
    trining_log.close()

def runDownstream():

    config = configparser.ConfigParser(allow_no_value=True)
    config.sections()
    config.read('config.ini')
    T = int(config['PARAMS']['T'])                                      # Frame window. The object predict is always in the last frame
    N = int(config['PARAMS']['N'])                                      # How many objects we will consider for each frame?
    STRIDE = int(config['PARAMS']['STRIDE'])                            # STRIDE for each sample
    MAX_EPOCH = int(config['PARAMS']['MAX_EPOCH'])                      # Training max epoch
    LR = float(config['PARAMS']['LR'])                                  # Learning rate
    OBJECT_FEATURE_SIZE = int(config['PARAMS']['OBJECT_FEATURE_SIZE'])  # OBJECT_FEATURE_SIZE
    SIMILARITY_THRESHOLD = float(config['PARAMS']['SIMILARITY_THRESHOLD'])

    FEA_DIM_IN = (OBJECT_FEATURE_SIZE * N * (T-1)) + (4 * N * (T-1))
    FEA_DIM_OUT = OBJECT_FEATURE_SIZE + 4
    EXIT_TOKEN = FEA_DIM_OUT

    #N_DOWNSTRAM = N         # Tamanho da janela para ver se é normal ou abnormal
    

    #pretext_path = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/graph_detector/results/pretext_task/"


    T_ = [T, T+1, T+2, T+3]
    N_ = [N, N+1, N+2]
    #LR_ = [LR*10, LR, LR/10]
    SIMILARITY_THRESHOLD_ = [SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD-0.1, SIMILARITY_THRESHOLD-0.2]

    for t in T_:
        for n in N_:

            #for lr in LR_:
            for st in SIMILARITY_THRESHOLD_:      

                N_DOWNSTRAM = n
                pretext_folder_sufix = "t="+str(t)+"-n="+str(n)+"-lr="+str(LR)+"-st="+str(st)
                pretext_folder = os.path.join(OUTPUT_PATH_PRETEXT_TASK, pretext_folder_sufix)                    
                pretext_checkpoint = os.path.join(pretext_folder, find_value(pretext_folder))
                
                downstream_folder = os.path.join(OUTPUT_PATH_DOWNSTREAM_TASK, pretext_folder_sufix)
                try:
                    os.mkdir(downstream_folder)
                except OSError as error:
                    print("Erro ao criar dir: ")
                    print(error)    
                    continue
                
                downstreamTask(t, n, st, N_DOWNSTRAM, FEA_DIM_IN, FEA_DIM_OUT, pretext_checkpoint, downstream_folder)


def find_value(dir):

    return ''
    score = []
    files = []
    # Find all files with model*.txt in the 'dir' folder
    for file in os.listdir(dir):    
        if 'model' in file and file.endswith(".txt"):
            number = file[5:][:-4]
            files.append(int(number))

    files.sort()
    the_one = files[-1]
    nome = "model"+str(the_one)+".pkl"
    return nome

if __name__ == '__main__':

    # Search training parameter
    #run()
    #downstreamTask()
    runDownstream()




            


