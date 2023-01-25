import os
import numpy as np
import cv2
import sys
sys.path.append(r'/media/denis/dados/CamNuvem/pesquisa/anomalyDetection/RTFM')
from utils import process_feat
sys.path.insert(0, "/media/denis/dados/CamNuvem/pesquisa/anomalyDetection/graph_detector")
import modelPretext

def verify_if_cache_exists(path):
    if os.path.exists(path):
        return True
    else:
        print("Path não existe: ")
        print(path)
        exit()

def read_all_dirs(path):
    if not os.path.exists(path):
        print("Error: The directory does not exist.")
        return []

    file_list = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            file_list.append(os.path.join(root, dir))

    return file_list


def read_all_from_dir(path):
    if not os.path.exists(path):
        print("Error: The directory does not exist.")
        return []

    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list

DEVICE = "cuda:0"
OBJECT_FEATURE_SIZE = 1024
T = 5
N = 5
FEA_DIM_IN = (OBJECT_FEATURE_SIZE * N * (T-1)) + (4 * N * (T-1))
FEA_DIM_OUT = OBJECT_FEATURE_SIZE + 4
model = modelPretext.ModelPretext(FEA_DIM_IN, FEA_DIM_OUT).to(DEVICE)

train_anomaly_qtd = 49

png_folder = "/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/cache_pt_task/test/T=5-N=5"
npy_folder = "/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/sshc/test"

# We need the cache created. If it's not, run the pretext task from graph_detector
verify_if_cache_exists(png_folder)

data = read_all_dirs(png_folder)      # Recebe um path de diretório e retona uma lista de paths absolutos dos arquivos dentro desse diretório

cont = 1
for i, dir in enumerate(data):

    files = read_all_from_dir(dir)

    # Vamos descobrir o tamanho do vetor de características
    fea_size = 1028

    features = []

    num_vec = len(files)
    fea = np.zeros((num_vec, fea_size))
    for j, file in enumerate(files):
        data = np.load(file, allow_pickle=True)
        if data != -1:
            # If data == -1, there insn't any object in scene. Lets keep the the zero tensor        
            input, target = data
            input = input.to(DEVICE)
            target = target.to(DEVICE)

            output = model(input)        
            fea[j,:] = output.cpu().detach().numpy()

    #fea = process_feat(fea, 32)
    #print(fea.shape)

    if cont <= train_anomaly_qtd:
        folder = "anomaly"
        id = cont
    else:
        id = cont - train_anomaly_qtd
        folder = "normal"

    path_to = os.path.join(npy_folder, folder, str(id)+".npy")

    cont += 1
    print(path_to)
    
    np.save(path_to, fea)
    


