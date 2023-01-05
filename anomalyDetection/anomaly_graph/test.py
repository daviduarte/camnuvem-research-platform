import os
from definitions import DATASET_DIR, ROOT_DIR, FRAMES_DIR
import cv2
import numpy as np
import torch
import temporalGraph
import util.utils
from trajectory_analysis import analysis
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import time

#param labels A txt file path containing all test/anomaly frame level labels
#param list A txt file path containing all absolut path of every test file (normal and anomaly)
def getLabels(labels, list_test):

    # Colocar isso no config.ini depois
    # TODO
    test_normal_folder = os.path.join(DATASET_DIR, "videos/samples/test/normal")
    test_anomaly_folder = os.path.join(DATASET_DIR, "videos/samples/test/anomaly")

    with open(labels) as file:
        lines = file.readlines()
    qtd_anomaly_files = len(lines)

    gt = []
    qtd_total_frame = 0
    anomaly_qtd = 0
    for line in lines:        

        line = line.strip()
        list = line.split("  ")

        video_name = list[0]
        video_path = os.path.join(test_anomaly_folder, video_name)
        
        # First we create an array with 'frame_qtd' zeros
        # Zeros represents the 
        cap = cv2.VideoCapture(video_path)
        frame_qtd = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        qtd_total_frame += frame_qtd

        frame_label = np.zeros(frame_qtd)

        labels = list[1]
        labels = labels.split(' ')

        assert(len(labels) % 2 == 0) # We don't want incorrect labels
        sample_qtd = int(len(labels)/2)
        
        
        for i in range(sample_qtd):
            index = i*2
            start = round(float(labels[index]) * frame_qtd)
            end = round(float(labels[index+1]) * frame_qtd)
            
            frame_label[start:end] = 1

        gt.append([video_name, frame_label])

        anomaly_qtd += 1




    #############################################################

    lines = []
    with open(list_test) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]


    list_ = []
    cont = 0
    for path in lines:
        cont+=1
        if cont <= anomaly_qtd:
            continue
        filename = os.path.basename(path)  
        list_.append(os.path.join(test_normal_folder, filename[:-4]+'.mp4'))


    # Lets get the normal videos
    qtd_total_frame = 0
    for video_path in list_:
        video_path = video_path.strip()

        # First we create an array with 'frame_qtd' zeros
        # Zeros represents the 
        
        cap = cv2.VideoCapture(video_path)
        frame_qtd = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        qtd_total_frame += frame_qtd

        frame_label = np.zeros(frame_qtd)   # All frames here are normal.
        
        gt.append([video_path, frame_label])
        
    return gt

def calculeAbnormality(data, has_cache, video, cache_folder, DEVICE, buffer_size, reference_frame, OBJECTS_ALLOWED, N, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH):

    input = torch.squeeze(data[0]).to(DEVICE)
    labels = data[1].to(DEVICE)
    folder_index = data[2]
    sample_index = data[3]        

    EDGES = GLOBAL_GRAPH[1]    
    edges = np.asarray(EDGES)

    fea_path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.numpy()[0]), str(sample_index.numpy()[0])+"_features.npy")
    graph_path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.numpy()[0]), str(sample_index.numpy()[0])+"_graph.npy")

    if not has_cache:
        T = input.shape[0]  # Cuz in test we consider the full frame video as a unique sample
        print("Tamanho da amostra que estamos carregando na memória: " + str(T))
        temporal_graph = temporalGraph.TemporalGraph(DEVICE, buffer_size, OBJECTS_ALLOWED, N, STRIDE)
        adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input, folder_index, sample_index)
        graph_test = util.utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N)        

        #os.makedirs(path, exist_ok=True)
        path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.numpy()[0]))
        os.makedirs(path, exist_ok=True)

        print(fea_path)
        np.save(fea_path, np.asarray([adj_mat, bbox_fea_list, box_list, score_list]))
        np.save(graph_path, graph_test)

    else:
        print("Temos a cache, usar xDDDDD")
        print(graph_path)
        adj_mat, bbox_fea_list, box_list, score_list = np.load(fea_path, allow_pickle=True).tolist()
        graph_test = np.load(graph_path, allow_pickle=True)            

    influence_global = np.zeros(len(video[1]))
    # Run each trajectory in the graph
    last_len = 0
    for i in range(len(graph_test)):
        size = len(graph_test[i])

        if size <= last_len:
            continue

        for j in range(len(graph_test[i][last_len:])):                

            obj_predicted = j + last_len
            path = analysis.calculeObjectPath(graph_test, i, obj_predicted)
            influence = np.zeros(len(video[1])) #[0 for i in range(len(video[1]))]

            # The object has to have at minimum two frames
            if len(path) <= 1:
                continue

            # Calcule frame similarity with predecessor frame

            #start = time.time()
            key_frames, position = analysis.calculeFrameSimilarity(i, path, score_list, bbox_fea_list, box_list, KEY_FRAME_SIM, DEVICE)
            #end = time.time()
            #print("Tempo 1: " + str(end-start))

            # key_frames is a vector of X elements
            # sim is a vector of X anoaly scores
            #start = time.time()
            sim = measure_divergency(key_frames, edges, GLOBAL_GRAPH, SIMILARITY_THRESHOLD, DEVICE)
            #end = time.time()
            #print("Tempo 2: " + str(end-start))

            #start_ = time.time()
            # Start from i, lets put each score in a set of 15 frames
            start = i*15
            for k in range(0, len(sim)-1):
                s = sim[k]
                e = sim[k+1]

                p_s = position[k]*15
                p_e = position[k+1]*15

                #print("Inserindo na posição "+str(start + p_s)+" até "+str(start + p_e))

                influence[start + p_s : start + p_e] = s            
            arr_aux = np.stack((influence_global, influence))
            arr_mean = np.mean(arr_aux, axis=0)
            influence_global = arr_mean
            #if influence_global.shape[0] == 0:
            #    influence_global = np.append(influence_global, influence, axis=0)
            #end_ = time.time()
            #print("Tempo 3: " + str(end_-start_))
        
            #influence_global.append(influence)

        last_len = size

    # influence_global é a contribuição de abnormalidade que cada pessoa fez no video 'video_index'.
    # Agora vamos fazer uma média
    #frame_qtd = influence_global.shape[0]   # Frames qtd
    #print(frame_qtd)
    #final_score = np.zeros(frame_qtd)
    #for frame in range(frame_qtd):
    #    scores = influence_global[:,frame]      # Get all score contributions from frame 'frame'
    #    final_score[frame] = np.mean(scores)       # TODO explore other metrics, like median, max, etc... #TODO Reranking here?

    #return final_score

    return influence_global
    

def test(normal_test_dataset, anomaly_test_dataset, max_sample_duration, DEVICE, buffer_size, reference_frame, OBJECTS_ALLOWED, N, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH):
    # For each trajectory, calculate the anomaly score
    # Anomaly first

    print("Iniciando o teste...")
    cache_folder = "cache/test/anomaly"
    has_cache = False
    if os.path.exists(os.path.join(FRAMES_DIR, cache_folder)):
        has_cache = True
        anomaly_test_dataset.has_cache = True
    print("has cache? " + str(has_cache))

    anomaly_test_iter = iter(anomaly_test_dataset)
    anomaly_qtd = len(anomaly_test_iter)
    print(anomaly_qtd)


    list_ = os.path.join(ROOT_DIR, "../", "files/graph_detector_test_05s.list")    

    # Recovery a frame-level label
    # Receber isso por parâmetro
    NUM_SAMPLE_FRAME = 15
    LABELS_PATH = os.path.join(DATASET_DIR, "videos/labels/test.txt")
    labels = getLabels(LABELS_PATH, list_) # 2d matrix containing the frame-level frame (columns) for each video (lines)

    pred = []
    gt_ = []
    for i in range(len(anomaly_test_iter)):

        video = labels[i]
    #for video_index, video in enumerate(labels):    # For each video
        # Adjust the labels to the truncated frame due computation capability militation
        truncated_frame_qtd = int((max_sample_duration) * NUM_SAMPLE_FRAME)    # The video has max this num of frame
        if len(video[1]) > truncated_frame_qtd:
            video[1] = video[1][0:truncated_frame_qtd]                  # If needed, truncate it
        print("Tem 1 nos labels desse video? ")
        print(max(video[1]))
        gt_.extend(video[1])
        print("Tem 1 nbo vetor gt_?")
        print(max(gt_))

        print("Qtd de frames adicionado no gt_: "+str(len(video[1])))


        # If folder doesn't exist, lets use the dataset and process the frames
        try:
            # Each iteraction is a full video
            print("obtendo próxima amostra")
            data = next(anomaly_test_iter)
        except:
            print("Acabou o test")
            break


        final_score = calculeAbnormality(data, has_cache, video, cache_folder, DEVICE, buffer_size, reference_frame, OBJECTS_ALLOWED, N, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH)
        pred.extend(final_score.tolist())        
        print("Qtd de frames adicionado no pred: "+str(len(final_score)))
        print("Tem 1 no pred? ")
        print(max(pred))

    print(len(gt_))
    print(len(pred))
    #return final_score

    print("Valores do gt_")
    print(min(gt_))
    print(max(gt_))

    print("Valores do pred")
    print(min(pred))
    print(max(pred))    

    fpr, tpr, threshold = roc_curve(gt_, pred)
    rec_auc = auc(fpr, tpr)

    print("AUC dos anomaly:")
    print(rec_auc)


    ########################################
    # TESTING THE NORMAL
    ########################################

    print("Iniciando o teste DOS NORMALS...")
    cache_folder = "cache/test/normal"
    has_cache = False
    if os.path.exists(os.path.join(FRAMES_DIR, cache_folder)):
        has_cache = True
        anomaly_test_dataset.has_cache = True
    print("has cache? " + str(has_cache))

    normal_test_iter = iter(normal_test_dataset)
    normal_sample_qtd = len(normal_test_iter)
    print(normal_sample_qtd)    

    list_ = os.path.join(ROOT_DIR, "../", "files/graph_detector_test_05s.list")    

    # Recovery a frame-level label
    # Receber isso por parâmetro
    NUM_SAMPLE_FRAME = 15
    LABELS_PATH = os.path.join(DATASET_DIR, "videos/labels/test.txt")
    labels = getLabels(LABELS_PATH, list_) # 2d matrix containing the frame-level frame (columns) for each video (lines)

    #pred = []
    #gt_ = []

    for i in range(normal_sample_qtd):
        video = labels[anomaly_qtd+i]   # we want just the normal
        # Adjust the labels to the truncated frame due computation capability militation
        truncated_frame_qtd = int((max_sample_duration) * NUM_SAMPLE_FRAME)    # The video has max this num of frame
        if len(video[1]) > truncated_frame_qtd:
            video[1] = video[1][0:truncated_frame_qtd]                  # If needed, truncate it
        
        print("Tem 1 nos labels desse video? ")
        print(max(video[1]))            
        
        gt_.extend(video[1])

        print("Qtd de frames adicionado no gt_: "+str(len(video[1])))


        # If folder doesn't exist, lets use the dataset and process the frames
        try:
            # Each iteraction is a full video
            print("obtendo próxima amostra")
            data = next(normal_test_iter)
        except:
            print("Acabou o test")
            break

        final_score = calculeAbnormality(data, has_cache, video, cache_folder, DEVICE, buffer_size, reference_frame, OBJECTS_ALLOWED, N, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH)
        pred.extend(final_score.tolist())        
        print("Qtd de frames adicionado no pred: "+str(len(final_score)))
        print("Tem 1 no pred? ")
        print(max(pred))        

    print(len(gt_))
    print(len(pred))
    #return final_score

    print("Valores do gt_")
    print(min(gt_))
    print(max(gt_))

    print("Valores do pred")
    print(min(pred))
    print(max(pred))    

    fpr, tpr, threshold = roc_curve(gt_, pred)
    rec_auc = auc(fpr, tpr)
    print("Auc do conjunto completo: ")
    print(rec_auc)

    # Lets calculate the AUC for all dataset: 


def measure_divergency(key_frames, edges, GLOBAL_GRAPH, SIMILARITY_THRESHOLD, DEVICE):

    #SIMILARITY_THRESHOLD = 0.57

    VERTEX = GLOBAL_GRAPH[0]

    #start = time.time()
    # Lets map each keyframe to global graph. -1 if there isn't a corresponding keyframe in global graph
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08).to(DEVICE)    

    vertex = torch.FloatTensor(VERTEX)
    
    # We nee to verify if one path have same keyframe in different momments
    
    kf_len = key_frames.shape[0]
    print("Shape do keyframe: " + str(kf_len))
    # If the key_frame is too large, we have to divide the comparision with the GLOBAL_GRAPH, to fit in memmory
    if kf_len <= 20:     # 20 I saw and fits in a 6GB gpu. 
        vec1_ = torch.repeat_interleave(key_frames, vertex.shape[0], dim=0).to(DEVICE)
        vec2_ = vertex.repeat(key_frames.shape[0], 1).to(DEVICE)        
        appea_dist_path = cos(vec1_, vec2_).view(key_frames.shape[0], vertex.shape[0])
    else:
        it = int(kf_len / 20)
        if kf_len % 20 > 0:
            it += 1

        start = 0
        join_list = []
        for i in range(it):
            print("Dividindo a comparação com o grapho. Iteração " + str(i))
            end = 20*(i+1)
            key_frames_tmp = key_frames[start:end]
            vec1_ = torch.repeat_interleave(key_frames_tmp, vertex.shape[0], dim=0).to(DEVICE)
            vec2_ = vertex.repeat(key_frames_tmp.shape[0], 1).to(DEVICE)        
            res = cos(vec1_, vec2_).view(key_frames_tmp.shape[0], vertex.shape[0])
            print(res.shape)
            join_list.append(res)

            start = end
        appea_dist_path = torch.cat(join_list, dim=0)
        

    #end = time.time()
    #print("Tempo decorrido: " + str(end-start))

    # Find the key frame in global graph
    scores = []
    probs = []      # list of [node, probability]
    for kf in range(len(key_frames)):

        # oi is a vector containing the similarity of all nodes in global_graph regarding the node "kf" keyframe
        oi = appea_dist_path[kf,:]

        # Get the similarest node
        imax = torch.argmax(oi)
        max_ = appea_dist_path[kf, imax]
        #print("The similarst node threshold id %s, index %s" % (max_, imax))

        if max_ < SIMILARITY_THRESHOLD:
            #print("Este nó não existe ")
            # Esse nó não tem correspondente no grapho global
            prob_an = 1.     # Probability of be anormal
            probs = []
        else:
            #print("Este nó existe no grafo local")
            # Ok, temos um correspondente no grafo global
        
            # We have to verify what action the person made, and what was the probability of this acrion. We know the probability of all possible normal actions in 'probs'

            # get the nodes
            nodes = []
            if len(probs) > 0:
                probs_ = np.asarray(probs)
                nodes = probs_[:,0].tolist()

            if imax in nodes:   # If this action is linked with the former node
                #print("Este nó é linkado com um nó anterior")
                #print(probs)
                prob = probs[nodes.index(imax)][1]
                # Greater the prob, less anomaly. Lesser the prob, greater the anomaly score
                #print("prob: "+str(prob))
                prob_an = 1 - prob
                #exit()
            else:               # If this action IS NOT linked with former node
                #print("Este nó NÃO é linkado com um nó anterior")
                prob_an = 1.


            # Vamos calcular a probabilidade de cada saída desse nó
            edges_from_kf = edges[kf,:]
            tot_sum = np.sum(edges_from_kf)
            
            # É possível ter um nó sem saída
            if tot_sum == 0:
                continue

            for i, value in enumerate(edges_from_kf):    # i is the value containig in the edges of global_graph
                if value == 0:              # if value == 0, there is no link between two nodes
                    continue
                prob = value / tot_sum      # Never will be a division by 0 because above if
                probs.append([i, float(prob)])    # That means, starting in node 'kf', the probability to go to 'i' is 'prob'
            
        # Each key frame has an anomaly score
        scores.append(prob_an)    

    return scores

    
