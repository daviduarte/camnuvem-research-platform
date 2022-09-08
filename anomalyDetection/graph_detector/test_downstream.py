import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils
import temporalGraph
import os
import cv2
import math

def getFrameQtd(frame_folder_path):
    video_path = frame_folder_path.replace('CamNuvem_dataset_normalizado_frames_05s', 'CamNuvem_dataset_normalizado/videos/samples')
    video_path = video_path.replace('/10', '/10.mp4')

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def getLabels(labels):

    # Colocar isso no config.ini depois
    # TODO
    test_normal_folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/test/normal"
    test_anomaly_folder = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/test/anomaly"

    with open(labels) as file:
        lines = file.readlines()


    gt = []
    qtd_total_frame = 0
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

    print("Qtd total de frame: ")
    print(qtd_total_frame)
    return gt





def test(dataloader, model_pt, model_ds, viz, device, ten_crop, gt_path, OBJECTS_ALLOWED, N_DOWNSTRAM, T, only_abnormal = False):

    dataloader = iter(dataloader)

    # Receber isso por parâmetro
    NUM_SAMPLE_FRAME = 15
    LABELS_PATH = "/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/labels/test.txt"
    labels = getLabels(LABELS_PATH) # 2d matrix containing the frame-level frame (columns) for each video (lines)


    gt = []
    scores = []
    temporal_graph = temporalGraph.TemporalGraph(device, OBJECTS_ALLOWED, N_DOWNSTRAM)    
    with torch.no_grad():
        model_pt.eval()
        model_ds.eval()    

        acc = 0
        for video_index, video in enumerate(labels):    # For each video
            qtdFrame = len(video[1])
            window = 0
            for i, j in enumerate(video[1]):     # For each label in this video
                print("Frame atual:  "+str(acc))
                acc += 1

                # One inference represents T frames
                if window != 0:

                    print(T*NUM_SAMPLE_FRAME-1)
                    print(qtdFrame - T*NUM_SAMPLE_FRAME-1)
                    print(i)
                    if (i > T*NUM_SAMPLE_FRAME-1 and i < (qtdFrame - T*NUM_SAMPLE_FRAME-1)):
                        gt.append(j)
                        scores.append(score)     

                    if window == NUM_SAMPLE_FRAME-1 or i == qtdFrame:
                        window = 0
                    else:
                        window += 1

                    continue

                input= next(dataloader)
                window += 1

                if i < T*NUM_SAMPLE_FRAME-1 or i > (qtdFrame - T*NUM_SAMPLE_FRAME-1):
                    print("skipando")
                    continue                

                # There may exists more .png files than seconds, for some FFMPEG weird reason. 
                while input[3].cpu().flatten() != video_index:
                    print("Há mais png do que frames")
                    print("input: ")
                    print(input[3])
                    print("video index")
                    print(video_index)
                    input = next(dataloader)

                gt.append(j)

                #print(input.shape)

                # Obtain the bounding box from images
                input, box_list_normal = utils.img2bbox(input, temporal_graph, N_DOWNSTRAM)

                # If the ins't any object on scene, we consider as NORMAL
                if input is -1:
                    score = 0.0
                else:
                    input = input.view(1, -1).to(device)
                    box_list_normal = box_list_normal.view(1, -1).to(device)

                    input = torch.cat((input, box_list_normal), dim=1)

                    score = model_ds(model_pt(input))
         
                    score = score.data.cpu().numpy().flatten()[0]     
                
                scores.append(score)     
                
            window = 0      


    print("Dimensões do gt: ")
    print(len(gt))
    print("Dimensões do scores: ")
    print(len(scores))

    exit()



    getFrameQtd()

    #print(gt_path)
    #with open(gt_path,'r') as file:
    #    lines = file.readlines()
    #    print(lines)
    #exit()



    temporal_graph = temporalGraph.TemporalGraph(device, OBJECTS_ALLOWED, N_DOWNSTRAM)    
    with torch.no_grad():
        model_pt.eval()
        model_ds.eval()
        pred_ = []

        gpu_id = 0
        cont = 0
        vai = True
        
        frames_restantes = 0
        id = -1
        for i, input in enumerate(dataloader):

            print("Iteração: " + str(i))
            print("\n")

            print("nooiz")
            print(input[2])

            qtd_total_frame = input[2].cpu()
            id = input[3]
            if vai:
                frames_restantes = qtd_total_frame
                vai = False

            # If the previous sample ID is equal of actual id (input[3])
            # and if there is more .png files than frames in video (beause some strange in ffmpeg)
            # Than we know that we can ignore this sample
            if id == input[3] and frames_restantes <= 0:
                vai = True
                continue

            # Obtain the bounding box from images
            input, box_list_normal = utils.img2bbox(input, temporal_graph, N_DOWNSTRAM)

            # If the ins't any object on scene, we consider as NORMAL
            if input is -1:
                score = 0.0
            else:
                input = input.view(1, -1).to(device)
                box_list_normal = box_list_normal.view(1, -1).to(device)

                input = torch.cat((input, box_list_normal), dim=1)

                score = model_ds(model_pt(input))
     
                score = score.data.cpu().numpy().flatten()[0]

            if frames_restantes < 15 and frames_restantes > 0:
                qtd_to_repeat = frames_restantes
            else:
                qtd_to_repeat = 15

            print("oi :) 00000000000")
            print(type(score))
            print(score)
            print(type(qtd_to_repeat))
            print(qtd_to_repeat)
            score = np.repeat(score, qtd_to_repeat)
            pred_.extend(score)
            cont += 1
            frames_restantes -= 15




        print("Qtd todal de exemplos de testr: ")
        print(cont)
        #exit()

        #if args.dataset == 'shanghai':
        gt = np.load(gt_path)
        #elif args.dataset == 'camnuvem':
        print("Carregando o gt da camnuvem")
        #    gt = np.load('list/gt-camnuvem.npy')
        #else:
        #    gt = np.load('list/gt-ucf.npy')

        print("Quantidade total de frames: ")
        print(len(pred_))        

        print("Quantidde total de frames no arquivo gt: ")
        print(gt.shape)


        #pred = list(pred_.cpu().detach().numpy())
        #pred = np.repeat(np.array(pred), args['segment_size'])


        print("td do pred: ")


        
        gt = list(gt)

        fpr, tpr, threshold = roc_curve(gt, pred_)

        if only_abnormal:            
            np.save('fpr_graph_only_abnormal.npy', fpr)
            np.save('tpr_graph_only_abnormal.npy', tpr)
        else:
            np.save('fpr_graph.npy', fpr)
            np.save('tpr_graph.npy', tpr)

        rec_auc = auc(fpr, tpr)

        print('auc : ' + str(rec_auc))

        best_threshold = threshold[np.argmax(tpr - fpr)]
        print("Best threshold: ")
        print(best_threshold)

        precision, recall, th = precision_recall_curve(list(gt), pred_)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred_)
        viz.lines('roc', tpr, fpr)
        return rec_auc



if __name__ == '__main__':

    gpu_id = 0
    args = {"gt": "list/gt-camnuvem.npy", "segment_size": 16}
    videos_pkl_test = "./arquivos/camnuvem-i3d-test-10crop.list"
    hdf5_path = "./arquivos/data_test.h5" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = 2048
    ten_crop = True

    model = HOE_model(nfeat=features, nclass=1, ten_crop=ten_crop)
    if gpu_id != -1:
        model = model.cuda(gpu_id)    

    test_loader = DataLoader(dataset_h5_test(videos_pkl_test, hdf5_path, ten_crop), pin_memory=False)

    auc = test(test_loader, model, args, device, ten_crop)                                  