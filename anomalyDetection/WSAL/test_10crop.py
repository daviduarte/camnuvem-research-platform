import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from .models import HOE_model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .dataset import dataset_h5_test
import os
import cv2

#DATASET_DIR = "/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado"
DATASET_DIR = "/media/denis/dados/CamNuvem/dataset/ucf_crime_dataset"
#param labels A txt file path containing all test/anomaly frame level labels
#param list A txt file path containing all absolut path of every test file (normal and anomaly)
def getLabels(labels, list_test):

    # Colocar isso no config.ini depois
    # TODO
    test_normal_folder = os.path.join(DATASET_DIR, "videos/samples/test/normal")
    test_anomaly_folder = os.path.join(DATASET_DIR, "videos/samples/test/anomaly")

    #i3d_list_test = list_test.replace("camnuvem-sshc-test", "aux_sshc")
    i3d_list_test = "/media/denis/dados/CamNuvem/pesquisa/anomalyDetection/files/aux_yolov5.list"

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

    lines_i3d = []
    with open(i3d_list_test) as f:
        lines_i3d = f.readlines()
        lines_i3d = [line.strip() for line in lines_i3d]


    list_ = []
    cont = 0
    for i in range(len(lines)):
        path = lines[i]
        
        cont+=1
        if cont <= anomaly_qtd:
            continue
        filename = os.path.basename(path)  
        filee = os.path.join(test_normal_folder, filename[:-4]+'.mp4')
        

        # SSHC WORKAROUND. V
        print(filee)
        if not os.path.exists(filee):
            path = lines_i3d[i]
            filename = os.path.basename(path)  
            filee = os.path.join(test_normal_folder, filename[:-4]+'.mp4')
            print("REntrou aki")
            print(filee)

        list_.append(filee)

    


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


def test(dataloader, model, args, viz, device, ten_crop, gt_path, only_abnormal = False):
    ROOT_DIR = args['root']
    #list_ = os.path.join(ROOT_DIR, "pesquisa/anomalyDetection/files/camnuvem-i3d-ssl-boxes+fea-normalized-test.list")
    list_ = os.path.join(ROOT_DIR, "pesquisa/anomalyDetection/files/ucf-crime-i3d-normalized-test.list")
    LABELS_PATH = os.path.join(DATASET_DIR, "videos/labels/test.txt")
    labels = getLabels(LABELS_PATH, list_) # 2d matrix containing the frame-level frame (columns) for each video (lines)


    #truncated_frame_qtd = int((200) * 75)    # The video has max this num of frame
    truncated_frame_qtd = 100 * 30

    #if len(video[1]) > truncated_frame_qtd:
    #    video[1] = video[1][0:truncated_frame_qtd]                  # If needed, truncate it

    gt = []
    with torch.no_grad():
        model.eval()
        pred_ = torch.zeros(0)
        pred_ = pred_.cpu()


        gpu_id = 0
        cont = 0

        segment_size = args["segment_size"]
        for i, input in enumerate(dataloader):

            video = labels[i]

            feat, pred, vid = input
            #feat = feat[0,:, 10]

            #print("feart antes da variable")
            #print(feat.shape)            
            #if ten_crop:
            #    feat = np.squeeze(feat, axis=0)

            #print("feat antes da variable")
            #print(feat.shape)
            feat, pred = Variable(feat), Variable(pred)
            #print("feat DEPOIS da variable")
            #print(feat.shape)

            feat = feat.to(device)
            pred = pred.to(device)
            #pred = pred.cuda(gpu_id)
            #print(feat.shape)
            #print(pred.shape)
            #exit()
            score, fea = model(feat, device)
            #print("Shape da predicao: ")
            #print(score.shape)
            #print(fea.shape)
            se_score = score.squeeze()  

            # se_score= spatial semantic
            # var = temporal variation

            # Each of the 10 crop going to be a new clip
            if ten_crop == True:
                segments = fea.shape[0]
                crops = fea.shape[1]
                features = fea.shape[2]
                fea = fea.view(segments * crops, features)
                fea = torch.unsqueeze(fea, 0)   # Add a diension at begining. 

            ano_score = torch.zeros_like(fea[0,:,0])
            #print(fea[0,:-1].shape)
            #print(fea[0,1:].shape)
            ano_cos = torch.cosine_similarity(fea[0,:-1], fea[0,1:], dim=1)

            ano_score[:-1] += 1-ano_cos
            ano_score[1:] += 1-ano_cos
            ano_score[1:-1] /= 2
            ano_score_ = ano_score.data.cpu().flatten()
            ano_score = ano_score.data.cpu().numpy().flatten()


            # After each of 10 crop receives a score, lets separate to see what segment bellong to what video
            if ten_crop == True:
                ano_score_ = ano_score_.reshape(segments, crops)
                #print(ano_score_.shape)
                ano_score_ = ano_score_.view(segments, crops).mean(0)
                ano_score = ano_score_.numpy()
                #print(ano_score_.shape)
                #exit()                

            # Isso aqui era pra dar 31 valores
            #print(ano_score_.shape)
            #exit()
            
            se_score = se_score.data.cpu().numpy().flatten()

            #new_pred = ano_score.copy()

            #print(new_pred)

            #print("Shape da predicao depois dos paranaue: ")
            #print(ano_score_.shape)

            print("Shape ano_score: ")
            print(ano_score.shape)
            ano_score_ = ano_score_[:-1]          # Lets ignore the last segment, because it may be a incomplete segment (i.e. a segment made with less than 75 frames). 
            shape_ = ano_score_.shape[0]
            gt_ = video[1][0:(shape_)*16]
            #gt_ = video[1]    
            gt.extend(gt_)

            if ano_score_.shape[0] * 16 != len(gt_):
                print("Erro. A quantidade de labels tem que ser a mesma de samples")
                exit()
            pred_ = torch.cat((pred_, ano_score_))

            #input = input.to(device)
            #input = input.permute(0, 2, 1, 3)
            #score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            #scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            #logits = torch.squeeze(logits, 1)
            #logits = torch.mean(logits, 0)
            #sig = logits
            #pred = torch.cat((pred, sig))

            #print("Shape do logit: ")
            #print(logits.shape)
            #print("shape do input data: ")
            #print(input.shape)

            #exit()
            cont += 1
        #print("Qtd todal de exemplos de testr: ")
        #print(cont)
        #exit()

        #if args.dataset == 'shanghai':
        #gt = np.load(gt_path)
        #elif args.dataset == 'camnuvem':
        #print("Carregando o gt da camnuvem")
        #    gt = np.load('list/gt-camnuvem.npy')
        #else:
        #    gt = np.load('list/gt-ucf.npy')

        print("Quantidade totaol de segmentos de 16: ")
        print(pred_.shape)        

        print("Quantidde total de frames no arquivo gt: ")
        print(len(gt))

        #print(pred_)

        pred = list(pred_.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), args['segment_size'])


        print("td do pred: ")
        print(pred.shape)

        print(list(gt))
        print(pred)
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        if only_abnormal:            
            np.save('fpr_wsal_only_abnormal_camnuvem_10c.npy', fpr)
            np.save('tpr_wsal_only_abnormal_camnuvem_10c.npy', tpr)
        else:
            np.save('fpr_wsal_camnuvem_10c.npy', fpr)
            np.save('tpr_wsal_camnuvem_10c.npy', tpr)

        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        best_threshold = threshold[np.argmax(tpr - fpr)]
        print("Best threshold: ")
        print(best_threshold)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        try:
            if only_abnormal == False:
                viz.plot_lines('pr_auc', pr_auc)
                viz.plot_lines('auc', rec_auc)
                viz.lines('scores', pred)
                viz.lines('roc', tpr, fpr)
            else:
                viz.plot_lines('auc only abnormal', rec_auc)                
        except:
            print("Viz desligado")

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
