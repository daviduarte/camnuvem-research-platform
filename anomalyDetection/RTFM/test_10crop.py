import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import os
import cv2
#torch.set_default_tensor_type('torch.FloatTensor')

DATASET_DIR = "/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado"
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


def test(dataloader, model, args, viz, device, _, only_abnormal = False):
    ROOT_DIR = args.root
    list_ = os.path.join(ROOT_DIR, "pesquisa/anomalyDetection/files/camnuvem-i3d-ssl-normalized-test.list")
    LABELS_PATH = os.path.join(DATASET_DIR, "videos/labels/test.txt")
    labels = getLabels(LABELS_PATH, list_) # 2d matrix containing the frame-level frame (columns) for each video (lines)
    
    #truncated_frame_qtd = int((200) * 75)    # The video has max this num of frame
    truncated_frame_qtd = 100 * 30

    # os labels tem que estar na mesma ordem dos arquivos nessa pasta: 
    #/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/test/anomaly
    dirs = os.listdir("/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/test/anomaly")

    gt = []
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        pred = pred.cpu().detach()
        device = 'cpu'
        model.device = 'cpu'
        cont = 0

        video_index = -1
        cont2 = 0
        #len_video = 0

        print(args)
        segment_size = args.segment_size
        truncated_frame_qtd = 200 * segment_size
        for i, input in enumerate(dataloader):

            #if cont2 == len_video:
            #video_index += 1
            video = labels[i]

            #if len(video[1]) > truncated_frame_qtd:
            #    video[1] = video[1][0:truncated_frame_qtd]                  # If needed, truncate it
            #len_video = len(video[1])   # Valor de amostras, já cortado

            input = input.to(device)
            input = input.permute(0, 2, 1, 3)

            model = model.to(device)

            #print("Valorr do input antes da rede")
            #print(input.shape)
            #print(input.type())

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            
            #print(logits)
            #exit()
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            sig = sig.cpu().detach()
            print(video[0])
            sig = sig[:-1]          # Lets ignore the last segment, because it may be a incomplete segment (i.e. a segment made with less than 75 frames). 
            shape_ = sig.shape[0]
            print(len(video[1]))
            gt_ = video[1][0:(shape_)*16]
            #gt_ = video[1]    
            gt.extend(gt_)

            print(sig.shape)
            print(len(gt_))
            if sig.shape[0] * 16 != len(gt_):
                print("Erro. A quantidade de labels tem que ser a mesma de samples")
                exit()
            pred = torch.cat((pred, sig))

            with open("vis/"+str(cont)+".txt", 'w') as file:
                for i in pred:
                    file.write(str(i)+"")            

            #print("Shape do logit: ")
            #print(logits.shape)
            #print("shape do input data: ")
            #print(input.shape)

            #exit()
            cont += 1
            torch.cuda.empty_cache()

            cont2 += 16
        #print("Qtd todal de exemplos de testr: ")
        #print(cont)
        #exit()

        #if args.dataset == 'shanghai':
        #print(gt)
        #gt = np.load(gt)
        #elif args.dataset == 'camnuvem':
        #print("Carregando o gt da camnuvem")
        #    gt = np.load('list/gt-camnuvem.npy')
        #else:
        #    gt = np.load('list/gt-ucf.npy')

        #print(pred)
        #exit()

        pred = list(pred.cpu().detach().numpy())

        pred = np.repeat(np.array(pred), args.segment_size)


        fpr, tpr, threshold = roc_curve(list(gt), pred)

        if only_abnormal:            
            np.save('fpr_rtfm_only_abnormal_ucf_10c.npy', fpr)
            np.save('tpr_rtfm_only_abnormal_ucf_10c.npy', tpr)
        else:
            np.save('fpr_rtfm_ucf_10c.npy', fpr)
            np.save('tpr_rtfm_ucf_10c.npy', tpr)

        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        best_threshold = threshold[np.argmax(tpr - fpr)]
        #print("Best threshold: ")
        #print(best_threshold)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)

        #device = 'cuda:0'
        #model.device = 'cuda:0'
        #model.to(device)

        return rec_auc

