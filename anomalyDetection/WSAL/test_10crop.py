import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from .models import HOE_model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .dataset import dataset_h5_test

DATASET_DIR = "/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado"
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


def test(dataloader, model, args, viz, device, ten_crop, gt_path, only_abnormal = False):
    ROOT_DIR = args.root
    list_ = os.path.join(ROOT_DIR, "../", "files/graph_detector_test_05s.list")
    LABELS_PATH = os.path.join(DATASET_DIR, "videos/labels/test.txt")
    labels = getLabels(LABELS_PATH, list_) # 2d matrix containing the frame-level frame (columns) for each video (lines)
    print(labels)
    exit()

    truncated_frame_qtd = int((max_sample_duration) * NUM_SAMPLE_FRAME)    # The video has max this num of frame
    if len(video[1]) > truncated_frame_qtd:
        video[1] = video[1][0:truncated_frame_qtd]                  # If needed, truncate it

    with torch.no_grad():
        model.eval()
        pred_ = torch.zeros(0)
        pred_ = pred_.cpu()


        gpu_id = 0
        cont = 0

        for i, input in enumerate(dataloader):

            feat, pred, vid = input
            #feat = feat[0,:, 10]

            print("feart antes da variable")
            print(feat.shape)            
            #if ten_crop:
            #    feat = np.squeeze(feat, axis=0)

            print("feat antes da variable")
            print(feat.shape)
            feat, pred = Variable(feat), Variable(pred)
            print("feat DEPOIS da variable")
            print(feat.shape)

            feat = feat.to(device)
            pred = pred.to(device)
            #pred = pred.cuda(gpu_id)
            print(feat.shape)
            #print(pred.shape)
            #exit()
            score, fea = model(feat, device)
            print("Shape da predicao: ")
            print(score.shape)
            print(fea.shape)
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
            print(fea[0,:-1].shape)
            print(fea[0,1:].shape)
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

            print("Shape da predicao depois dos paranaue: ")
            print(ano_score_.shape)


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

        print("Quantidade totaol de segmentos de 16: ")
        print(pred_.shape)        

        print("Quantidde total de frames no arquivo gt: ")
        print(gt.shape)

        print(pred_)

        pred = list(pred_.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), args['segment_size'])


        print("td do pred: ")
        print(pred.shape)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        if only_abnormal:            
            np.save('fpr_wsal_only_abnormal_ucf_10c.npy', fpr)
            np.save('tpr_wsal_only_abnormal_ucf_10c.npy', tpr)
        else:
            np.save('fpr_wsal_ucf_10c.npy', fpr)
            np.save('tpr_wsal_ucf_10c.npy', tpr)

        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        best_threshold = threshold[np.argmax(tpr - fpr)]
        print("Best threshold: ")
        print(best_threshold)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
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