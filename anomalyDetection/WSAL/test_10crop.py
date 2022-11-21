import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from .models import HOE_model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .dataset import dataset_h5_test

def test(dataloader, model, args, viz, device, ten_crop, gt_path, only_abnormal = False):
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