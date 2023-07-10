from __future__ import print_function
from __future__ import division
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.nn.functional import sigmoid
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.nn import SmoothL1Loss
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import sys
#sys.path.append("/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/WSAL")
from .models import HOE_model
from .dataset import dataset_h5
from .utils import *
from .utils import save_best_record
import pdb
import os
import time
from torch import nn
from .utils import Visualizer
from .test_10crop import test
from .dataset import dataset_h5_test

viz = Visualizer(env='WSAL i3d 10crop', use_incoming_socket=False)


def train_wsal(videos_pkl_train, videos_pkl_test, hdf5_path_train, hdf5_path_test, gt, segment_size, root, ten_crop, gpu_id, checkpoint, gt_only_anomaly):
    print("********************************")
    torch.cuda.empty_cache()

    ver ='WSAL'

    args = {"gt": gt, "segment_size": segment_size, "root": root}

    #mask_path = "./arquivos/mask.h5"

    #hdf5_path = "/test/UCF-Crime/UCF/gcn_feas.hdf5" 

    #mask_path = "/test/UCF-Crime/UCF/gcn_mask.hdf5"

    modality = "rgb"
    batch_size = 30
    iter_size = 30//batch_size
    random_crop = False
    #ten_crop = True
    features = 2048
    #features = 174
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device(gpu_id)

    train_loader = torch.utils.data.DataLoader(dataset_h5(videos_pkl_train, hdf5_path_train, ten_crop),
                                                    batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

    test_loader = DataLoader(dataset_h5_test(videos_pkl_test, hdf5_path_test, ten_crop), pin_memory=False)    

    test_loader_only_anomaly = DataLoader(dataset_h5_test(videos_pkl_test, hdf5_path_test, ten_crop, only_anomaly=True), pin_memory=False)

    model = HOE_model(nfeat=features, nclass=1, ten_crop = ten_crop)
    model = model.to(device)

    print(checkpoint)
    print("mano")

    if checkpoint != "False":    
        checkpoint = torch.load(checkpoint, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("********************************88")
        auc1 = test(test_loader, model, args, viz, device, ten_crop, gt)

        auc2 = test(test_loader_only_anomaly, model, args, viz, device, ten_crop, gt_only_anomaly, only_abnormal=True)
        
        print(auc1)
        print(auc2)
        exit()

    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    Rcriterion = torch.nn.MarginRankingLoss(margin=1.0, reduction = 'mean')

    #if gpu_id != -1:

    #    model = model.cuda(gpu_id)
    model.to(device)
        #criterion = criterion.cuda(gpu_id)
    criterion.to(device)
        #Rcriterion = Rcriterion.cuda(gpu_id)
    Rcriterion.to(device)
        #test_loader = test_loader.cuda(gpu_id)
        #train_loader = train_loader.cuda(gpu_id)

    optimizer = optim.Adagrad(model.parameters(), lr=0.0001) # original é 0.001
    opt_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400], gamma=0.5)
    start_epoch = 0
        
    resume = './weights/WSAL_1.1/rgb_300.pth'
    if os.path.isfile(resume):
      print("=> loading checkpoint '{}'".format(resume))
      checkpoint = torch.load(resume)
      start_epoch = checkpoint['epoch']
      opt_scheduler.load_state_dict(checkpoint['scheduler'])
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])

    iter_count = 0
    alpha = 0.5
    vid2mean_pred = {}
    losses = AverageMeter()
    data_time = AverageMeter()
    model.train()

    # Before training let's get the AUC
    best_auc = 0
    auc = test(test_loader, model, args, viz, device, ten_crop, gt)
    auc2 = test(test_loader_only_anomaly, model, args, viz, device, ten_crop, gt_only_anomaly, only_abnormal=True)
    for epoch in range(start_epoch, 75):
       
        end = time.time()
        pbar = tqdm(total=len(train_loader))
        for step, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            an_feats, no_feats, preds = data

            #print("Dimensão do ano_feas: ")
            #print(an_feats.shape)

            #print("Dimensão do nor_feas: ")
            #print(no_feats.shape)        

            #print("Dimensão do pred: ")
            #print(preds.shape)

            an_feat, no_feat, pred = Variable(an_feats), Variable(no_feats), Variable(preds)
            
            #if gpu_id != -1:
                #an_feat = an_feat.cuda(gpu_id)
            an_feat.to(device)
                #no_feat = no_feat.cuda(gpu_id)
            no_feat.to(device)
                #pred = pred.cuda(gpu_id).float()
            pred = pred.to(device)

            if iter_count % iter_size == 0:
                optimizer.zero_grad()

            if ten_crop:
                #pred = pred.reshape(pred.shape[0] * pred.shape[1])
                pred = pred[:,0:10].reshape(10*batch_size)
                #print("Novo shape do pred: ")
                #print(pred.shape)
                an_feat = np.squeeze(an_feat, axis=0)
                no_feat = np.squeeze(no_feat, axis=0)
                #print("Ten crop ")
                #print(an_feat.shape)

            print("\n\nan_feat: ")
            print(an_feat.shape)
            print(torch.max(an_feat))
            print(torch.min(an_feat))
            print(torch.mean(an_feat))
            ano_ss, ano_fea = model(an_feat, device)
            nor_ss, nor_fea = model(no_feat, device)

            

            ano_cos = torch.cosine_similarity(ano_fea[:,1:], ano_fea[:,:-1], dim=2)
            
            dynamic_score_ano = 1-ano_cos
            nor_cos = torch.cosine_similarity(nor_fea[:,1:], nor_fea[:,:-1], dim=2)
            dynamic_score_nor = 1-nor_cos
            
            ano_max = torch.max(dynamic_score_ano,1)[0].to(device)
            nor_max = torch.max(dynamic_score_nor,1)[0].to(device)

            #print("Shape das predicoes: ")
            #print(ano_max.shape)
            #print(nor_max.shape)
            #print("Shape dos labels: ")
            #print(pred.shape)
            #print(pred)

            # pred[:,0] is all anormal labels from batch

            if ten_crop:
                loss_dy = Rcriterion(ano_max, nor_max, pred)
            else:
                loss_dy = Rcriterion(ano_max, nor_max, pred[:,0])
            
            semantic_margin_ano = torch.max(ano_ss,1)[0]-torch.min(ano_ss,1)[0]
            semantic_margin_nor = torch.max(nor_ss,1)[0]-torch.min(nor_ss,1)[0]


            if ten_crop:
                loss_se = Rcriterion(semantic_margin_ano, semantic_margin_nor, pred)    
            else:
                loss_se = Rcriterion(semantic_margin_ano, semantic_margin_nor, pred[:,0])

            loss_3 = torch.mean(torch.sum(dynamic_score_ano,1))+torch.mean(torch.sum(dynamic_score_nor,1))+torch.mean(torch.sum(ano_ss,1))+torch.mean(torch.sum(nor_ss,1))
            loss_5 = torch.mean(torch.sum((dynamic_score_ano[:,:-1]-dynamic_score_ano[:,1:])**2,1))+torch.mean(torch.sum((ano_ss[:,:-1]-ano_ss[:,1:])**2,1))

            loss_train = loss_se + loss_dy+ loss_3*0.00008+ loss_5*0.00008 

            
            iter_count += 1
            loss_train.backward()
            losses.update(loss_train.item(), 1)

            if (iter_count + 1) % iter_size == 0:
                optimizer.step()

            pbar.set_postfix({
                    'Data': '{data_time.val:.3f}({data_time.avg:.4f})\t'.format(data_time=data_time),
                    ver: '{0}'.format(epoch),
                    'lr': '{lr:.5f}\t'.format(lr=optimizer.param_groups[-1]['lr']),
                    'Loss': '{loss.val:.4f}({loss.avg:.4f})\t '.format(loss=losses)
                    })
            
            pbar.update(1)

        pbar.close()
        model_path = os.path.join(root, 'pesquisa/anomalyDetection/WSAL/weights/'+ver+'/')
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        if epoch%5==0:
            
            auc = test(test_loader, model, args, viz, device, ten_crop, gt)
            auc2 = test(test_loader_only_anomaly, model, args, viz, device, ten_crop, gt_only_anomaly, only_abnormal=True)

            state = {
              'epoch': epoch,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'scheduler': opt_scheduler.state_dict(),
            }
            if auc > best_auc:
                best_auc = auc
                torch.save(state, model_path+"rgb_%d.pth" % epoch)
                save_best_record(best_auc, auc2, epoch, model_path+"rgb_%d.txt" % epoch)

            # model = model.cuda(gpu_id)
        # if epoch%25==0:
        losses.reset()
        opt_scheduler.step()


if __name__ == '__main__':

    videos_pkl_train = "./arquivos/camnuvem-i3d-train-10crop.list"
    videos_pkl_test = "./arquivos/camnuvem-i3d-test-10crop.list"
    gt = "list/gt-camnuvem.npy"

    hdf5_path_test = "./arquivos/data_test.h5" 
    hdf5_path_train = "./arquivos/data_train.h5"

    root = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/WSAL"

    segment_size = 16
    train_wsal(videos_pkl_train, videos_pkl_test, hdf5_path_train, hdf5_path_test, gt, 16, root)

