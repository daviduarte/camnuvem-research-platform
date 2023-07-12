from torch.utils.data import DataLoader
from .learner import Learner
from .loss import *
from .dataset import *
import os
from sklearn import metrics
import time

TEN_CROP = False
BATCH_SIZE = 30
SEGMENTS_NUM = 32



def train(epoch, model, anomaly_train_loader, normal_train_loader, device, criterion, optimizer, scheduler):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    zip_ = zip(normal_train_loader, anomaly_train_loader)

    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip_):
        #print(normal_inputs.shape)
        #print(anomaly_inputs.shape)
        
        start = time.time()
        if TEN_CROP:
            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=2).to(device)
        else:
            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1).to(device)


        
        batch_size = inputs.shape[0]
        if TEN_CROP:
            bs, _, segments, fea_size = inputs.shape
        else:
            bs, segments, fea_size = inputs.shape       

        inputs = inputs.view(-1, inputs.size(-1)).to(device)

        
        
        outputs = model(inputs, bs, segments)

        #print("Dim do output")
        #print(outputs.shape)

        loss = criterion(outputs, batch_size).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        end = time.time()
        print("Tempo da ifnerenciA: ")
        print(end-start)        
    print('loss = {}', train_loss/len(normal_train_loader))
    scheduler.step()

def test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, checkpoint = -1):
    if checkpoint != -1:
        print("CARREGANDO O MODELO")
        model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))    
    
    auc = 0
    with torch.no_grad():
        predictions = []
        gt = []        
        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames = data

            if TEN_CROP:
                bs, _, segments, fea_size = inputs.shape
            else:
                bs, segments, fea_size = inputs.shape
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))


        
            score = model(inputs, bs, segments)
    
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, int(frames[0]/16), 33))

            #print(np.max(score))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]
            predictions.extend(score_list)

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1
            gt.extend(gt_list)


        for i, data2 in enumerate(normal_test_loader):
            inputs2, gts2, frames2 = data2
            if TEN_CROP:
                bs, _, segments, fea_size = inputs2.shape
            else:
                bs, segments, fea_size = inputs2.shape            
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2 = model(inputs2, bs, segments)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, int(frames2[0]/16), 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            predictions.extend(score_list2)

            gt_list2 = np.zeros(frames2[0])
            gt.extend(gt_list2)

            #score_list3 = np.concatenate((score_list, score_list2), axis=0)
            #gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(gt, predictions, pos_label=1)
        print(fpr)
        name = ""
        if TEN_CROP:
            name = "_10c"          
        np.save("anomalyDetection/RADS/fpr_camnuvem_sultani"+name+".npy", fpr)
        np.save("anomalyDetection/RADS/tpr_camnuvem_sultani"+name+".npy", tpr)
        auc += metrics.auc(fpr, tpr)
        #print('auc = ', auc/140)
        #print(auc)
        auc = auc
        print('auc = ', auc)
        return auc, fpr, tpr

def test_abnormal_anomaly_only(epoch, model, anomaly_test_loader, checkpoint = -1):
    
    if checkpoint != -1:
        print("CARREGANDO O MODELO")
        model.load_state_dict(torch.load(checkpoint))
    auc = 0

    model = model.to(torch.device('cuda'))

    with torch.no_grad():
        predictions = []
        gt = []

        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames = data

            if TEN_CROP:
                bs, _, segments, fea_size = inputs.shape
            else:
                bs, segments, fea_size = inputs.shape

            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))

            score = model(inputs, bs, segments)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, int(frames[0]/16), 33))

            #print(np.max(score))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]
            predictions.extend(score_list)

            gt_list = np.zeros(frames[0])

            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1
            gt.extend(gt_list)

            #inputs2, gts2, frames2 = data2
            #inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            #score2 = model(inputs2)
            #score2 = score2.cpu().detach().numpy()
            #score_list2 = np.zeros(frames2[0])
            #step2 = np.round(np.linspace(0, int(frames2[0]/16), 33))

            #for kk in range(32):
            #    score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            #gt_list2 = np.zeros(frames2[0])


            #score_list3 = np.concatenate((score_list, score_list2), axis=0)
            #gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            #fpr, tpr, thresholds = metrics.roc_curve(gt_list, score_list, pos_label=1)

            #auc += metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(gt, predictions, pos_label=1)  
        name = ""
        if TEN_CROP:
            name = "_10c"          
        np.save("anomalyDetection/RADS/fpr_camnuvem_sultani_only_abnormal"+name+".npy", fpr)
        np.save("anomalyDetection/RADS/tpr_camnuvem_sultani_only_abnormal"+name+".npy", tpr)        
        auc += metrics.auc(fpr, tpr)        
        #print('auc = ', auc/140)
        #print(auc)
        auc = auc

        print('auc = ', auc)
        return auc, fpr, tpr        

def main(training_list_file_final_name, test_list_file_final_name, num_frames_in_each_feature_vector, root, crop_10, gpu_id, checkpoint, datasetInfos, feature_extractor):


    dataset_path  = os.path.join(root, "dataset", datasetInfos[2], feature_extractor)
    is_ucf = (datasetInfos[2] == 'ucf_crime_dataset')
    normal_train_dataset = Normal_Loader(dataset_path, is_train=1, ten_crop = TEN_CROP, is_ucf = is_ucf)
    normal_test_dataset = Normal_Loader(dataset_path, is_train=0, ten_crop = TEN_CROP, is_ucf = is_ucf)

    anomaly_train_dataset = Anomaly_Loader(dataset_path, is_train=1, ten_crop = TEN_CROP, is_ucf = is_ucf)
    anomaly_test_dataset = Anomaly_Loader(dataset_path, is_train=0, ten_crop = TEN_CROP, is_ucf = is_ucf)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=gpu_id))
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device=gpu_id))

    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator(device=gpu_id)) 
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device=gpu_id))



    model = Learner(input_dim=2048, drop_p=0.0, ten_crop = TEN_CROP).to(gpu_id)
    #                                                   lr original: 0.001
    optimizer = torch.optim.Adagrad(model.parameters(), lr= 0.1, weight_decay=0.0010000000474974513)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    criterion = MIL


    best_auc = float("-Inf")

    if checkpoint != "False":    
        model.eval()
        auc = test_abnormal(0, model, anomaly_test_loader, normal_test_loader, checkpoint)
        auc = test_abnormal_anomaly_only(0, model, anomaly_test_loader, checkpoint)
        exit()

    epoch = 0
    model.eval()
    model = model.cuda()

    for epoch in range(0, 75):
        train(epoch, model, anomaly_train_loader, normal_train_loader, gpu_id, criterion, optimizer, scheduler)

        model.eval()
        auc_ab, fpr_ab, tpr_ab = test_abnormal_anomaly_only(epoch, model, anomaly_test_loader)
        auc, fpr, tpr = test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "anomalyDetection/RADS/checkpoints/checkpoint-epoch-"+str(epoch)+".pth")

            with open("anomalyDetection/RADS/checkpoints/checkpoint-epoch-"+str(epoch)+".txt", 'w') as file:
                file.write("AUC: " + str(best_auc))



