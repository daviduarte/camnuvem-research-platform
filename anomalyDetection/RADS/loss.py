import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, is_transformer=0):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    #print(y_pred)
    
    
    for i in range(batch_size):
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()

        y_anomaly = y_pred[i, :32][anomaly_index] # Permute the 32 random segment from  anomaly video i
        y_normal  = y_pred[i, 32:][normal_index]

        y_anomaly_max = torch.max(y_anomaly) # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal) # normal
        y_normal_min = torch.min(y_normal)

        #print("Batch " + str(i))
        #print(" maor anomaly: ")
        #print(y_anomaly_max)
        #print(" maior normal: ")
        #print(str(y_normal_max) )

        #print(y_anomaly_max)
        #print(y_normal_max)
        
        loss += F.relu(1.-y_anomaly_max+y_normal_max)
        #print(loss)
        #print("\n")

        #print(y_anomaly_max)
        #print(y_anomaly_min)
        #print(y_normal_max)
        #print(y_normal_min)

        #print(loss)


        sparsity += torch.sum(y_anomaly)*0.00008
        #print(y_pred[i,1:32])
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008

    #print("loss: ")
    #print(loss)

    #print("sparsity: ")
    #print(sparsity)

    #print("smooth: ")
    #print(smooth)
    #print("\n\n")
    loss = (loss+sparsity+smooth)/batch_size

    return loss
