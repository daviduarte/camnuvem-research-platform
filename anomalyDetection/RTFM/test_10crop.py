import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
#torch.set_default_tensor_type('torch.FloatTensor')

def test(dataloader, model, args, viz, device, gt, only_abnormal = False):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        pred = pred.cpu().detach()
        device = 'cpu'
        model.device = 'cpu'
        cont = 0

        for i, input in enumerate(dataloader):

            input = input.to(device)
            input = input.permute(0, 2, 1, 3)

            model = model.to(device)

            #print("Valorr do input antes da rede")
            #print(input.shape)
            #print(input.type())

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            
            print(logits)
            #exit()
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            sig = logits
            sig = sig.cpu().detach()
            #print(sig)
            pred = torch.cat((pred, sig))


            print("Preeed")
            print(pred)

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
        #print("Qtd todal de exemplos de testr: ")
        #print(cont)
        #exit()

        #if args.dataset == 'shanghai':
        #print(gt)
        gt = np.load(gt)
        #elif args.dataset == 'camnuvem':
        #print("Carregando o gt da camnuvem")
        #    gt = np.load('list/gt-camnuvem.npy')
        #else:
        #    gt = np.load('list/gt-ucf.npy')

        #print("Quantidade totaol de segmentos de 16: ")
        #print(pred.shape)        

        #print("Quantidde total de frames no arquivo gt: ")
        #print(gt.shape)

        #print(pred)
        #exit()

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), args.segment_size)

        print(pred)
        #print("td do pred: ")
        #print(pred.shape)

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

